"""
dataset_registry.py — 数据集注册表
===================================

职责：
  1. 根据文件内容嗅探 kind：eval_csv / train_manifest
  2. 扫描某个目录（默认 DATASET_BASE_DIR），把识别到的文件 upsert 进 datasets 表
  3. 提供单条数据集的预览（前 N 行）

设计原则：
  - 一条记录 = 一个文件（不绑定 train+test 对）
  - 扫描失败/格式不对的文件静默跳过，但不让一个坏文件搞挂整次 scan
  - 目录不存在就当空扫描，不主动建目录
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

from sqlalchemy.orm import Session

from database import Dataset
from dataset_loader import DATASET_BASE_DIR, REF_COLUMNS, HYP_COLUMNS


# ──────────────────────────────────────
# 嗅探
# ──────────────────────────────────────

KIND_EVAL_CSV = "eval_csv"
KIND_TRAIN_MANIFEST = "train_manifest"

# 扫描时跳过的常见"垃圾"目录
_SKIP_DIR_NAMES = {
    "__pycache__", ".git", ".cache", "downloads",
    "downloads_extracted", "locks", ".huggingface",
    ".ipynb_checkpoints", "node_modules",
}


@dataclass
class ProbeResult:
    kind: str
    rows: Optional[int]
    duration_sec: Optional[float]


def _probe_eval_csv(path: Path) -> Optional[ProbeResult]:
    """
    判断是否是 NewEval 可用的 CSV。

    条件：表头包含 REF_COLUMNS 中任意一列 AND HYP_COLUMNS 中任意一列。
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header:
                return None
            lower = {(col or "").strip().lower() for col in header}
            if not (lower & REF_COLUMNS) or not (lower & HYP_COLUMNS):
                return None
            # 有匹配列才数行数（避免对无关 CSV 花时间）
            rows = sum(1 for _ in reader)
        return ProbeResult(kind=KIND_EVAL_CSV, rows=rows, duration_sec=None)
    except Exception:
        return None


def _probe_train_manifest(path: Path) -> Optional[ProbeResult]:
    """
    判断是否是 Whisper-Finetune 用的 JSONL manifest。

    条件：第一行是 JSON object，至少含 `duration`，且含 `sentence` 或 `sentences`。
    数 rows 时顺便把 duration 累加，方便前端显示总时长。
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            first = ""
            for line in fh:
                if line.strip():
                    first = line.strip()
                    break
            if not first:
                return None
            obj = json.loads(first)
            if not isinstance(obj, dict):
                return None
            if "duration" not in obj:
                return None
            if "sentence" not in obj and "sentences" not in obj:
                return None

        # 确认是 manifest 后再完整数一遍
        rows = 0
        total_duration = 0.0
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                rows += 1
                d = item.get("duration")
                if isinstance(d, (int, float)):
                    total_duration += float(d)

        return ProbeResult(
            kind=KIND_TRAIN_MANIFEST,
            rows=rows,
            duration_sec=round(total_duration, 2) if total_duration > 0 else None,
        )
    except Exception:
        return None


def probe_file(path: Path) -> Optional[ProbeResult]:
    """根据后缀先粗筛，再调对应的 _probe_*。"""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _probe_eval_csv(path)
    if suffix in {".json", ".jsonl"}:
        return _probe_train_manifest(path)
    return None


# ──────────────────────────────────────
# 扫描 + upsert
# ──────────────────────────────────────

def _iter_files(root: Path) -> Iterator[Path]:
    """递归遍历，跳过常见无用目录。"""
    if not root.exists():
        return
    for cur, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".")
            and d not in _SKIP_DIR_NAMES
        ]
        cur_path = Path(cur)
        for name in filenames:
            if name.startswith("."):
                continue
            yield cur_path / name


def _default_name(path: Path, base: Path) -> str:
    """
    name 优先用相对 base 的路径（少几层更可读），fallback 到文件名。
    例子：base=/data/datasets, path=/data/datasets/mn/train.json → 'mn/train.json'
    """
    try:
        rel = path.relative_to(base)
        return str(rel)
    except ValueError:
        return path.name


def scan_and_upsert(
    db: Session,
    base_dir: Optional[str] = None,
    source: str = "local",
    source_repo: Optional[str] = None,
) -> Tuple[int, int, int, int]:
    """
    执行一次扫描并把结果写入数据库。

    Args:
        db: 已有的 SQLAlchemy session
        base_dir: 要扫的根目录，默认 DATASET_BASE_DIR
        source: 写入 Dataset.source。从 HF 拉取后 scan 时传 'huggingface'
        source_repo: 如果本次 scan 来自一次 HF pull，传仓库 ID

    Returns:
        (scanned, added, updated, removed)
          scanned — 本次走过的候选文件数
          added   — 新插入的行数
          updated — 路径已存在、元信息被更新的行数
          removed — 原本 ready、但现在文件不见了 → 标为 missing
    """
    root = Path(base_dir or DATASET_BASE_DIR)
    scanned = 0
    added = 0
    updated = 0

    # 缓存本次扫描涉及到的路径，方便之后做 missing 标记
    seen_paths: set[str] = set()

    for file_path in _iter_files(root):
        scanned += 1
        result = probe_file(file_path)
        if result is None:
            continue

        abs_path = str(file_path.resolve())
        seen_paths.add(abs_path)

        size = file_path.stat().st_size

        existing = db.query(Dataset).filter(Dataset.path == abs_path).first()
        if existing is None:
            ds = Dataset(
                name=_default_name(file_path, root),
                kind=result.kind,
                path=abs_path,
                rows=result.rows,
                size_bytes=size,
                duration_sec=result.duration_sec,
                source=source,
                source_repo=source_repo,
                status="ready",
            )
            db.add(ds)
            added += 1
        else:
            changed = False
            for field, value in {
                "kind": result.kind,
                "rows": result.rows,
                "size_bytes": size,
                "duration_sec": result.duration_sec,
                "status": "ready",
            }.items():
                if getattr(existing, field) != value:
                    setattr(existing, field, value)
                    changed = True
            # source_repo 只在从 HF 拉取过来时覆盖，本地扫描不要把它清掉
            if source_repo and existing.source_repo != source_repo:
                existing.source_repo = source_repo
                existing.source = source
                changed = True
            if changed:
                existing.updated_at = datetime.utcnow()
                updated += 1

    # 标 missing：路径在扫描根目录下、但文件不在了
    removed = 0
    root_abs = str(root.resolve())
    candidates = (
        db.query(Dataset)
        .filter(Dataset.status == "ready")
        .filter(Dataset.path.like(f"{root_abs}%"))
        .all()
    )
    for ds in candidates:
        if not os.path.exists(ds.path):
            ds.status = "missing"
            ds.updated_at = datetime.utcnow()
            removed += 1

    db.commit()
    return scanned, added, updated, removed


# ──────────────────────────────────────
# 预览
# ──────────────────────────────────────

def preview_dataset(ds: Dataset, n: int = 5) -> dict:
    """
    读数据集的前 N 行返回 dict 结构：
      { "kind": "...", "columns": [...], "rows": [...] }

    - eval_csv：csv.DictReader 取前 N 行
    - train_manifest：读前 N 个 json 行
    """
    path = Path(ds.path)
    if not path.exists():
        return {"kind": ds.kind, "columns": None, "rows": []}

    if ds.kind == KIND_EVAL_CSV:
        out_rows: List[dict] = []
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            columns = list(reader.fieldnames or [])
            for i, row in enumerate(reader):
                if i >= n:
                    break
                out_rows.append(dict(row))
        return {"kind": ds.kind, "columns": columns, "rows": out_rows}

    if ds.kind == KIND_TRAIN_MANIFEST:
        out_rows = []
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(out_rows) >= n:
                    break
        # columns：用第一行的 key 粗略代表
        columns = list(out_rows[0].keys()) if out_rows else []
        return {"kind": ds.kind, "columns": columns, "rows": out_rows}

    return {"kind": ds.kind, "columns": None, "rows": []}
