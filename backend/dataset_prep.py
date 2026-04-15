"""
dataset_prep.py — 数据集预处理
=================================

目前只实现 Common Voice 风格目录的准备工作：

    <base>/
        audio/{lang}/{split}/*.tar         ← 音频压缩包
        transcript/{lang}/{split}.tsv      ← 转写 (client_id, path, sentence, ...)
        transcript/{lang}/clip_durations.tsv

产出：

    <base>/manifests/{lang}/{split}.jsonl

每行 JSON：
    {"audio": {"path": "<abs mp3>"}, "sentence": "...", "duration": <sec>}

满足 Whisper-Finetune / ASRGenericDataset 的标准格式。
"""

from __future__ import annotations

import csv
import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ──────────────────────────────────────
# 探测：给定目录里有哪些语言 / 哪些 split
# ──────────────────────────────────────

# CV 里常见的 split 文件名（除这些以外还可能有 reported / unvalidated_sentences 等，
# 那些不是 path+sentence 的标准行，这里不处理）
_STANDARD_SPLITS = {"train", "test", "dev", "validated", "invalidated", "other"}


@dataclass
class CVSplit:
    name: str                    # 'train' / 'test' / ...
    tsv_path: str                # 绝对路径
    tar_paths: List[str] = field(default_factory=list)
    rows: Optional[int] = None


@dataclass
class CVLanguage:
    lang: str
    transcript_dir: str
    audio_dir: str
    splits: List[CVSplit] = field(default_factory=list)
    has_clip_durations: bool = False


@dataclass
class CVProbe:
    base_dir: str
    languages: List[CVLanguage] = field(default_factory=list)

    @property
    def is_cv(self) -> bool:
        return bool(self.languages)


def probe_cv_layout(base_dir: str) -> CVProbe:
    """
    检查 base_dir 是否是 Common Voice 风格的布局，
    返回可处理的语言/split 列表。
    """
    base = Path(base_dir)
    probe = CVProbe(base_dir=str(base))

    transcript_root = base / "transcript"
    audio_root = base / "audio"
    if not transcript_root.is_dir() or not audio_root.is_dir():
        return probe

    for lang_dir in sorted(transcript_root.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name

        audio_lang = audio_root / lang
        if not audio_lang.is_dir():
            continue

        cv_lang = CVLanguage(
            lang=lang,
            transcript_dir=str(lang_dir),
            audio_dir=str(audio_lang),
            has_clip_durations=(lang_dir / "clip_durations.tsv").is_file(),
        )

        for tsv in sorted(lang_dir.glob("*.tsv")):
            stem = tsv.stem
            if stem not in _STANDARD_SPLITS:
                continue

            split_audio_dir = audio_lang / stem
            tar_paths: List[str] = []
            if split_audio_dir.is_dir():
                tar_paths = [str(p) for p in sorted(split_audio_dir.glob("*.tar"))]

            # 粗略数一下行数（含 header）
            rows: Optional[int] = None
            try:
                with tsv.open("r", encoding="utf-8", errors="replace") as fh:
                    rows = sum(1 for _ in fh) - 1
                    if rows < 0:
                        rows = 0
            except OSError:
                pass

            cv_lang.splits.append(CVSplit(
                name=stem,
                tsv_path=str(tsv),
                tar_paths=tar_paths,
                rows=rows,
            ))

        if cv_lang.splits:
            probe.languages.append(cv_lang)

    return probe


# ──────────────────────────────────────
# 转换：tar → mp3 + TSV+duration → JSONL
# ──────────────────────────────────────

@dataclass
class PrepResult:
    manifest_path: str
    written: int
    missing_audio: int
    missing_duration: int


class PrepError(Exception):
    pass


def _load_durations(durations_tsv: Path) -> Dict[str, float]:
    """
    clip_durations.tsv:
        clip\tduration[ms]
        common_voice_mn_xxx.mp3\t3210

    返回 filename -> seconds。
    """
    durations: Dict[str, float] = {}
    if not durations_tsv.is_file():
        return durations

    with durations_tsv.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            clip = row.get("clip") or row.get("path")
            if not clip:
                continue
            # 列名通常是 "duration[ms]"
            dur_raw = None
            for key in row:
                if key and key.startswith("duration"):
                    dur_raw = row[key]
                    break
            if dur_raw is None:
                continue
            try:
                ms = float(dur_raw)
            except (TypeError, ValueError):
                continue
            durations[clip] = ms / 1000.0
    return durations


def _extract_tars(tar_paths: List[str], target_dir: Path, log: List[str]) -> None:
    """
    解压 split 下所有 tar 到 target_dir（同级目录），幂等：
    - 文件已存在跳过
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    for tar_str in tar_paths:
        tar_path = Path(tar_str)
        log.append(f"extract {tar_path.name}...")
        try:
            with tarfile.open(tar_path) as tf:
                members = tf.getmembers()
                for m in members:
                    if not m.isfile():
                        continue
                    # tar 里音频文件名一般是 common_voice_xx_xxx.mp3（无子目录）
                    # 防御性处理：即使有子路径，只保留 basename，避免 Zip Slip
                    safe_name = Path(m.name).name
                    dst = target_dir / safe_name
                    if dst.exists() and dst.stat().st_size > 0:
                        continue
                    src = tf.extractfile(m)
                    if src is None:
                        continue
                    with dst.open("wb") as out:
                        out.write(src.read())
        except (tarfile.TarError, OSError) as exc:
            raise PrepError(f"failed to extract {tar_path}: {exc}") from exc
    log.append(f"extract done → {target_dir}")


def prepare_cv_split(
    base_dir: str,
    lang: str,
    split: str,
    log: Optional[List[str]] = None,
) -> PrepResult:
    """
    把 CV 一个 split 从原料转成 JSONL manifest。
    """
    base = Path(base_dir)
    log = log if log is not None else []

    transcript_dir = base / "transcript" / lang
    audio_dir = base / "audio" / lang / split
    split_tsv = transcript_dir / f"{split}.tsv"
    durations_tsv = transcript_dir / "clip_durations.tsv"
    manifest_dir = base / "manifests" / lang
    manifest_path = manifest_dir / f"{split}.jsonl"

    if not split_tsv.is_file():
        raise PrepError(f"split tsv not found: {split_tsv}")

    log.append(f"--- prep {lang}/{split} ---")
    log.append(f"split_tsv    = {split_tsv}")
    log.append(f"audio_dir    = {audio_dir}")
    log.append(f"manifest_out = {manifest_path}")

    # 1. 解压 tar
    tar_paths = sorted(str(p) for p in audio_dir.glob("*.tar")) if audio_dir.is_dir() else []
    if tar_paths:
        _extract_tars(tar_paths, audio_dir, log)
    else:
        log.append("(no tar files, assuming mp3 already present)")

    # 2. 读 durations
    durations = _load_durations(durations_tsv)
    log.append(f"durations loaded: {len(durations)}")

    # 3. 读 TSV + join，写 JSONL
    manifest_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    missing_audio = 0
    missing_duration = 0

    with split_tsv.open("r", encoding="utf-8", errors="replace", newline="") as tsvf, \
         manifest_path.open("w", encoding="utf-8") as jsonlf:
        reader = csv.DictReader(tsvf, delimiter="\t")
        for row in reader:
            filename = (row.get("path") or "").strip()
            sentence = (row.get("sentence") or "").strip()
            if not filename or not sentence:
                continue

            audio_path = (audio_dir / filename).resolve()
            if not audio_path.is_file():
                missing_audio += 1
                continue

            duration = durations.get(filename)
            if duration is None:
                missing_duration += 1
                continue

            jsonlf.write(json.dumps(
                {
                    "audio": {"path": str(audio_path)},
                    "sentence": sentence,
                    "duration": round(duration, 3),
                },
                ensure_ascii=False,
            ) + "\n")
            written += 1

    log.append(
        f"written={written}, missing_audio={missing_audio}, missing_duration={missing_duration}"
    )

    return PrepResult(
        manifest_path=str(manifest_path),
        written=written,
        missing_audio=missing_audio,
        missing_duration=missing_duration,
    )
