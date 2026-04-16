"""
job_worker.py — 训练任务 worker（真实执行版）
=============================================

职责：
  1. 轮询 train_runs 表中的 queued 任务
  2. 将任务状态更新为 running
  3. 以 subprocess 方式启动 whisper/finetune/finetune.py
  4. 实时解析子进程 stdout / stderr，
     - 全部写入日志文件
     - 从中提取训练进度 (step / epoch / loss) 回写数据库
  5. 子进程退出后根据 returncode 更新 completed / failed

说明：
  - 设计上它应作为独立进程运行（python -m backend.job_worker）
  - 只消费 `base_model / train_data / test_data / output_dir` 作为必需参数，
    其他参数直接从 TrainRun 模型读取，缺省值由表本身保证。
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import json as _json

from database import SessionLocal, TrainRun, DatasetPull, DatasetPrepJob, init_db
from dataset_loader import DATASET_BASE_DIR
from dataset_registry import scan_and_upsert
from dataset_prep import prepare_cv_split, PrepError


# ──────────────────────────────────────
# 常量 & 路径
# ──────────────────────────────────────

BACKEND_DIR = Path(__file__).resolve().parent
FINETUNE_DIR = BACKEND_DIR / "whisper" / "finetune"
FINETUNE_SCRIPT = FINETUNE_DIR / "finetune.py"
MERGE_SCRIPT = FINETUNE_DIR / "merge_lora.py"

DATA_DIR = Path(os.environ.get("ASR_DATA_DIR", BACKEND_DIR / "data"))
LOG_DIR = DATA_DIR / "train_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────
# 正则：从 Trainer 的 stdout 里抓关键指标
# ──────────────────────────────────────

# huggingface Trainer 默认 log 长这样:
#   {'loss': 1.23, 'learning_rate': 0.0009, 'epoch': 0.05}
# 进度条行:
#   12/1000 [00:30<10:00, ...]
_LOSS_RE = re.compile(r"'loss':\s*([0-9.eE+-]+)")
_EPOCH_RE = re.compile(r"'epoch':\s*([0-9.eE+-]+)")
_EVAL_LOSS_RE = re.compile(r"'eval_loss':\s*([0-9.eE+-]+)")
_PROGRESS_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[")


# ──────────────────────────────────────
# 任务认领 / 状态更新
# ──────────────────────────────────────

def _claim_next_train_run() -> Optional[int]:
    """取出最早的 queued 任务，并将其置为 running。"""
    db = SessionLocal()
    try:
        train_run = (
            db.query(TrainRun)
            .filter(TrainRun.status == "queued")
            .order_by(TrainRun.created_at.asc())
            .first()
        )
        if not train_run:
            return None

        train_run.status = "running"
        train_run.started_at = datetime.utcnow()
        train_run.error_message = None
        train_run.current_step = 0
        train_run.total_steps = 0
        train_run.current_epoch = 0.0
        train_run.current_loss = None
        train_run.current_eval_loss = None
        train_run.phase = "finetune"
        db.commit()
        db.refresh(train_run)

        print(f"[WORKER] Claimed train run #{train_run.id}: {train_run.name}")
        return train_run.id
    finally:
        db.close()


def _update_progress(run_id: int, **fields) -> None:
    """把解析到的进度字段写回 DB。只更新提供的字段。"""
    if not fields:
        return
    db = SessionLocal()
    try:
        train_run = db.query(TrainRun).filter(TrainRun.id == run_id).first()
        if not train_run:
            return
        for key, value in fields.items():
            if hasattr(train_run, key):
                setattr(train_run, key, value)
        db.commit()
    finally:
        db.close()


def _mark_completed(
    run_id: int,
    checkpoint_path: Optional[str],
    merged_model_path: Optional[str] = None,
) -> None:
    db = SessionLocal()
    try:
        train_run = db.query(TrainRun).filter(TrainRun.id == run_id).first()
        if not train_run:
            return
        train_run.status = "completed"
        train_run.completed_at = datetime.utcnow()
        train_run.phase = "done"
        if checkpoint_path:
            train_run.checkpoint_path = checkpoint_path
        if merged_model_path:
            train_run.merged_model_path = merged_model_path
        db.commit()
        print(f"[WORKER] Train run #{run_id} completed")
    finally:
        db.close()


def _mark_failed(run_id: int, error_message: str) -> None:
    db = SessionLocal()
    try:
        train_run = db.query(TrainRun).filter(TrainRun.id == run_id).first()
        if not train_run:
            return
        train_run.status = "failed"
        train_run.error_message = error_message[:2000]
        train_run.completed_at = datetime.utcnow()
        db.commit()
        print(f"[WORKER] Train run #{run_id} failed: {error_message[:200]}")
    finally:
        db.close()


def _load_train_run(run_id: int) -> Optional[TrainRun]:
    """读一次完整配置，worker 后续只依赖这个快照，不再查 DB。"""
    db = SessionLocal()
    try:
        return db.query(TrainRun).filter(TrainRun.id == run_id).first()
    finally:
        db.close()


# ──────────────────────────────────────
# 构造 finetune.py 的命令行
# ──────────────────────────────────────

def _build_finetune_cmd(run: TrainRun) -> List[str]:
    """
    只把必需的 4 个参数 + TrainRun 里跟默认值不同的字段透传给 finetune.py。
    finetune.py 内部自己有默认值，无需每次都把所有字段塞进命令行。
    """
    cmd: List[str] = [
        sys.executable, "-u", str(FINETUNE_SCRIPT),
        f"--train_data={run.train_data_path}",
        f"--test_data={run.test_data_path}",
        f"--base_model={run.base_model}",
        f"--output_dir={run.output_dir}",
    ]

    # 其余参数：有值才传（TrainRun 字段本身就带默认值，保持和 finetune.py 一致）
    optional = {
        "language": run.language,
        "task": run.task,
        "timestamps": run.timestamps,
        "num_train_epochs": run.num_train_epochs,
        "learning_rate": run.learning_rate,
        "warmup_steps": run.warmup_steps,
        "logging_steps": run.logging_steps,
        "eval_steps": run.eval_steps,
        "save_steps": run.save_steps,
        "per_device_train_batch_size": run.per_device_train_batch_size,
        "per_device_eval_batch_size": run.per_device_eval_batch_size,
        "gradient_accumulation_steps": run.gradient_accumulation_steps,
        "save_total_limit": run.save_total_limit,
        "use_adalora": run.use_adalora,
        "use_8bit": run.use_8bit,
        "fp16": run.fp16,
        "use_compile": run.use_compile,
        "local_files_only": run.local_files_only,
        "push_to_hub": run.push_to_hub,
    }
    for key, value in optional.items():
        if value is None:
            continue
        cmd.append(f"--{key}={value}")

    # 可空路径/ID
    if run.augment_config_path:
        cmd.append(f"--augment_config_path={run.augment_config_path}")
    if run.resume_from_checkpoint:
        cmd.append(f"--resume_from_checkpoint={run.resume_from_checkpoint}")
    if run.hub_model_id:
        cmd.append(f"--hub_model_id={run.hub_model_id}")

    return cmd


# ──────────────────────────────────────
# 子进程执行 + 日志解析
# ──────────────────────────────────────

def _parse_line_and_update(run_id: int, line: str, state: dict) -> None:
    """
    对单行 stdout 做指标抽取，必要时回写 DB。
    state 用来跨行缓存 step/total，避免每行都写库。
    """
    fields: dict = {}

    m = _LOSS_RE.search(line)
    if m:
        try:
            fields["current_loss"] = float(m.group(1))
        except ValueError:
            pass

    m = _EPOCH_RE.search(line)
    if m:
        try:
            fields["current_epoch"] = float(m.group(1))
        except ValueError:
            pass

    m = _EVAL_LOSS_RE.search(line)
    if m:
        try:
            fields["current_eval_loss"] = float(m.group(1))
        except ValueError:
            pass

    m = _PROGRESS_RE.search(line)
    if m:
        try:
            cur = int(m.group(1))
            tot = int(m.group(2))
            # 进度条每 tick 都更新太频繁，只在 step 变化时写库
            if cur != state.get("current_step") or tot != state.get("total_steps"):
                fields["current_step"] = cur
                fields["total_steps"] = tot
                state["current_step"] = cur
                state["total_steps"] = tot
        except ValueError:
            pass

    if fields:
        _update_progress(run_id, **fields)


def _guess_checkpoint_path(run: TrainRun) -> Optional[str]:
    """
    finetune.py 完成后把 LoRA adapter 保存到
        {output_dir}/{basename(base_model)}/checkpoint-final
    这里推断一下，好让 merge_lora.py / 前端能定位到 adapter。
    """
    base_model = (run.base_model or "").rstrip("/")
    basename = os.path.basename(base_model) if base_model else ""
    if not basename:
        return None
    final_dir = Path(run.output_dir) / basename / "checkpoint-final"
    return str(final_dir)


def _merged_output_dir(run: TrainRun) -> Path:
    """
    merge_lora.py 的 --output_dir（父目录）。
    我们固定放到 {output_dir}/merged，
    merge 脚本内部会在里面再建一层 `{basename(base_model)}-finetune/`。
    """
    return Path(run.output_dir) / "merged"


def _guess_merged_model_path(run: TrainRun) -> Optional[str]:
    """
    merge_lora.py 最终保存路径：
        {output_dir}/merged/{basename(base_model)}-finetune/
    这是后续推理 / 评测应加载的模型目录。
    """
    base_model = (run.base_model or "").rstrip("/")
    basename = os.path.basename(base_model) if base_model else ""
    if not basename:
        return None
    return str(_merged_output_dir(run) / f"{basename}-finetune")


def _run_merge_subprocess(run: TrainRun, lora_path: str) -> int:
    """
    运行 merge_lora.py，把 LoRA adapter 与 base model 合并成完整模型。
    日志追加到同一个 train log 里（phase=merge），便于前端一屏看完整个流程。
    """
    merge_output = _merged_output_dir(run)
    merge_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", str(MERGE_SCRIPT),
        f"--lora_model={lora_path}",
        f"--output_dir={merge_output}",
        f"--local_files_only={run.local_files_only}",
    ]

    log_file = LOG_DIR / f"train_run_{run.id}.log"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if run.gpu_id:
        env["CUDA_VISIBLE_DEVICES"] = run.gpu_id

    print(f"[WORKER] Merging LoRA: {' '.join(shlex.quote(c) for c in cmd)}")

    _update_progress(run.id, phase="merging")

    log_fh = log_file.open("a", encoding="utf-8")
    log_fh.write("\n" + "#" * 82 + "\n")
    log_fh.write(f"# phase=merge | {datetime.utcnow().isoformat()}Z\n")
    log_fh.write(f"# cmd: {' '.join(shlex.quote(c) for c in cmd)}\n")
    log_fh.write("#" + "=" * 80 + "\n")
    log_fh.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(FINETUNE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True,
    )
    _update_progress(run.id, pid=proc.pid)

    def _pump(stream, tag: str) -> None:
        try:
            for raw in stream:
                log_fh.write(f"[merge/{tag}] {raw.rstrip()}\n")
                log_fh.flush()
        finally:
            stream.close()

    t_out = threading.Thread(target=_pump, args=(proc.stdout, "stdout"), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, "stderr"), daemon=True)
    t_out.start()
    t_err.start()

    proc.wait()
    t_out.join(timeout=5)
    t_err.join(timeout=5)

    log_fh.write(f"# merge exited with code {proc.returncode}\n")
    log_fh.close()
    return proc.returncode


def _run_finetune_subprocess(run: TrainRun) -> int:
    """
    以 subprocess 方式启动 finetune.py，
    并通过 2 个线程读 stdout/stderr，
    所有输出同时写入日志文件 + 解析 -> 数据库。

    Returns:
        子进程 returncode（0 表示成功）
    """
    cmd = _build_finetune_cmd(run)
    log_file = LOG_DIR / f"train_run_{run.id}.log"

    # worker 的工作目录切到 finetune/ 下，保证其 utils.* 相对 import 能解析
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    # 如果指定了 GPU，设置 CUDA_VISIBLE_DEVICES
    if run.gpu_id:
        env["CUDA_VISIBLE_DEVICES"] = run.gpu_id
        print(f"[WORKER] CUDA_VISIBLE_DEVICES={run.gpu_id}")

    print(f"[WORKER] Launching: {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"[WORKER] Log file: {log_file}")

    _update_progress(run.id, log_path=str(log_file))

    # 在打开日志文件之前记录一下启动信息
    log_fh = log_file.open("w", encoding="utf-8")
    log_fh.write(f"# train_run #{run.id} | {datetime.utcnow().isoformat()}Z\n")
    log_fh.write(f"# cmd: {' '.join(shlex.quote(c) for c in cmd)}\n")
    log_fh.write(f"# cwd: {FINETUNE_DIR}\n")
    log_fh.write("#" + "=" * 80 + "\n")
    log_fh.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(FINETUNE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        text=True,
    )
    _update_progress(run.id, pid=proc.pid)

    state: dict = {}

    def _pump(stream, tag: str) -> None:
        try:
            for raw in stream:
                line = raw.rstrip("\n")
                log_fh.write(f"[{tag}] {line}\n")
                log_fh.flush()
                # 只解析 stdout；stderr 主要是 tqdm 进度条，也顺便解析一下
                try:
                    _parse_line_and_update(run.id, line, state)
                except Exception as exc:  # 解析失败不应拖垮 worker
                    log_fh.write(f"[WORKER] parse error: {exc}\n")
        finally:
            stream.close()

    t_out = threading.Thread(target=_pump, args=(proc.stdout, "stdout"), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, "stderr"), daemon=True)
    t_out.start()
    t_err.start()

    proc.wait()
    t_out.join(timeout=5)
    t_err.join(timeout=5)

    log_fh.write(f"# exited with code {proc.returncode}\n")
    log_fh.close()
    return proc.returncode


# ──────────────────────────────────────
# 入口
# ──────────────────────────────────────

# ──────────────────────────────────────
# DatasetPull（HuggingFace 拉取）消费
# ──────────────────────────────────────

def _claim_next_dataset_pull() -> Optional[int]:
    """取出最早的 queued 拉取任务。"""
    db = SessionLocal()
    try:
        pull = (
            db.query(DatasetPull)
            .filter(DatasetPull.status == "queued")
            .order_by(DatasetPull.created_at.asc())
            .first()
        )
        if not pull:
            return None
        pull.status = "running"
        pull.started_at = datetime.utcnow()
        pull.error_message = None
        db.commit()
        db.refresh(pull)
        print(f"[WORKER] Claimed dataset pull #{pull.id}: {pull.repo_id}")
        return pull.id
    finally:
        db.close()


def _finish_dataset_pull(
    pull_id: int,
    *,
    status: str,
    local_dir: Optional[str] = None,
    error: Optional[str] = None,
    log_tail: Optional[str] = None,
    registered_count: int = 0,
) -> None:
    db = SessionLocal()
    try:
        pull = db.query(DatasetPull).filter(DatasetPull.id == pull_id).first()
        if not pull:
            return
        pull.status = status
        pull.completed_at = datetime.utcnow()
        if local_dir is not None:
            pull.local_dir = local_dir
        if error is not None:
            pull.error_message = error[:2000]
        if log_tail is not None:
            pull.log_tail = log_tail[-4000:]
        pull.registered_count = registered_count
        db.commit()
    finally:
        db.close()


def _repo_slug(repo_id: str) -> str:
    """把 user/name 转成目录安全名：user__name。"""
    return repo_id.replace("/", "__").replace("\\", "__")


def _run_huggingface_pull(pull_id: int) -> None:
    """
    真正执行一次 HF 数据集下载。
    走 huggingface_hub.snapshot_download，然后扫描目标目录把里面识别到的
    CSV/JSONL upsert 进 datasets 表。
    """
    db = SessionLocal()
    try:
        pull = db.query(DatasetPull).filter(DatasetPull.id == pull_id).first()
        if not pull:
            return
        repo_id = pull.repo_id
        revision = pull.revision
        allow_raw = pull.allow_patterns
    finally:
        db.close()

    # 解析 allow_patterns: 逗号分隔的 glob 串 → list[str] or None
    allow_patterns: Optional[List[str]] = None
    if allow_raw:
        parts = [p.strip() for p in allow_raw.split(",") if p.strip()]
        if parts:
            allow_patterns = parts

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        msg = "huggingface_hub not installed. `pip install huggingface_hub`"
        print(f"[WORKER] HF pull #{pull_id} FAILED: {msg}")
        _finish_dataset_pull(pull_id, status="failed", error=msg)
        return

    target_dir = Path(DATASET_BASE_DIR) / _repo_slug(repo_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[WORKER] HF pull → {target_dir}  "
        f"(repo={repo_id}, rev={revision}, allow_patterns={allow_patterns})"
    )

    log_lines: list[str] = [
        f"[{datetime.utcnow().isoformat()}Z] snapshot_download start",
        f"repo_id        = {repo_id}",
        f"revision       = {revision or 'main'}",
        f"local_dir      = {target_dir}",
        f"allow_patterns = {allow_patterns if allow_patterns else '(none → full repo)'}",
    ]

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
        )
        log_lines.append("snapshot_download done")
        print(f"[WORKER] HF pull #{pull_id} snapshot_download done → {target_dir}")
    except Exception as exc:
        # 常见错误做个人类可读提示，特别是 gated repo 的 401/403
        err_str = str(exc)
        hint = ""
        low = err_str.lower()
        if "401" in err_str or "403" in err_str or "gated" in low or "restricted" in low or "authoriz" in low:
            hint = (
                "\nHINT: this repo looks gated/private. Either:"
                "\n  1) open the repo page on HuggingFace and accept its license,"
                "\n     then `huggingface-cli login` in the env that runs the worker, or"
                "\n  2) export HF_TOKEN=<your read token> before launching the worker."
            )
        log_lines.append(f"snapshot_download FAILED: {exc}{hint}")
        print(f"[WORKER] HF pull #{pull_id} FAILED (snapshot_download): {exc}{hint}")
        _finish_dataset_pull(
            pull_id,
            status="failed",
            local_dir=str(target_dir),
            error=f"{exc}{hint}",
            log_tail="\n".join(log_lines),
        )
        return

    # 下载成功，扫描入库
    registered_count = 0
    db = SessionLocal()
    try:
        scanned, added, updated, _removed = scan_and_upsert(
            db,
            base_dir=str(target_dir),
            source="huggingface",
            source_repo=repo_id,
        )
        registered_count = added + updated
        log_lines.append(
            f"scan: scanned={scanned}, added={added}, updated={updated}"
        )
    except Exception as exc:
        log_lines.append(f"scan FAILED: {exc}")
        print(f"[WORKER] HF pull #{pull_id} FAILED (scan): {exc}")
        _finish_dataset_pull(
            pull_id,
            status="failed",
            local_dir=str(target_dir),
            error=f"downloaded but scan failed: {exc}",
            log_tail="\n".join(log_lines),
        )
        return
    finally:
        db.close()

    _finish_dataset_pull(
        pull_id,
        status="completed",
        local_dir=str(target_dir),
        log_tail="\n".join(log_lines),
        registered_count=registered_count,
    )
    print(f"[WORKER] HF pull #{pull_id} done, registered {registered_count} files")


def process_one_dataset_pull() -> bool:
    pull_id = _claim_next_dataset_pull()
    if pull_id is None:
        return False
    try:
        _run_huggingface_pull(pull_id)
    except Exception as exc:
        print(f"[WORKER] HF pull #{pull_id} FAILED (unexpected): {exc}")
        _finish_dataset_pull(pull_id, status="failed", error=f"unexpected: {exc}")
    return True


# ──────────────────────────────────────
# DatasetPrepJob（Common Voice 等预处理）消费
# ──────────────────────────────────────

def _claim_next_prep_job() -> Optional[int]:
    db = SessionLocal()
    try:
        job = (
            db.query(DatasetPrepJob)
            .filter(DatasetPrepJob.status == "queued")
            .order_by(DatasetPrepJob.created_at.asc())
            .first()
        )
        if not job:
            return None
        job.status = "running"
        job.started_at = datetime.utcnow()
        job.error_message = None
        db.commit()
        db.refresh(job)
        print(f"[WORKER] Claimed prep job #{job.id}: {job.kind} lang={job.lang}")
        return job.id
    finally:
        db.close()


def _finish_prep_job(
    job_id: int,
    *,
    status: str,
    error: Optional[str] = None,
    log_tail: Optional[str] = None,
    produced_manifests: Optional[List[str]] = None,
    registered_count: int = 0,
) -> None:
    db = SessionLocal()
    try:
        job = db.query(DatasetPrepJob).filter(DatasetPrepJob.id == job_id).first()
        if not job:
            return
        job.status = status
        job.completed_at = datetime.utcnow()
        if error is not None:
            job.error_message = error[:2000]
        if log_tail is not None:
            job.log_tail = log_tail[-4000:]
        if produced_manifests is not None:
            job.produced_manifests = _json.dumps(produced_manifests)
        job.registered_count = registered_count
        db.commit()
    finally:
        db.close()


def _run_cv_prep(job_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.query(DatasetPrepJob).filter(DatasetPrepJob.id == job_id).first()
        if not job:
            return
        source_dir = job.source_dir
        lang = job.lang
        try:
            splits = _json.loads(job.splits or "[]")
        except _json.JSONDecodeError:
            splits = []
    finally:
        db.close()

    if not splits:
        _finish_prep_job(job_id, status="failed", error="no splits selected")
        return

    log_lines: List[str] = [
        f"[{datetime.utcnow().isoformat()}Z] cv prep start",
        f"source_dir = {source_dir}",
        f"lang       = {lang}",
        f"splits     = {splits}",
    ]
    print(f"[WORKER] CV prep #{job_id} lang={lang} splits={splits}")

    produced: List[str] = []
    total_written = 0
    try:
        for split in splits:
            result = prepare_cv_split(source_dir, lang, split, log=log_lines)
            produced.append(result.manifest_path)
            total_written += result.written
            print(
                f"[WORKER] CV prep #{job_id} {lang}/{split} → "
                f"{result.written} lines ({result.missing_audio} no-audio, "
                f"{result.missing_duration} no-duration)"
            )
    except PrepError as exc:
        log_lines.append(f"PrepError: {exc}")
        print(f"[WORKER] CV prep #{job_id} FAILED: {exc}")
        _finish_prep_job(
            job_id,
            status="failed",
            error=str(exc),
            log_tail="\n".join(log_lines),
            produced_manifests=produced,
        )
        return
    except Exception as exc:
        log_lines.append(f"Unexpected: {exc}")
        print(f"[WORKER] CV prep #{job_id} FAILED (unexpected): {exc}")
        _finish_prep_job(
            job_id,
            status="failed",
            error=f"unexpected: {exc}",
            log_tail="\n".join(log_lines),
            produced_manifests=produced,
        )
        return

    # 触发 scan，把新写的 manifest 登记进 datasets 表
    registered = 0
    db = SessionLocal()
    try:
        scanned, added, updated, _removed = scan_and_upsert(
            db,
            base_dir=source_dir,
            source="huggingface",
            # source_repo 保留 — scan 不会覆盖已有的 source_repo
        )
        registered = added + updated
        log_lines.append(
            f"scan: scanned={scanned}, added={added}, updated={updated}"
        )
    except Exception as exc:
        log_lines.append(f"scan FAILED: {exc}")
        print(f"[WORKER] CV prep #{job_id} scan FAILED: {exc}")
    finally:
        db.close()

    _finish_prep_job(
        job_id,
        status="completed",
        log_tail="\n".join(log_lines),
        produced_manifests=produced,
        registered_count=registered,
    )
    print(
        f"[WORKER] CV prep #{job_id} done: {total_written} lines, "
        f"{len(produced)} manifests, {registered} registered"
    )


def process_one_prep_job() -> bool:
    job_id = _claim_next_prep_job()
    if job_id is None:
        return False
    try:
        _run_cv_prep(job_id)
    except Exception as exc:
        print(f"[WORKER] CV prep #{job_id} FAILED (outer): {exc}")
        _finish_prep_job(job_id, status="failed", error=f"unexpected: {exc}")
    return True


# ──────────────────────────────────────
# TrainRun 消费
# ──────────────────────────────────────

def process_one_train_run() -> bool:
    """
    处理一条任务。

    Returns:
        True 表示处理了一条任务，False 表示当前没有任务。
    """
    run_id = _claim_next_train_run()
    if run_id is None:
        return False

    run = _load_train_run(run_id)
    if run is None:
        _mark_failed(run_id, "train run disappeared after claim")
        return True

    if not FINETUNE_SCRIPT.exists():
        _mark_failed(run_id, f"finetune.py not found at {FINETUNE_SCRIPT}")
        return True

    try:
        rc = _run_finetune_subprocess(run)
    except Exception as exc:
        _mark_failed(run_id, f"subprocess launch failed: {exc}")
        return True

    if rc != 0:
        _mark_failed(run_id, f"finetune.py exited with code {rc}")
        return True

    # 微调成功，尝试 merge LoRA adapter
    lora_path = _guess_checkpoint_path(run)
    _update_progress(run_id, checkpoint_path=lora_path)

    if not lora_path or not Path(lora_path).exists():
        # 没产出 adapter 目录（比如是全量微调 / 路径异常），跳过 merge
        _mark_completed(run_id, checkpoint_path=lora_path)
        return True

    if not MERGE_SCRIPT.exists():
        _mark_failed(
            run_id,
            f"finetune ok but merge_lora.py not found at {MERGE_SCRIPT}",
        )
        return True

    try:
        merge_rc = _run_merge_subprocess(run, lora_path)
    except Exception as exc:
        _mark_failed(run_id, f"merge_lora launch failed: {exc}")
        return True

    if merge_rc != 0:
        _mark_failed(run_id, f"merge_lora.py exited with code {merge_rc}")
        return True

    merged_path = _guess_merged_model_path(run)
    _mark_completed(
        run_id,
        checkpoint_path=lora_path,
        merged_model_path=merged_path,
    )
    return True


def _recover_orphaned_tasks() -> None:
    """
    Worker 启动时调用一次。

    单 worker 部署下，任何处于 running 的任务一定是上次被 kill 时留下的孤儿，
    这里把它们统一标成 failed，防止前端一直显示"running"。

    注意：这假设同一时间只有一个 worker 实例在跑。如果将来部署多 worker，
    需要改成基于 heartbeat_at 的 stale 检测。
    """
    db = SessionLocal()
    try:
        # DatasetPull 孤儿
        stale_pulls = (
            db.query(DatasetPull)
            .filter(DatasetPull.status.in_(["running", "queued"]))
            .all()
        )
        for p in stale_pulls:
            if p.status == "running":
                p.status = "failed"
                p.completed_at = datetime.utcnow()
                p.error_message = (
                    "worker restarted while task was running (orphaned); "
                    "recreate the pull if you still want it"
                )
                print(f"[WORKER] Recovered orphaned dataset pull #{p.id} ({p.repo_id}) → failed")
            # queued 的保持 queued，worker 直接重新消费就行

        # DatasetPrepJob 孤儿
        stale_prep = (
            db.query(DatasetPrepJob)
            .filter(DatasetPrepJob.status == "running")
            .all()
        )
        for j in stale_prep:
            j.status = "failed"
            j.completed_at = datetime.utcnow()
            j.error_message = "worker restarted while prep was running (orphaned)"
            print(f"[WORKER] Recovered orphaned prep job #{j.id} ({j.lang}) → failed")

        # TrainRun 孤儿
        stale_runs = (
            db.query(TrainRun)
            .filter(TrainRun.status == "running")
            .all()
        )
        for r in stale_runs:
            r.status = "failed"
            r.completed_at = datetime.utcnow()
            r.error_message = (
                "worker restarted while training was running (orphaned); "
                "resume_from_checkpoint may be needed to continue"
            )
            r.phase = None
            print(f"[WORKER] Recovered orphaned train run #{r.id} ({r.name}) → failed")

        if stale_pulls or stale_prep or stale_runs:
            db.commit()
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR Eval training worker")
    parser.add_argument("--poll-interval", type=float, default=3.0, help="轮询间隔（秒）")
    parser.add_argument("--once", action="store_true", help="只处理一条任务后退出")
    args = parser.parse_args()

    init_db()
    _recover_orphaned_tasks()

    print(
        f"[WORKER] Started. poll_interval={args.poll_interval:.1f}s, once={args.once}"
    )
    print(f"[WORKER] finetune script: {FINETUNE_SCRIPT}")
    print(f"[WORKER] log dir: {LOG_DIR}")

    while True:
        try:
            # 先处理轻量的数据集拉取和预处理（不占 GPU），再轮训练任务
            handled = process_one_dataset_pull()
            if not handled:
                handled = process_one_prep_job()
            if not handled:
                handled = process_one_train_run()
        except KeyboardInterrupt:
            print("[WORKER] Interrupted, exiting.")
            break
        except Exception as exc:
            # 认领阶段就失败了（DB 连不上等），打印后继续轮询
            print(f"[WORKER] Unexpected error: {exc}")
            handled = False

        if args.once:
            break

        if not handled:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
