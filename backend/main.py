"""
main.py — FastAPI 后端入口
============================

职责：
  1. 定义所有 REST API 路由
  2. 处理跨域请求（CORS，让前端能访问）
  3. 启动时初始化数据库
  4. 后台任务调度（评测在后台异步执行，不阻塞 API）

API 路由总览：
  GET  /api/datasets             获取可用数据集列表
  POST /api/train-runs           创建训练任务
  GET  /api/train-runs           获取训练任务列表
  GET  /api/train-runs/{id}      获取训练任务详情
  POST /api/evaluations          发起新评测
  GET  /api/evaluations          获取评测记录列表
  GET  /api/evaluations/{id}     获取单次评测详情
  DEL  /api/evaluations/{id}     删除评测记录
  POST /api/compare              多模型对比
  GET  /api/evaluations/{id}/export  导出报告 PDF

启动方式：
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database import (
    init_db, get_db,
    Evaluation, EvaluationDetail, TrainRun,
    Dataset, DatasetPull, DatasetPrepJob,
)
from schemas import (
    EvalCreate, EvalSummary, EvalFullResponse, EvalDetailItem,
    EditOpsBreakdown, WerDistribution,
    TrainRunEvalRequest,
    DatasetListResponse,
    DatasetOut, DatasetPreview, ScanResponse,
    DatasetPullCreate, DatasetPullOut,
    CVProbeResponse, CVLanguageInfo, CVSplitInfo,
    DatasetPrepCreate, DatasetPrepOut,
    TrainRunCreate, TrainRunSummary, TrainRunDetail,
    CompareRequest, CompareResponse, CompareItem,
    MessageResponse,
    GpuInfo, GpuStatusResponse,
)
from dataset_loader import scan_datasets, load_dataset, DATASET_BASE_DIR
from dataset_registry import scan_and_upsert, preview_dataset
from dataset_prep import probe_cv_layout


# ──────────────────────────────────────
# 1. 应用初始化
# ──────────────────────────────────────

app = FastAPI(
    title="藏语 ASR 评测平台",
    description="Tibetan ASR Evaluation Platform API",
    version="1.0.0",
)

# CORS 中间件：允许前端（不同端口/域名）访问后端 API
# 开发环境允许所有来源，生产环境应限制为前端域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 开发阶段允许所有来源
    allow_credentials=True,
    allow_methods=["*"],           # 允许所有 HTTP 方法
    allow_headers=["*"],           # 允许所有请求头
)


@app.on_event("startup")
def on_startup():
    """
    应用启动时执行：
      - 初始化数据库（建表）
      - 打印数据集目录信息
    """
    init_db()
    print(f"[INFO] 数据库已初始化")
    print(f"[INFO] 数据集目录: {DATASET_BASE_DIR}")


# ──────────────────────────────────────
# GPU 状态 API
# ──────────────────────────────────────

def _query_gpu_status() -> GpuStatusResponse:
    """调用 nvidia-smi 获取 GPU 实时状态。"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return GpuStatusResponse(available=False)

        # 获取 driver / cuda 版本
        ver_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        driver_ver = ver_result.stdout.strip().split("\n")[0].strip() if ver_result.returncode == 0 else None

        cuda_ver = None
        ver2 = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5,
        )
        if ver2.returncode == 0:
            import re as _re
            m = _re.search(r"CUDA Version:\s*([\d.]+)", ver2.stdout)
            if m:
                cuda_ver = m.group(1)

        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            idx = int(parts[0])
            name = parts[1]
            util = float(parts[2]) if parts[2] not in ("[N/A]", "") else 0.0
            mem_used = float(parts[3]) if parts[3] not in ("[N/A]", "") else 0.0
            mem_total = float(parts[4]) if parts[4] not in ("[N/A]", "") else 1.0
            temp = int(parts[5]) if len(parts) > 5 and parts[5] not in ("[N/A]", "") else None
            power_draw = float(parts[6]) if len(parts) > 6 and parts[6] not in ("[N/A]", "") else None
            power_limit = float(parts[7]) if len(parts) > 7 and parts[7] not in ("[N/A]", "") else None
            gpus.append(GpuInfo(
                index=idx,
                name=name,
                utilization_pct=util,
                memory_used_mb=mem_used,
                memory_total_mb=mem_total,
                memory_pct=round(mem_used / mem_total * 100, 1) if mem_total > 0 else 0,
                temperature=temp,
                power_draw_w=power_draw,
                power_limit_w=power_limit,
            ))
        return GpuStatusResponse(available=True, driver_version=driver_ver, cuda_version=cuda_ver, gpus=gpus)
    except FileNotFoundError:
        return GpuStatusResponse(available=False)
    except Exception:
        return GpuStatusResponse(available=False)


@app.get("/api/gpu-status", response_model=GpuStatusResponse, tags=["系统"])
def get_gpu_status():
    """获取服务器 GPU 实时状态。"""
    return _query_gpu_status()


# ──────────────────────────────────────
# 2. 数据集 API
# ──────────────────────────────────────

@app.get("/api/datasets/legacy", response_model=DatasetListResponse, tags=["数据集"])
def list_datasets_legacy():
    """
    老版"扫一下磁盘立即返回"接口，不走注册表。

    保留是为了兼容早期前端，新代码请用 `/api/datasets`。
    """
    datasets = scan_datasets()
    return DatasetListResponse(
        datasets=datasets,
        base_dir=DATASET_BASE_DIR,
    )


# ── 注册表 API ──

@app.get("/api/datasets", response_model=list[DatasetOut], tags=["数据集"])
def list_registered_datasets(
    kind: Optional[str] = Query(None, description="按 kind 过滤: eval_csv / train_manifest"),
    source: Optional[str] = Query(None, description="按来源过滤: local / huggingface"),
    db: Session = Depends(get_db),
):
    """从注册表返回数据集列表。"""
    query = db.query(Dataset)
    if kind:
        query = query.filter(Dataset.kind == kind)
    if source:
        query = query.filter(Dataset.source == source)
    rows = query.order_by(Dataset.updated_at.desc()).all()
    return [DatasetOut.model_validate(r) for r in rows]


@app.post("/api/datasets/scan", response_model=ScanResponse, tags=["数据集"])
def scan_datasets_endpoint(db: Session = Depends(get_db)):
    """
    触发一次全量扫描，对 DATASET_BASE_DIR 下所有文件做 kind 嗅探并 upsert。
    """
    scanned, added, updated, removed = scan_and_upsert(db)
    return ScanResponse(scanned=scanned, added=added, updated=updated, removed=removed)


@app.get("/api/datasets/{ds_id}/preview", response_model=DatasetPreview, tags=["数据集"])
def preview_dataset_endpoint(
    ds_id: int,
    n: int = Query(5, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """读数据集前 N 行，前端用来检查列名是否对得上。"""
    ds = db.query(Dataset).filter(Dataset.id == ds_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {ds_id} not found")
    data = preview_dataset(ds, n=n)
    return DatasetPreview(**data)


@app.delete("/api/datasets/{ds_id}", response_model=MessageResponse, tags=["数据集"])
def delete_dataset(ds_id: int, db: Session = Depends(get_db)):
    """
    从注册表移除一条记录（不删除磁盘文件）。

    如果用户想同时清掉磁盘文件，自己去删 → 下次 scan 会把它标 missing。
    """
    ds = db.query(Dataset).filter(Dataset.id == ds_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset {ds_id} not found")
    db.delete(ds)
    db.commit()
    return MessageResponse(message=f"Dataset {ds_id} removed from registry")


# ── HuggingFace 拉取 API ──

@app.post("/api/dataset-pulls", response_model=DatasetPullOut, tags=["数据集"])
def create_dataset_pull(req: DatasetPullCreate, db: Session = Depends(get_db)):
    """
    提交一个 HF 数据集拉取任务。worker 会消费它并把数据下到 DATASET_BASE_DIR 下。
    """
    pull = DatasetPull(
        repo_id=req.repo_id.strip(),
        revision=(req.revision or None),
        allow_patterns=(req.allow_patterns.strip() if req.allow_patterns else None),
        status="queued",
    )
    db.add(pull)
    db.commit()
    db.refresh(pull)
    return DatasetPullOut.model_validate(pull)


@app.get("/api/dataset-pulls", response_model=list[DatasetPullOut], tags=["数据集"])
def list_dataset_pulls(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """拉取任务列表，最近的在前。"""
    rows = (
        db.query(DatasetPull)
        .order_by(DatasetPull.created_at.desc())
        .limit(limit)
        .all()
    )
    return [DatasetPullOut.model_validate(r) for r in rows]


@app.get("/api/dataset-pulls/{pull_id}", response_model=DatasetPullOut, tags=["数据集"])
def get_dataset_pull(pull_id: int, db: Session = Depends(get_db)):
    pull = db.query(DatasetPull).filter(DatasetPull.id == pull_id).first()
    if not pull:
        raise HTTPException(status_code=404, detail=f"Pull job {pull_id} not found")
    return DatasetPullOut.model_validate(pull)


@app.delete("/api/dataset-pulls/{pull_id}", response_model=MessageResponse, tags=["数据集"])
def delete_dataset_pull(pull_id: int, db: Session = Depends(get_db)):
    """
    删除一条 pull 任务记录。
    - queued / running 的记录会被直接丢弃（不会真的去杀 worker 进程 —
      实际 worker 重启时 _recover_orphaned_pulls() 会兜底修正状态）
    - completed / failed 的记录就是纯清理
    不会删除已经下到磁盘上的文件。
    """
    pull = db.query(DatasetPull).filter(DatasetPull.id == pull_id).first()
    if not pull:
        raise HTTPException(status_code=404, detail=f"Pull job {pull_id} not found")
    db.delete(pull)
    db.commit()
    return MessageResponse(message=f"Pull job {pull_id} deleted")


@app.get(
    "/api/dataset-pulls/{pull_id}/cv-probe",
    response_model=CVProbeResponse,
    tags=["数据集"],
)
def cv_probe_pull(pull_id: int, db: Session = Depends(get_db)):
    """
    检查这次 pull 的落盘目录是不是 Common Voice 风格，
    返回可处理的语言 + splits，供前端渲染 Prepare 弹窗。
    """
    pull = db.query(DatasetPull).filter(DatasetPull.id == pull_id).first()
    if not pull:
        raise HTTPException(status_code=404, detail=f"Pull job {pull_id} not found")
    if not pull.local_dir:
        raise HTTPException(status_code=400, detail="Pull has no local_dir yet")

    probe = probe_cv_layout(pull.local_dir)
    return CVProbeResponse(
        base_dir=probe.base_dir,
        is_cv=probe.is_cv,
        languages=[
            CVLanguageInfo(
                lang=lang.lang,
                transcript_dir=lang.transcript_dir,
                audio_dir=lang.audio_dir,
                has_clip_durations=lang.has_clip_durations,
                splits=[
                    CVSplitInfo(
                        name=s.name,
                        tsv_path=s.tsv_path,
                        tar_count=len(s.tar_paths),
                        rows=s.rows,
                    )
                    for s in lang.splits
                ],
            )
            for lang in probe.languages
        ],
    )


# ── 数据集预处理任务 ──

@app.post("/api/dataset-prep-jobs", response_model=DatasetPrepOut, tags=["数据集"])
def create_dataset_prep_job(req: DatasetPrepCreate, db: Session = Depends(get_db)):
    """
    提交一个预处理任务（目前只处理 Common Voice 风格的目录）。
    worker 会：解 tar → join TSV + durations → 写 JSONL manifest → 自动 scan 入库。
    """
    if req.kind != "cv":
        raise HTTPException(status_code=400, detail=f"unsupported prep kind: {req.kind}")

    job = DatasetPrepJob(
        kind=req.kind,
        source_dir=req.source_dir,
        source_pull_id=req.source_pull_id,
        lang=req.lang.strip(),
        splits=json.dumps(req.splits),
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return DatasetPrepOut.model_validate(job)


@app.get("/api/dataset-prep-jobs", response_model=list[DatasetPrepOut], tags=["数据集"])
def list_dataset_prep_jobs(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(DatasetPrepJob)
        .order_by(DatasetPrepJob.created_at.desc())
        .limit(limit)
        .all()
    )
    return [DatasetPrepOut.model_validate(r) for r in rows]


@app.get("/api/dataset-prep-jobs/{job_id}", response_model=DatasetPrepOut, tags=["数据集"])
def get_dataset_prep_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(DatasetPrepJob).filter(DatasetPrepJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Prep job {job_id} not found")
    return DatasetPrepOut.model_validate(job)


@app.delete("/api/dataset-prep-jobs/{job_id}", response_model=MessageResponse, tags=["数据集"])
def delete_dataset_prep_job(job_id: int, db: Session = Depends(get_db)):
    job = db.query(DatasetPrepJob).filter(DatasetPrepJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Prep job {job_id} not found")
    db.delete(job)
    db.commit()
    return MessageResponse(message=f"Prep job {job_id} deleted")


# ──────────────────────────────────────
# 3. 训练 API
# ──────────────────────────────────────

@app.post("/api/train-runs", response_model=TrainRunDetail, tags=["训练"])
def create_train_run(req: TrainRunCreate, db: Session = Depends(get_db)):
    """
    创建一个微调训练任务。

    Step 1 先只保存配置到数据库，状态初始为 queued，
    暂时不触发真实训练。
    """
    train_run = TrainRun(
        name=req.name,
        base_model=req.base_model,
        train_data_path=req.train_data_path,
        test_data_path=req.test_data_path,
        output_dir=req.output_dir,
        language=req.language,
        task=req.task,
        timestamps=req.timestamps,
        num_train_epochs=req.num_train_epochs,
        learning_rate=req.learning_rate,
        warmup_steps=req.warmup_steps,
        logging_steps=req.logging_steps,
        eval_steps=req.eval_steps,
        save_steps=req.save_steps,
        per_device_train_batch_size=req.per_device_train_batch_size,
        per_device_eval_batch_size=req.per_device_eval_batch_size,
        gradient_accumulation_steps=req.gradient_accumulation_steps,
        save_total_limit=req.save_total_limit,
        use_adalora=req.use_adalora,
        use_8bit=req.use_8bit,
        fp16=req.fp16,
        use_compile=req.use_compile,
        local_files_only=req.local_files_only,
        push_to_hub=req.push_to_hub,
        augment_config_path=req.augment_config_path,
        resume_from_checkpoint=req.resume_from_checkpoint,
        hub_model_id=req.hub_model_id,
        gpu_id=(req.gpu_id.strip() if req.gpu_id else None),
        status="queued",
    )
    db.add(train_run)
    db.commit()
    db.refresh(train_run)

    return TrainRunDetail.model_validate(train_run)


@app.get("/api/train-runs", response_model=list[TrainRunSummary], tags=["训练"])
def list_train_runs(
    status: Optional[str] = Query(None, description="按状态过滤: queued/running/completed/failed"),
    base_model: Optional[str] = Query(None, description="按基础模型过滤"),
    limit: int = Query(50, ge=1, le=200, description="返回条数"),
    offset: int = Query(0, ge=0, description="跳过条数（分页）"),
    db: Session = Depends(get_db),
):
    """获取训练任务列表。"""
    query = db.query(TrainRun)

    if status:
        query = query.filter(TrainRun.status == status)
    if base_model:
        query = query.filter(TrainRun.base_model.ilike(f"%{base_model}%"))

    train_runs = (
        query
        .order_by(TrainRun.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return [TrainRunSummary.model_validate(run) for run in train_runs]


@app.get("/api/train-runs/{run_id}", response_model=TrainRunDetail, tags=["训练"])
def get_train_run(run_id: int, db: Session = Depends(get_db)):
    """获取单个训练任务的完整配置与状态。"""
    train_run = db.query(TrainRun).filter(TrainRun.id == run_id).first()
    if not train_run:
        raise HTTPException(status_code=404, detail=f"训练任务 {run_id} 不存在")

    return TrainRunDetail.model_validate(train_run)


@app.get("/api/train-runs/{run_id}/log", tags=["训练"])
def get_train_run_log(
    run_id: int,
    tail: int = Query(500, ge=1, le=20000, description="返回最后 N 行日志"),
    db: Session = Depends(get_db),
):
    """
    读取训练日志文件的最后 N 行。

    worker 把 finetune.py / merge_lora.py 的 stdout/stderr
    都写到 {DATA_DIR}/train_logs/train_run_{id}.log。
    前端轮询这个接口来展示"实时日志"。
    """
    train_run = db.query(TrainRun).filter(TrainRun.id == run_id).first()
    if not train_run:
        raise HTTPException(status_code=404, detail=f"训练任务 {run_id} 不存在")
    if not train_run.log_path or not os.path.exists(train_run.log_path):
        return {"lines": [], "total_lines": 0, "path": train_run.log_path}

    # 简单实现：读全文再切尾。训练日志通常不会超过几十 MB，先这么做，
    # 如果后面真的很大，再换成 seek 从文件尾倒着读的方式。
    with open(train_run.log_path, "r", encoding="utf-8", errors="replace") as fh:
        all_lines = fh.readlines()

    sliced = all_lines[-tail:]
    return {
        "lines": [line.rstrip("\n") for line in sliced],
        "total_lines": len(all_lines),
        "path": train_run.log_path,
    }


@app.get("/api/train-runs/{run_id}/metrics", tags=["训练"])
def get_train_run_metrics(run_id: int, db: Session = Depends(get_db)):
    """
    从日志里抽训练过程中的 loss / eval_loss 序列，供前端画曲线。

    Trainer 默认每 logging_steps 会打印一行形如：
        {'loss': 1.23, 'learning_rate': 0.0009, 'epoch': 0.05}
    评估时还会打印：
        {'eval_loss': 0.98, 'eval_runtime': ..., 'epoch': 0.2}
    这里用两个正则把它们抽出来。
    """
    import re

    train_run = db.query(TrainRun).filter(TrainRun.id == run_id).first()
    if not train_run:
        raise HTTPException(status_code=404, detail=f"训练任务 {run_id} 不存在")
    if not train_run.log_path or not os.path.exists(train_run.log_path):
        return {"train": [], "eval": []}

    loss_re = re.compile(r"'loss':\s*([0-9.eE+-]+)")
    eval_loss_re = re.compile(r"'eval_loss':\s*([0-9.eE+-]+)")
    epoch_re = re.compile(r"'epoch':\s*([0-9.eE+-]+)")

    train_points = []
    eval_points = []

    with open(train_run.log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            epoch_m = epoch_re.search(line)
            epoch = float(epoch_m.group(1)) if epoch_m else None

            ev = eval_loss_re.search(line)
            if ev:
                try:
                    eval_points.append({"epoch": epoch, "eval_loss": float(ev.group(1))})
                except ValueError:
                    pass
                continue  # eval 行不再当成 train loss

            m = loss_re.search(line)
            if m:
                try:
                    train_points.append({"epoch": epoch, "loss": float(m.group(1))})
                except ValueError:
                    pass

    return {"train": train_points, "eval": eval_points}


# ──────────────────────────────────────
# 3.4 手动触发 LoRA Merge
# ──────────────────────────────────────

FINETUNE_DIR = BACKEND_DIR / "whisper" / "finetune"
MERGE_SCRIPT = FINETUNE_DIR / "merge_lora.py"


def _run_merge_task(train_run_id: int) -> None:
    """
    后台任务：以 subprocess 运行 merge_lora.py，把 LoRA adapter 合并成完整模型。
    成功后更新 merged_model_path；失败则写 error_message。
    """
    from database import SessionLocal as _SL
    db = _SL()
    try:
        run = db.query(TrainRun).filter(TrainRun.id == train_run_id).first()
        if not run:
            return

        run.phase = "merging"
        db.commit()

        lora_path = run.checkpoint_path
        # 计算合并输出目录
        merged_parent = os.path.join(run.output_dir, "merged")
        os.makedirs(merged_parent, exist_ok=True)

        base_model = (run.base_model or "").rstrip("/")
        basename = os.path.basename(base_model)
        merged_model_dir = os.path.join(merged_parent, f"{basename}-finetune")

        cmd = [
            sys.executable, "-u", str(MERGE_SCRIPT),
            f"--lora_model={lora_path}",
            f"--output_dir={merged_parent}",
            f"--local_files_only={run.local_files_only}",
        ]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if run.gpu_id:
            env["CUDA_VISIBLE_DEVICES"] = run.gpu_id

        print(f"[MERGE] Running: {' '.join(shlex.quote(c) for c in cmd)}")

        result = subprocess.run(
            cmd,
            cwd=str(FINETUNE_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 小时超时
        )

        if result.stdout:
            for line in result.stdout.strip().split("\n")[-15:]:
                print(f"[MERGE/stdout] {line}")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-15:]:
                print(f"[MERGE/stderr] {line}")

        if result.returncode != 0:
            run.phase = "merge_failed"
            run.error_message = (
                f"merge_lora.py exited with code {result.returncode}\n"
                f"{result.stderr[-500:] if result.stderr else ''}"
            )
            db.commit()
            return

        # 验证输出目录存在
        if not os.path.isdir(merged_model_dir):
            run.phase = "merge_failed"
            run.error_message = f"Merge completed but output not found: {merged_model_dir}"
            db.commit()
            return

        run.merged_model_path = merged_model_dir
        run.phase = "done"
        run.error_message = None
        db.commit()
        print(f"[MERGE] Train run #{train_run_id} merged → {merged_model_dir}")

    except Exception as exc:
        try:
            run = db.query(TrainRun).filter(TrainRun.id == train_run_id).first()
            if run:
                run.phase = "merge_failed"
                run.error_message = f"merge exception: {exc}"
                db.commit()
        except Exception:
            pass
        print(f"[MERGE] ERROR for #{train_run_id}: {exc}")
    finally:
        db.close()


@app.post("/api/train-runs/{train_id}/merge", response_model=TrainRunDetail, tags=["训练"])
def merge_train_run(
    train_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    手动触发 LoRA adapter 合并。

    前提：训练已完成且有 checkpoint_path（LoRA adapter 路径）。
    """
    run = db.query(TrainRun).filter(TrainRun.id == train_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"训练任务 {train_id} 不存在")
    if run.status != "completed":
        raise HTTPException(status_code=400, detail=f"训练任务尚未完成 (status={run.status})")
    if not run.checkpoint_path:
        raise HTTPException(status_code=400, detail="没有找到 LoRA adapter (checkpoint_path 为空)")
    if not Path(run.checkpoint_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"LoRA adapter 路径不存在: {run.checkpoint_path}",
        )
    if run.phase == "merging":
        raise HTTPException(status_code=400, detail="正在合并中，请稍候")

    # 清除之前的 merge 错误（允许重试）
    run.error_message = None
    run.merged_model_path = None
    run.phase = "merging"
    db.commit()
    db.refresh(run)

    background_tasks.add_task(_run_merge_task, run.id)

    return TrainRunDetail.model_validate(run)


@app.delete("/api/train-runs/{train_id}", response_model=MessageResponse, tags=["训练"])
def delete_train_run(train_id: int, db: Session = Depends(get_db)):
    """
    删除训练任务记录。
    仅允许删除已终止的任务（completed / failed）。
    """
    run = db.query(TrainRun).filter(TrainRun.id == train_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"训练任务 {train_id} 不存在")
    if run.status in ("queued", "running"):
        raise HTTPException(status_code=400, detail="不能删除正在排队或运行中的任务")
    db.delete(run)
    db.commit()
    return MessageResponse(message=f"训练任务 #{train_id} 已删除")


# ──────────────────────────────────────
# 3.5 从训练任务一键发起推理评测
# ──────────────────────────────────────

BACKEND_DIR = Path(__file__).resolve().parent
BATCH_INFER_SCRIPT = BACKEND_DIR / "scripts" / "batch_infer.py"
EVAL_OUTPUT_DIR = Path(os.environ.get(
    "ASR_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "data"),
)) / "eval_outputs"
EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _run_batch_inference(evaluation) -> str:
    """
    以 subprocess 运行 batch_infer.py，返回输出 CSV 路径。
    在 BackgroundTasks 线程中同步调用。
    """
    out_csv = str(EVAL_OUTPUT_DIR / f"eval_{evaluation.id}.csv")

    # 从关联的 TrainRun 读取 language 参数
    language = "Chinese"
    if evaluation.train_run_id:
        from database import SessionLocal as _SL
        _db = _SL()
        try:
            tr = _db.query(TrainRun).filter(TrainRun.id == evaluation.train_run_id).first()
            if tr and tr.language:
                language = tr.language
        finally:
            _db.close()

    cmd = [
        sys.executable, "-u", str(BATCH_INFER_SCRIPT),
        f"--model_path={evaluation.model_path}",
        f"--test_data={evaluation.test_manifest_path}",
        f"--out_csv={out_csv}",
        f"--language={language}",
        "--batch_size=8",
        "--local_files_only=True",
    ]

    # 设置 GPU
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if evaluation.gpu_id:
        env["CUDA_VISIBLE_DEVICES"] = evaluation.gpu_id
        print(f"[EVAL] CUDA_VISIBLE_DEVICES={evaluation.gpu_id}")

    print(f"[EVAL] Running inference: {' '.join(shlex.quote(c) for c in cmd)}")

    result = subprocess.run(
        cmd,
        cwd=str(BACKEND_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,  # 2 小时超时
    )

    # 打印子进程输出便于调试
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-20:]:
            print(f"[EVAL/stdout] {line}")
    if result.stderr:
        for line in result.stderr.strip().split("\n")[-20:]:
            print(f"[EVAL/stderr] {line}")

    if result.returncode != 0:
        raise RuntimeError(
            f"batch_infer.py exited with code {result.returncode}\n"
            f"stderr: {result.stderr[-500:] if result.stderr else '(empty)'}"
        )

    if not os.path.isfile(out_csv):
        raise FileNotFoundError(f"Inference completed but output CSV not found: {out_csv}")

    print(f"[EVAL] Inference done → {out_csv}")
    return out_csv


@app.post("/api/train-runs/{train_id}/evaluate", response_model=EvalSummary, tags=["训练"])
def evaluate_train_run(
    train_id: int,
    req: TrainRunEvalRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    从一个已完成的训练任务，一键发起推理 + 评测。

    流程：
      1. 校验训练任务状态和合并模型路径
      2. 创建 Evaluation（带 model_path → 触发先推理后评测）
      3. 后台执行推理 + 评测
    """
    run = db.query(TrainRun).filter(TrainRun.id == train_id).first()
    if not run:
        raise HTTPException(status_code=404, detail=f"训练任务 {train_id} 不存在")
    if run.status != "completed":
        raise HTTPException(status_code=400, detail=f"训练任务尚未完成 (status={run.status})")
    if not run.merged_model_path:
        raise HTTPException(status_code=400, detail="训练任务没有合并模型路径，可能 merge 阶段失败")

    dataset_name = req.dataset_name.strip() if req.dataset_name.strip() else Path(req.test_data_path).stem

    evaluation = Evaluation(
        model_name=run.name,
        dataset_name=dataset_name,
        dataset_path="",                         # 推理完成后填入
        model_path=run.merged_model_path,         # ★ 触发推理阶段
        test_manifest_path=req.test_data_path,
        train_run_id=run.id,
        tokenize_mode=req.tokenize_mode,
        gpu_id=(req.gpu_id.strip() if req.gpu_id else None),
        status="pending",
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)

    background_tasks.add_task(run_evaluation, evaluation.id)

    return EvalSummary.model_validate(evaluation)


# ──────────────────────────────────────
# 4. 评测 API
# ──────────────────────────────────────

@app.post("/api/evaluations", response_model=EvalSummary, tags=["评测"])
def create_evaluation(
    req: EvalCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    发起新评测。
    
    流程：
      1. 在数据库中创建一条 status="pending" 的记录
      2. 把实际评测任务丢到后台执行（不阻塞 API 响应）
      3. 立即返回评测 ID，前端可以轮询状态
    
    为什么用后台任务：
      评测可能需要几秒到几分钟（取决于数据集大小），
      如果同步执行，HTTP 请求会超时。
      后台任务让 API 立即返回，前端通过轮询获取进度。
    """
    # 创建数据库记录
    evaluation = Evaluation(
        model_name=req.model_name,
        dataset_name=req.dataset_name,
        dataset_path=req.dataset_path,
        tokenize_mode=req.tokenize_mode,
        status="pending",
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)  # 获取自增 ID

    # 将评测任务丢到后台执行
    # run_evaluation 定义在下方
    background_tasks.add_task(run_evaluation, evaluation.id)

    return EvalSummary.model_validate(evaluation)


@app.get("/api/evaluations", response_model=list[EvalSummary], tags=["评测"])
def list_evaluations(
    status: Optional[str] = Query(None, description="按状态过滤: pending/running/completed/failed"),
    model_name: Optional[str] = Query(None, description="按模型名称过滤"),
    limit: int = Query(50, ge=1, le=200, description="返回条数"),
    offset: int = Query(0, ge=0, description="跳过条数（分页）"),
    db: Session = Depends(get_db),
):
    """
    获取评测记录列表。
    
    支持过滤（按状态、模型名）和分页。
    按创建时间倒序排列（最新的在前）。
    """
    query = db.query(Evaluation)

    if status:
        query = query.filter(Evaluation.status == status)
    if model_name:
        query = query.filter(Evaluation.model_name.ilike(f"%{model_name}%"))

    evaluations = (
        query
        .order_by(Evaluation.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return [EvalSummary.model_validate(e) for e in evaluations]


@app.get("/api/evaluations/{eval_id}", response_model=EvalFullResponse, tags=["评测"])
def get_evaluation(
    eval_id: int,
    detail_limit: int = Query(100, ge=0, le=10000, description="返回逐句明细条数"),
    detail_offset: int = Query(0, ge=0, description="逐句明细偏移"),
    sort_by: str = Query("idx", description="排序: idx(序号) / wer_desc(WER降序) / wer_asc(WER升序)"),
    db: Session = Depends(get_db),
):
    """
    获取单次评测的完整详情。
    
    包含：汇总指标 + 编辑操作分解 + WER分布 + 逐句明细（分页）。
    前端报告详情页用这个接口。
    """
    evaluation = db.query(Evaluation).filter(Evaluation.id == eval_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail=f"评测记录 {eval_id} 不存在")

    # 查询逐句明细（支持排序和分页）
    detail_query = (
        db.query(EvaluationDetail)
        .filter(EvaluationDetail.evaluation_id == eval_id)
    )

    if sort_by == "wer_desc":
        detail_query = detail_query.order_by(EvaluationDetail.wer.desc())
    elif sort_by == "wer_asc":
        detail_query = detail_query.order_by(EvaluationDetail.wer.asc())
    else:
        detail_query = detail_query.order_by(EvaluationDetail.sentence_idx)

    details = detail_query.offset(detail_offset).limit(detail_limit).all()

    # 构造响应
    response = EvalFullResponse(
        id=evaluation.id,
        model_name=evaluation.model_name,
        dataset_name=evaluation.dataset_name,
        dataset_path=evaluation.dataset_path,
        num_sentences=evaluation.num_sentences,
        corpus_wer=evaluation.corpus_wer,
        corpus_cer=evaluation.corpus_cer,
        corpus_ser=evaluation.corpus_ser,
        corpus_mer=evaluation.corpus_mer,
        corpus_wil=evaluation.corpus_wil,
        corpus_wip=evaluation.corpus_wip,
        edit_ops=EditOpsBreakdown(
            substitutions=evaluation.total_sub,
            insertions=evaluation.total_ins,
            deletions=evaluation.total_del,
            correct=evaluation.total_cor,
        ) if evaluation.status == "completed" else None,
        wer_distribution=WerDistribution(
            mean=evaluation.wer_mean or 0,
            median=evaluation.wer_median or 0,
            std=evaluation.wer_std or 0,
        ) if evaluation.status == "completed" else None,
        status=evaluation.status,
        error_message=evaluation.error_message,
        report_path=evaluation.report_path,
        created_at=evaluation.created_at,
        completed_at=evaluation.completed_at,
        details=[EvalDetailItem.model_validate(d) for d in details],
    )

    return response


@app.delete("/api/evaluations/{eval_id}", response_model=MessageResponse, tags=["评测"])
def delete_evaluation(eval_id: int, db: Session = Depends(get_db)):
    """
    删除评测记录及其关联的逐句明细。
    同时删除对应的报告文件（如果有）。
    """
    evaluation = db.query(Evaluation).filter(Evaluation.id == eval_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail=f"评测记录 {eval_id} 不存在")

    # 删除报告文件
    if evaluation.report_path and os.path.exists(evaluation.report_path):
        try:
            os.remove(evaluation.report_path)
        except OSError:
            pass  # 文件删不掉不影响数据库记录删除

    # 删除数据库记录（cascade 会自动删除关联的 details）
    db.delete(evaluation)
    db.commit()

    return MessageResponse(message=f"评测记录 {eval_id} 已删除")


# ──────────────────────────────────────
# 5. 模型对比 API
# ──────────────────────────────────────

@app.post("/api/compare", response_model=CompareResponse, tags=["对比"])
def compare_evaluations(req: CompareRequest, db: Session = Depends(get_db)):
    """
    多模型对比：传入多个评测 ID，返回对比数据。
    前端用这个数据渲染对比图表。
    """
    evaluations = (
        db.query(Evaluation)
        .filter(Evaluation.id.in_(req.evaluation_ids))
        .filter(Evaluation.status == "completed")
        .all()
    )

    if len(evaluations) < 2:
        raise HTTPException(
            status_code=400,
            detail="需要至少 2 条已完成的评测记录才能对比",
        )

    items = [
        CompareItem(
            eval_id=e.id,
            model_name=e.model_name,
            dataset_name=e.dataset_name,
            num_sentences=e.num_sentences,
            corpus_wer=e.corpus_wer,
            corpus_cer=e.corpus_cer,
            corpus_ser=e.corpus_ser,
            wer_mean=e.wer_mean,
            wer_median=e.wer_median,
            total_sub=e.total_sub,
            total_ins=e.total_ins,
            total_del=e.total_del,
            total_cor=e.total_cor,
            created_at=e.created_at,
        )
        for e in evaluations
    ]

    return CompareResponse(items=items)


# ──────────────────────────────────────
# 6. 报告导出 API
# ──────────────────────────────────────

@app.get("/api/evaluations/{eval_id}/export", tags=["报告"])
def export_report(eval_id: int, db: Session = Depends(get_db)):
    """
    下载评测报告 PDF。
    如果报告还未生成，返回 404。
    """
    evaluation = db.query(Evaluation).filter(Evaluation.id == eval_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail=f"评测记录 {eval_id} 不存在")
    
    if not evaluation.report_path or not os.path.exists(evaluation.report_path):
        raise HTTPException(status_code=404, detail="报告尚未生成或文件不存在")

    return FileResponse(
        path=evaluation.report_path,
        filename=f"eval_report_{eval_id}_{evaluation.model_name}.pdf",
        media_type="application/pdf",
    )


# ──────────────────────────────────────
# 7. 后台评测任务
# ──────────────────────────────────────

def run_evaluation(eval_id: int):
    """
    在后台执行评测任务。
    
    流程：
      1. 更新状态为 running
      2. 加载数据集
      3. 逐句计算指标（调用 eval_engine）
      4. 汇总语料级指标
      5. 保存逐句明细到数据库
      6. 生成 PDF 报告
      7. 更新状态为 completed
    
    如果任何步骤失败：
      - 更新状态为 failed
      - 记录错误信息
    
    注意：后台任务中不能使用 Depends(get_db)，
    需要手动创建和关闭数据库会话。
    """
    from database import SessionLocal

    db = SessionLocal()
    try:
        # 获取评测记录
        evaluation = db.query(Evaluation).filter(Evaluation.id == eval_id).first()
        if not evaluation:
            return

        # ── 1. 更新状态 ──
        evaluation.status = "running"
        db.commit()

        # ── 1.5 如果有 model_path，先跑推理生成 CSV ──
        if evaluation.model_path:
            print(f"[EVAL] Evaluation {eval_id} needs inference (model_path set)")
            out_csv = _run_batch_inference(evaluation)
            evaluation.dataset_path = out_csv
            db.commit()

        # ── 2. 加载数据集 ──
        pairs = load_dataset(evaluation.dataset_path)
        evaluation.num_sentences = len(pairs)
        db.commit()

        # ── 3. 导入评估引擎并计算 ──
        # eval_engine.py 将在 Step 2 实现
        # 这里先用占位逻辑，Step 2 替换为真实实现
        from eval_engine import compute_all_metrics

        sentence_metrics, corpus_metrics = compute_all_metrics(
            pairs,
            tokenize_mode=evaluation.tokenize_mode or "auto",
        )

        # ── 4. 保存逐句明细 ──
        detail_objects = []
        for m in sentence_metrics:
            detail = EvaluationDetail(
                evaluation_id=eval_id,
                sentence_idx=m["idx"] + 1,
                reference=m["reference"],
                hypothesis=m["hypothesis"],
                ref_syllables=m["ref_words"],
                hyp_syllables=m["hyp_words"],
                ref_chars=m.get("ref_chars", 0),
                hyp_chars=m.get("hyp_chars", 0),
                wer=m["wer"],
                cer=m["cer"],
                mer=m["mer"],
                wil=m["wil"],
                wip=m["wip"],
                word_sub=m["word_sub"],
                word_ins=m["word_ins"],
                word_del=m["word_del"],
                word_cor=m["word_cor"],
                is_correct=m["is_correct"],
            )
            detail_objects.append(detail)

        db.bulk_save_objects(detail_objects)

        # ── 5. 保存语料级指标 ──
        evaluation.corpus_wer = corpus_metrics["corpus_wer"]
        evaluation.corpus_cer = corpus_metrics["corpus_cer"]
        evaluation.corpus_ser = corpus_metrics["corpus_ser"]
        evaluation.corpus_mer = corpus_metrics["corpus_mer"]
        evaluation.corpus_wil = corpus_metrics["corpus_wil"]
        evaluation.corpus_wip = corpus_metrics["corpus_wip"]
        evaluation.total_sub = corpus_metrics["total_word_sub"]
        evaluation.total_ins = corpus_metrics["total_word_ins"]
        evaluation.total_del = corpus_metrics["total_word_del"]
        evaluation.total_cor = corpus_metrics["total_word_cor"]
        evaluation.wer_mean = corpus_metrics["wer_mean"]
        evaluation.wer_median = corpus_metrics["wer_median"]
        evaluation.wer_std = corpus_metrics["wer_std"]

        # ── 6. 生成 PDF 报告（Step 2 实现） ──
        try:
            from eval_engine import generate_report
            report_dir = os.path.join(
                os.environ.get("ASR_DATA_DIR", os.path.join(os.path.dirname(__file__), "data")),
                "reports"
            )
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, f"report_{eval_id}.pdf")
            generate_report(sentence_metrics, corpus_metrics, report_path)
            evaluation.report_path = report_path
        except Exception as e:
            print(f"[WARN] 报告生成失败: {e}")
            # 报告生成失败不影响评测完成

        # ── 7. 标记完成 ──
        evaluation.status = "completed"
        evaluation.completed_at = datetime.utcnow()
        db.commit()

        print(f"[INFO] 评测 {eval_id} 完成: WER={evaluation.corpus_wer:.4f}")

    except Exception as e:
        # ── 出错：标记失败 ──
        db.rollback()
        try:
            evaluation = db.query(Evaluation).filter(Evaluation.id == eval_id).first()
            if evaluation:
                evaluation.status = "failed"
                evaluation.error_message = str(e)[:1000]  # 截断，防止过长
                db.commit()
        except Exception:
            pass
        print(f"[ERROR] 评测 {eval_id} 失败: {e}")

    finally:
        db.close()


# ──────────────────────────────────────
# 8. 健康检查
# ──────────────────────────────────────

@app.get("/api/health", tags=["系统"])
def health_check():
    """健康检查，用于监控和负载均衡器探活"""
    return {
        "status": "ok",
        "dataset_dir": DATASET_BASE_DIR,
        "version": "1.0.0",
    }


# ──────────────────────────────────────
# 9. 启动入口
# ──────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,         # 开发模式：文件修改后自动重启
    )
