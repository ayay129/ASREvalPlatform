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
  POST /api/evaluations          发起新评测
  GET  /api/evaluations          获取评测记录列表
  GET  /api/evaluations/{id}     获取单次评测详情
  DEL  /api/evaluations/{id}     删除评测记录
  POST /api/compare              多模型对比
  GET  /api/evaluations/{id}/export  导出报告 PDF

启动方式：
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from database import init_db, get_db, Evaluation, EvaluationDetail
from schemas import (
    EvalCreate, EvalSummary, EvalFullResponse, EvalDetailItem,
    EditOpsBreakdown, WerDistribution,
    DatasetListResponse,
    CompareRequest, CompareResponse, CompareItem,
    MessageResponse,
)
from dataset_loader import scan_datasets, load_dataset, DATASET_BASE_DIR


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
# 2. 数据集 API
# ──────────────────────────────────────

@app.get("/api/datasets", response_model=DatasetListResponse, tags=["数据集"])
def list_datasets():
    """
    获取可用数据集列表。
    
    扫描 GPU 服务器上的数据集目录，返回所有可识别的数据集。
    前端用这个接口填充"选择数据集"下拉列表。
    """
    datasets = scan_datasets()
    return DatasetListResponse(
        datasets=datasets,
        base_dir=DATASET_BASE_DIR,
    )


# ──────────────────────────────────────
# 3. 评测 API
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
# 4. 模型对比 API
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
# 5. 报告导出 API
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
# 6. 后台评测任务
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

        # ── 2. 加载数据集 ──
        pairs = load_dataset(evaluation.dataset_path)
        evaluation.num_sentences = len(pairs)
        db.commit()

        # ── 3. 导入评估引擎并计算 ──
        # eval_engine.py 将在 Step 2 实现
        # 这里先用占位逻辑，Step 2 替换为真实实现
        from eval_engine import compute_all_metrics

        sentence_metrics, corpus_metrics = compute_all_metrics(pairs)

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
# 7. 健康检查
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
# 8. 启动入口
# ──────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,         # 开发模式：文件修改后自动重启
    )
