"""
schemas.py - API 数据校验模型
==========================================

职责:
1. 定义 API 请求体(Request Body)的数据结构和校验规则.
2. 定义 API 响应体(Response Body) 的数据结构.
3. 在请求进来时自动校验, 格式不对直接返回422错误
为什么需要这个校验逻辑,而不是直接用dict
  - 自动类型校验: 前端传错类型直接报错,不会污染数据库
  - 自动文档生成: 根据这些模型生成Swagger文档,前端一目了然
  - 前后端契约: 前端开发者看这个文件就知道要传什么字段

命名规范:
  - XxxCreate = 创建时传入的字段
  - XxxResponse = 返回给前端的字段
  - XxxSummary = 列表页用的精简版本
"""

from datatime import datetime
from typing import Optional, List
from Pydantic import BaseModel, Field


# ──────────────────────────────────────
# 1. 评测相关
# ──────────────────────────────────────

class EvalCreate(BaseModel):
    """
    发起评测时, 前端需要传的字段
    示例请求体:
    {
        "model_name": "gpt-3.5-turbo",
        "dataset_name": "squad",
        "dataset_path": "s3://my-bucket/squad.json"
    }
    """
    model_name: str = Filed(
        ...,
        min_length=1,
        max_length=200,
        description="评测使用的模型名称, 例如 whisper-large-v3-turbo",
        example=["whisper-large-v3-turbo", "whisper-small"]
    )

    dataset_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="数据集显示名称",
        example=["common-voice-test"]
    )
    dataset_path: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="数据集存储路径, 例如 s3://my-bucket/squad.json",
        example=["s3://my-bucket/squad.json", "gs://my-bucket/dataset.csv"]
    )

class EvalSummary(BaseModel):
    """
    评测列表页使用的精简模型
    不返回逐句明细, 只返回指标汇总, 加快列表加载速度
    """
    id: int
    model_name: str
    dataset_name: str

    # 核心指标(可能为None, 因为评测可能还在进行中)
    corpus_wer: Optional[float] = None
    corpus_cer: Optional[float] = None
    corpus_ser: Optional[float] = None

    status: str
    create_at: datetime
    complete_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
    # from_attributes=True 允许直接从ORM模型实例创建这个Pydantic模型, 省去手动转换的麻烦
    # 即: EvalSummary.model_validate(db_evaluation_object) 就能直接得到一个 EvalSummary 实例

class EvalDetailItem(BaseModel):
    """
    单条句子的评测明细
    用在评测报告详情页, 展示每条句子的指标
    """
    sentence_idx: int
    reference: str
    hypothesis: str
    ref_syllables: int
    hyp_syllables: int
    wer: float
    cer: float
    word_sub: int
    word_ins: int
    word_del: int
    word_cor: int
    is_correct: bool
    model_config = {"from_attributes": True}

class EditOpsBreakdown(BaseModel):
    """
    编辑操作的详细分解
    """
    substitutions: int
    insertions: int
    deletions: int
    correct: int

class WerDistribution(BaseModel):
    """
    WER分布情况
    """
    mean: float
    median: float
    std: float

class EvalFullResponse(BaseModel):
    """
    单词评测的完整响应体, 包含:
    - 汇总指标
    - 编辑操作分解
    - WER分布
    - 逐句明细列表(可分页)
    用在评测报告详情页, 前端可以根据需要选择展示哪些部分
    """
    id: int
    model_name: str
    dataset_name: str
    dataset_path: str
    num_sentences: int

    # 语料级指标
    corpus_wer: Optional[float] = None
    corpus_cer: Optional[float] = None
    corpus_ser: Optional[float] = None
    corpus_mer: Optional[float] = None
    corpus_wil: Optional[float] = None
    corpus_wip: Optional[float] = None

    # 编辑操作
    edit_ops: Optional[EditOpsBreakdown] = None

    # WER分布
    wer_distribution: Optional[WerDistribution] = None

    # 状态
    status: str
    error_message: Optional[str] = None
    report_path: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    # 逐句明细(可分页)
    details: List[EvalDetailItem] = []

    # model_config = {"from_attributes": True}  # 允许从ORM对象直接创建Pydantic模型实例


# ──────────────────────────────────────
# 2. 数据集相关
# ──────────────────────────────────────

class DatasetInfo(BaseModel):
    """
    数据集信息模型, 由后端扫描本地目录后返回
    前端展示为一个下拉列表, 用户选择后发起评测
    """
    name: str = Field(
        ...,
        description="数据集名称, 例如 common-voice-test",
    )
    path: str = Field(
        ...,
        description="数据集存储路径, 例如 s3://my-bucket/squad.json",
    )
    file_count: int = Field(0,description="数据集中的文件数量")
    total_rows: Optional[int] = Field(None, description="数据集中的总行数, 可能需要扫描文件才能得到")
    format: str = Field("unknown", description="数据集格式, 例如 jsonl, csv, json,huggingface,directory等")
    size_md: float = Field(0.0, description="数据集大小, 单位MB")

class DatasetListResponse(BaseModel):
    """
    数据集列表响应体, 包含多个数据集的信息
    """
    datasets: List[DatasetInfo]
    base_dir: str = Field(..., description="数据集的基础目录, 前端可以根据这个路径构造完整的文件路径")

# ──────────────────────────────────────
# 3. 模型对比相关
# ──────────────────────────────────────

class CompareRequest(BaseModel):
    """
    模型对比请求: 传入多个评测ID, 返回对比数据.
    示例:
    {
        "eval_ids": [1, 2, 3]
    }
    """
    evalution_ids: List[int] = Field(..., min_length=2, max_length=10,description="要对比的评测ID列表")

class CompareItem(BaseModel):
    """
    对比表中的单行: 一个模型在某数据集上的指标
    """
    eval_id: int
    model_name: str
    dataset_name: str
    num_sentences: int
    corpus_wer: Optional[float] = None
    corpus_cer: Optional[float] = None
    corpus_ser: Optional[float] = None
    wer_mean: Optional[float] = None
    wer_median: Optional[float] = None
    total_sub: int = 0
    total_ins: int = 0
    total_del: int = 0
    total_cor: int = 0
    create_at: datetime

class CompareResponse(BaseModel):
    """
    模型对比响应体: 包含多个CompareItem
    """
    comparisons: List[CompareItem]

# ──────────────────────────────────────
# 4. 通用响应体
# ──────────────────────────────────────

class MessageResponse(BaseModel):
    """
    通用消息响应体, 包含一个message字段
    用于返回错误信息或成功提示
    """
    message: str
    success: bool = True
