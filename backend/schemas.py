"""
schemas.py — API 数据校验模型 (Pydantic)
==========================================

职责：
  1. 定义 API 请求体（Request Body）的格式和校验规则
  2. 定义 API 响应体（Response Body）的格式
  3. 在请求进来时自动校验，格式不对直接返回 422 错误

为什么需要这个文件（而不是直接用 dict）：
  - 自动类型校验：前端传错类型直接报错，不会污染数据库
  - 自动文档生成：FastAPI 会根据这些模型生成 Swagger 文档
  - 前后端契约：前端开发者看这个文件就知道要传什么字段

命名规范：
  XxxCreate  = 创建时前端传入的字段
  XxxResponse = 返回给前端的字段
  XxxSummary  = 列表页用的精简版
"""

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


# ──────────────────────────────────────
# 1. 评测相关
# ──────────────────────────────────────

class EvalCreate(BaseModel):
    """
    发起新评测时，前端需要传的字段。
    
    示例请求体：
    {
        "model_name": "whisper-large-v3-tibetan",
        "dataset_name": "common-voice-test",
        "dataset_path": "/data/datasets/common-voice-test"
    }
    """
    model_name: str = Field(
        ...,                          # ... 表示必填
        min_length=1,                 # 不能为空字符串
        max_length=200,
        description="模型名称，如 whisper-large-v3-tibetan",
        examples=["whisper-large-v3-tibetan"],
    )
    dataset_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="数据集显示名称",
        examples=["common-voice-test"],
    )
    dataset_path: str = Field(
        ...,
        min_length=1,
        description="数据集在服务器上的路径（CSV 文件路径或目录路径）",
        examples=["/data/datasets/common-voice-test/test.csv"],
    )
    tokenize_mode: str = Field(
        "auto",
        description="分词模式: auto(自动检测语言) / whisper(Whisper tokenizer) / char(按字符) / space(按空格)",
        examples=["auto"],
    )


class EvalSummary(BaseModel):
    """
    评测列表页使用的精简模型。
    不返回逐句明细，只返回汇总指标，加快列表加载速度。
    """
    id: int
    model_name: str
    dataset_name: str
    num_sentences: int

    # 核心指标（可能为 None，因为评测可能还在进行中）
    corpus_wer: Optional[float] = None
    corpus_cer: Optional[float] = None
    corpus_ser: Optional[float] = None

    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
    # from_attributes=True 允许从 SQLAlchemy 模型对象直接构造
    # 即：EvalSummary.model_validate(db_evaluation_object)


class EvalDetailItem(BaseModel):
    """
    单条句子的评测明细。
    用在评测报告详情页，展示每条句子的指标。
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
    """编辑操作分解"""
    substitutions: int
    insertions: int
    deletions: int
    correct: int


class WerDistribution(BaseModel):
    """WER 分布统计"""
    mean: float
    median: float
    std: float


class EvalFullResponse(BaseModel):
    """
    单次评测的完整响应，包含：
    - 汇总指标
    - 编辑操作分解
    - WER 分布
    - 逐句明细列表（可分页）
    
    用在评测报告详情页。
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

    # WER 分布
    wer_distribution: Optional[WerDistribution] = None

    # 状态
    status: str
    error_message: Optional[str] = None
    report_path: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    # 逐句明细
    details: List[EvalDetailItem] = []

    model_config = {"from_attributes": True}


# ──────────────────────────────────────
# 2. 数据集相关
# ──────────────────────────────────────

class DatasetInfo(BaseModel):
    """
    数据集信息，由后端扫描本地目录后返回。
    
    前端展示为一个下拉列表，用户选择后发起评测。
    """
    name: str = Field(..., description="数据集名称（目录名或文件名）")
    path: str = Field(..., description="数据集完整路径")
    file_count: int = Field(0, description="包含的文件数量")
    total_rows: Optional[int] = Field(None, description="CSV 的总行数（如果是单文件）")
    format: str = Field("unknown", description="格式：csv / huggingface / directory")
    size_mb: float = Field(0.0, description="文件大小 (MB)")


class DatasetListResponse(BaseModel):
    """数据集列表响应"""
    datasets: List[DatasetInfo]
    base_dir: str = Field(..., description="数据集根目录")


# ──────────────────────────────────────
# 3. 训练任务相关
# ──────────────────────────────────────

class TrainRunCreate(BaseModel):
    """
    发起微调训练任务时前端需要传的字段。

    Step 1 只负责保存配置，不真正启动训练。
    """
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="训练任务显示名称",
        examples=["tibetan-whisper-small-v1"],
    )
    base_model: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Whisper 基础模型名称或本地路径",
        examples=["openai/whisper-small"],
    )
    train_data_path: str = Field(
        ...,
        min_length=1,
        description="训练集 manifest 路径",
        examples=["/data/manifests/tibetan/train.json"],
    )
    test_data_path: str = Field(
        ...,
        min_length=1,
        description="验证/测试集 manifest 路径",
        examples=["/data/manifests/tibetan/test.json"],
    )
    output_dir: str = Field(
        "output/",
        min_length=1,
        description="训练输出目录",
        examples=["/data/whisper/output"],
    )
    language: str = Field(
        "Chinese",
        description="Whisper 训练语言参数",
        examples=["Chinese"],
    )
    task: str = Field(
        "transcribe",
        description="Whisper 任务类型: transcribe / translate",
        examples=["transcribe"],
    )
    timestamps: bool = Field(False, description="是否使用时间戳训练")
    num_train_epochs: int = Field(3, ge=1, description="训练轮数")
    learning_rate: float = Field(1e-3, gt=0, description="学习率")
    warmup_steps: int = Field(50, ge=0, description="预热步数")
    logging_steps: int = Field(100, ge=1, description="日志打印步数")
    eval_steps: int = Field(1000, ge=1, description="评估步数")
    save_steps: int = Field(1000, ge=1, description="保存 checkpoint 步数")
    per_device_train_batch_size: int = Field(8, ge=1, description="单卡训练 batch size")
    per_device_eval_batch_size: int = Field(8, ge=1, description="单卡评估 batch size")
    gradient_accumulation_steps: int = Field(1, ge=1, description="梯度累积步数")
    save_total_limit: int = Field(10, ge=1, description="最多保留 checkpoint 数量")
    use_adalora: bool = Field(True, description="是否使用 AdaLora")
    use_8bit: bool = Field(False, description="是否加载 8-bit 模型")
    fp16: bool = Field(True, description="是否开启 fp16")
    use_compile: bool = Field(False, description="是否开启 torch.compile")
    local_files_only: bool = Field(False, description="是否只从本地加载模型")
    push_to_hub: bool = Field(False, description="训练完成后是否 push 到 Hugging Face Hub")
    augment_config_path: Optional[str] = Field(None, description="数据增强配置文件路径")
    resume_from_checkpoint: Optional[str] = Field(None, description="恢复训练的 checkpoint 路径")
    hub_model_id: Optional[str] = Field(None, description="推送到 Hub 时的仓库 ID")


class TrainRunSummary(BaseModel):
    """训练任务列表页使用的精简模型。"""
    id: int
    name: str
    base_model: str
    train_data_path: str
    test_data_path: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class TrainRunDetail(BaseModel):
    """训练任务详情。"""
    id: int
    name: str
    base_model: str
    train_data_path: str
    test_data_path: str
    output_dir: str
    language: str
    task: str
    timestamps: bool
    num_train_epochs: int
    learning_rate: float
    warmup_steps: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    save_total_limit: int
    use_adalora: bool
    use_8bit: bool
    fp16: bool
    use_compile: bool
    local_files_only: bool
    push_to_hub: bool
    augment_config_path: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    hub_model_id: Optional[str] = None
    status: str
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ──────────────────────────────────────
# 4. 模型对比相关
# ──────────────────────────────────────

class CompareRequest(BaseModel):
    """
    模型对比请求：传入多个评测 ID，返回对比数据。
    
    示例：
    {
        "evaluation_ids": [1, 3, 5]
    }
    """
    evaluation_ids: List[int] = Field(
        ...,
        min_length=2,            # 至少选 2 个才有对比意义
        max_length=10,           # 最多 10 个，防止页面过于拥挤
        description="要对比的评测记录 ID 列表",
    )


class CompareItem(BaseModel):
    """对比表中的单行：一个模型在某数据集上的指标"""
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
    created_at: datetime


class CompareResponse(BaseModel):
    """模型对比响应"""
    items: List[CompareItem]


# ──────────────────────────────────────
# 5. 通用响应
# ──────────────────────────────────────

class MessageResponse(BaseModel):
    """通用消息响应，用于删除、状态更新等操作"""
    message: str
    success: bool = True
