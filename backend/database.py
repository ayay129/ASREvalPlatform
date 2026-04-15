"""
database.py — 数据库模型与连接管理
===================================

职责：
  1. 创建 SQLite 数据库连接（文件存储在 /data/asr_platform.db）
  2. 定义 ORM 模型（表结构）
  3. 提供数据库会话（Session）给 API 层使用

为什么选 SQLite：
  - 零配置，无需安装数据库服务器
  - 单文件存储，方便备份和迁移
  - 对于评测平台的读写频率完全够用

表设计：
  evaluations      — 每次评测的汇总信息（一行 = 一次评测）
  evaluation_details — 每次评测中每条句子的详细指标
  train_runs       — 每次微调训练任务的配置与状态
"""

import os
from datetime import datetime

from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime,
    Boolean, ForeignKey, create_engine, inspect, text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker


# ──────────────────────────────────────
# 1. 数据库连接配置
# ──────────────────────────────────────

# 数据库文件路径，可通过环境变量自定义
# 默认放在项目根目录的 data/ 下
DATA_DIR = os.environ.get("ASR_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
os.makedirs(DATA_DIR, exist_ok=True)

DATABASE_URL = f"sqlite:///{os.path.join(DATA_DIR, 'asr_platform.db')}"

# create_engine：创建数据库引擎
#   - check_same_thread=False：允许 FastAPI 的异步框架跨线程访问 SQLite
#   - echo=False：不打印 SQL 语句（调试时可改为 True）
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

# sessionmaker：创建会话工厂
#   - autocommit=False：需要手动 commit，防止意外写入
#   - autoflush=False：手动控制何时刷新到数据库
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# declarative_base：ORM 模型的基类，所有表模型都继承它
Base = declarative_base()


# ──────────────────────────────────────
# 2. 表模型定义
# ──────────────────────────────────────

class Evaluation(Base):
    """
    evaluations 表 — 评测记录汇总
    
    每次用户发起一次评测，就会创建一行记录。
    存储：模型名称、数据集信息、语料级指标、任务状态等。
    
    字段说明：
      id              自增主键
      model_name      模型名称（如 "whisper-large-v3-tibetan"）
      dataset_name    数据集名称（如 "tibetan-common-voice-test"）
      dataset_path    数据集在服务器上的绝对路径
      num_sentences   评测的句子总数
      
      corpus_wer      语料级词错误率 (Word Error Rate)
      corpus_cer      语料级字符错误率 (Character Error Rate)
      corpus_ser      语料级句错误率 (Sentence Error Rate)
      corpus_mer      匹配错误率 (Match Error Rate)
      corpus_wil      词信息丢失率 (Word Information Lost)
      corpus_wip      词信息保留率 (Word Information Preserved)
      
      total_sub       总替换次数 (Substitutions)
      total_ins       总插入次数 (Insertions)
      total_del       总删除次数 (Deletions)
      total_cor       总正确次数 (Correct)
      
      wer_mean/median/std  WER 统计分布
      
      status          任务状态：pending/running/completed/failed
      error_message   失败时的错误信息
      report_path     生成的 PDF 报告路径
      
      created_at      创建时间
      completed_at    完成时间
      
      details         关联的逐句明细（一对多关系）
    """
    __tablename__ = "evaluations"

    # ── 基本信息 ──
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(200), nullable=False, index=True)
    dataset_name = Column(String(200), nullable=False)
    dataset_path = Column(Text, nullable=False)
    tokenize_mode = Column(String(20), default="auto")  # auto/whisper/char/space
    num_sentences = Column(Integer, default=0)

    # ── 语料级指标 ──
    corpus_wer = Column(Float, nullable=True)
    corpus_cer = Column(Float, nullable=True)
    corpus_ser = Column(Float, nullable=True)
    corpus_mer = Column(Float, nullable=True)
    corpus_wil = Column(Float, nullable=True)
    corpus_wip = Column(Float, nullable=True)

    # ── 编辑操作统计 ──
    total_sub = Column(Integer, default=0)
    total_ins = Column(Integer, default=0)
    total_del = Column(Integer, default=0)
    total_cor = Column(Integer, default=0)

    # ── WER 分布统计 ──
    wer_mean = Column(Float, nullable=True)
    wer_median = Column(Float, nullable=True)
    wer_std = Column(Float, nullable=True)

    # ── 任务状态 ──
    status = Column(String(20), default="pending", index=True)
    error_message = Column(Text, nullable=True)
    report_path = Column(Text, nullable=True)

    # ── 时间戳 ──
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # ── 关联：一次评测 → 多条逐句明细 ──
    # cascade="all, delete-orphan" 表示删除评测时自动删除关联的明细
    details = relationship(
        "EvaluationDetail",
        back_populates="evaluation",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return (
            f"<Evaluation(id={self.id}, model='{self.model_name}', "
            f"dataset='{self.dataset_name}', wer={self.corpus_wer}, "
            f"status='{self.status}')>"
        )


class EvaluationDetail(Base):
    """
    evaluation_details 表 — 逐句评测明细
    
    每次评测中的每条句子，都会存一行记录。
    用于：查看哪些句子识别得好/差、按长度分析、导出明细 CSV。
    
    字段说明：
      id              自增主键
      evaluation_id   关联的评测记录 ID（外键）
      sentence_idx    句子在数据集中的序号（从 1 开始）
      
      reference       真实文本（ground truth）
      hypothesis      预测文本（模型输出）
      
      ref_syllables   参考文本音节数
      hyp_syllables   预测文本音节数
      ref_chars       参考文本字符数
      hyp_chars       预测文本字符数
      
      wer/cer/mer/wil/wip  该句子的各项指标
      
      word_sub/ins/del/cor  该句子的编辑操作计数
      is_correct      该句子是否完全正确
    """
    __tablename__ = "evaluation_details"

    # ── 基本信息 ──
    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(
        Integer,
        ForeignKey("evaluations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sentence_idx = Column(Integer, nullable=False)

    # ── 文本 ──
    reference = Column(Text, nullable=False)
    hypothesis = Column(Text, nullable=False)

    # ── 长度 ──
    ref_syllables = Column(Integer, default=0)
    hyp_syllables = Column(Integer, default=0)
    ref_chars = Column(Integer, default=0)
    hyp_chars = Column(Integer, default=0)

    # ── 指标 ──
    wer = Column(Float, default=0.0)
    cer = Column(Float, default=0.0)
    mer = Column(Float, default=0.0)
    wil = Column(Float, default=0.0)
    wip = Column(Float, default=0.0)

    # ── 编辑操作 ──
    word_sub = Column(Integer, default=0)
    word_ins = Column(Integer, default=0)
    word_del = Column(Integer, default=0)
    word_cor = Column(Integer, default=0)

    # ── 是否完全正确 ──
    is_correct = Column(Boolean, default=False)

    # ── 反向关联 ──
    evaluation = relationship("Evaluation", back_populates="details")


class TrainRun(Base):
    """
    train_runs 表 — 微调训练任务

    Step 1 先只负责保存训练配置和任务状态，
    后续再由独立 worker 消费 queued 任务并执行真实训练。
    """
    __tablename__ = "train_runs"

    # ── 基本信息 ──
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    base_model = Column(String(255), nullable=False, index=True)
    train_data_path = Column(Text, nullable=False)
    test_data_path = Column(Text, nullable=False)
    output_dir = Column(Text, nullable=False)

    # ── 训练参数 ──
    language = Column(String(100), default="Chinese")
    task = Column(String(20), default="transcribe")
    timestamps = Column(Boolean, default=False)
    num_train_epochs = Column(Integer, default=3)
    learning_rate = Column(Float, default=1e-3)
    warmup_steps = Column(Integer, default=50)
    logging_steps = Column(Integer, default=100)
    eval_steps = Column(Integer, default=1000)
    save_steps = Column(Integer, default=1000)
    per_device_train_batch_size = Column(Integer, default=8)
    per_device_eval_batch_size = Column(Integer, default=8)
    gradient_accumulation_steps = Column(Integer, default=1)
    save_total_limit = Column(Integer, default=10)

    # ── 运行开关 ──
    use_adalora = Column(Boolean, default=True)
    use_8bit = Column(Boolean, default=False)
    fp16 = Column(Boolean, default=True)
    use_compile = Column(Boolean, default=False)
    local_files_only = Column(Boolean, default=False)
    push_to_hub = Column(Boolean, default=False)

    # ── 可选路径/扩展配置 ──
    augment_config_path = Column(Text, nullable=True)
    resume_from_checkpoint = Column(Text, nullable=True)
    hub_model_id = Column(String(255), nullable=True)

    # ── 任务状态 ──
    status = Column(String(20), default="queued", index=True)
    error_message = Column(Text, nullable=True)

    # ── 进度追踪 ──
    # worker 在运行 finetune.py 时会实时解析 stdout 并把以下字段回写，
    # 供前端轮询或后续 SSE 展示训练进度、损失曲线
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=0)
    current_epoch = Column(Float, default=0.0)
    current_loss = Column(Float, nullable=True)
    current_eval_loss = Column(Float, nullable=True)
    log_path = Column(Text, nullable=True)
    # LoRA adapter 的输出路径（finetune.py 产物）
    checkpoint_path = Column(Text, nullable=True)
    # merge_lora.py 合并后的完整模型路径，后续推理/评测用这个
    merged_model_path = Column(Text, nullable=True)
    # 当前阶段：finetune / merging / done
    phase = Column(String(20), nullable=True)
    pid = Column(Integer, nullable=True)

    # ── 时间戳 ──
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return (
            f"<TrainRun(id={self.id}, name='{self.name}', "
            f"base_model='{self.base_model}', status='{self.status}')>"
        )


class Dataset(Base):
    """
    datasets 表 — 数据集注册表

    平台只认"注册过"的数据集。注册有两种来源：
      1. 磁盘扫描 (source='local')：walk DATASET_BASE_DIR，识别
         CSV / JSONL / manifest 后 upsert 一行
      2. HuggingFace 拉取 (source='huggingface')：走 DatasetPull
         下载完再扫描入库，本字段会填 source_repo

    kind 决定它能被用到哪里：
      - eval_csv       → NewEval 可用（含 transcription+predicted_string 列的 CSV）
      - train_manifest → NewTrainRun 可用（Whisper-Finetune JSONL manifest）
    """
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    kind = Column(String(32), nullable=False, index=True)   # eval_csv / train_manifest
    path = Column(Text, nullable=False, unique=True)

    # 元信息
    rows = Column(Integer, nullable=True)
    size_bytes = Column(Integer, nullable=True)
    duration_sec = Column(Float, nullable=True)   # 仅对 train_manifest 有值
    language = Column(String(64), nullable=True)

    # 来源
    source = Column(String(32), default="local")   # local / huggingface
    source_repo = Column(String(255), nullable=True)
    source_split = Column(String(64), nullable=True)

    # 状态
    status = Column(String(20), default="ready")   # ready / missing
    note = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Dataset(id={self.id}, name='{self.name}', kind='{self.kind}')>"


class DatasetPull(Base):
    """
    dataset_pulls 表 — HuggingFace 数据集拉取任务

    worker 消费 queued 任务，调用 snapshot_download 到 DATASET_BASE_DIR 下，
    完成后触发 scan 把里面可识别的文件注册进 datasets 表。
    """
    __tablename__ = "dataset_pulls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    repo_id = Column(String(255), nullable=False, index=True)
    revision = Column(String(128), nullable=True)
    local_dir = Column(Text, nullable=True)   # worker 填入实际落盘目录

    status = Column(String(20), default="queued", index=True)  # queued/running/completed/failed
    error_message = Column(Text, nullable=True)
    log_tail = Column(Text, nullable=True)    # 保存最后几行 stdout，便于前端展示

    # 拉完后回填：创建了多少条 datasets 记录
    registered_count = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<DatasetPull(id={self.id}, repo_id='{self.repo_id}', status='{self.status}')>"


# ──────────────────────────────────────
# 3. 工具函数
# ──────────────────────────────────────

def _migrate_train_runs():
    """
    手动给 train_runs 表做兼容性迁移。

    SQLite 的 `create_all` 只创建新表，不会给已有表加列。
    这里读一次表结构，把 ORM 里新增的列逐一 ALTER 出来，
    避免老数据库升级后出现 `no such column` 错误。

    只处理 ADD COLUMN，不做复杂的类型迁移/重命名。
    """
    expected_columns = {
        "current_step": "INTEGER DEFAULT 0",
        "total_steps": "INTEGER DEFAULT 0",
        "current_epoch": "FLOAT DEFAULT 0.0",
        "current_loss": "FLOAT",
        "current_eval_loss": "FLOAT",
        "log_path": "TEXT",
        "checkpoint_path": "TEXT",
        "merged_model_path": "TEXT",
        "phase": "VARCHAR(20)",
        "pid": "INTEGER",
    }

    inspector = inspect(engine)
    if "train_runs" not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns("train_runs")}
    missing = [(name, ddl) for name, ddl in expected_columns.items() if name not in existing]
    if not missing:
        return

    with engine.begin() as conn:
        for name, ddl in missing:
            conn.execute(text(f"ALTER TABLE train_runs ADD COLUMN {name} {ddl}"))
            print(f"[DB] Added column train_runs.{name}")


def init_db():
    """
    初始化数据库：创建所有表（如果不存在），并给老表补齐新列。
    在 FastAPI 启动时调用一次即可。
    """
    Base.metadata.create_all(bind=engine)
    _migrate_train_runs()


def get_db():
    """
    获取数据库会话的依赖注入函数。
    
    在 FastAPI 中这样使用：
        @app.get("/xxx")
        def some_api(db: Session = Depends(get_db)):
            ...
    
    yield 语法确保：
      - 请求开始时创建 session
      - 请求结束后自动关闭 session（即使出错也会关闭）
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
