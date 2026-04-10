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
"""

import os
from datetime import datetime

from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime,
    Boolean, ForeignKey, create_engine
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


# ──────────────────────────────────────
# 3. 工具函数
# ──────────────────────────────────────

def init_db():
    """
    初始化数据库：创建所有表（如果不存在）。
    在 FastAPI 启动时调用一次即可。
    """
    Base.metadata.create_all(bind=engine)


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
