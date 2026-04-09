"""
database.py - 数据库模型与连接管理
职责:
1. 创建 SQLite 数据库连接(文件存储在/data/asr_platform.db)
2. 定义ORM模型(表结构)
3. 提供数据库会话(Session)管理工具
选SQLite的依据:
- 零配置, 无需安装数据库服务器
- 单文件存储,方便备份和迁移
- 对于评测平台的读写频率完全够用
表设计:
evaluations - 每次评测的汇总信息(一行 = 一次评测)
evaluation_details- 每次评测中每条句子的详细指标
"""
import os
from datatime import datetime
from sqlalchemy import (
    Column,Integer,Float,String,Text,DateTime,
    Boolean,create_engine
)
from sqlalchemy.orm import declarative_base, relationship,sessionmaker

# ──────────────────────────────────────
# 1. 数据库连接配置
# ──────────────────────────────────────

# 数据库文件路径，可通过环境变量 ASR_DATA_DIR自定义
# 默认放在项目根目录的 data/ 下
DATA_DIR = os.environ.get("ASR_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
os.makedirs(DATA_DIR, exist_ok=True)  # 确保目录存在
DATABASE_URL = f"sqlite:///{os.path.join(DATA_DIR, 'asr_platform.db')}"
# 创建数据库引擎
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # SQLite特有配置，允许多线程访问
    echo=False  # 关闭SQLAlchemy的日志输出，生产环境建议关闭
)
# 创建会话工厂
# sessionmaker是SQLAlchemy提供的一个工厂函数，用于创建新的数据库会话(Session)对象
# - autocommit=False: 需要手动提交事务，增加数据安全性
# - autoflush=False: 需要手动刷新数据到数据库，避免不必要的性能开销
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# declarative_base()函数用于创建一个ORM基类，所有的ORM模型都将继承这个基类

# ──────────────────────────────────────
# 2. 表模型定义
# ──────────────────────────────────────
class Evaluation(Base):
    """
    elvaluations评测表 - 评测记录汇总
    用户每次发起一次评测,就会创建一行记录
    存储: 模型名称, 数据集信息, 语料级指标,任务状态等
    字段说明:
    - id: 主键，自增
    - model_name: 评测使用的ASR模型名称,如“whisper-large-v3-turbo"
    - dataset_name: 评测数据集名称
    - dataset_path: 数据集在服务器上的绝对路径
    - num_sentences: 评测的句子总数

    - corpus_wer: 语料级词错误率(Word Error Rate)
    - corpus_cer: 语料级字符错误率(Character Error Rate)
    - corpus_ser: 语料级句子错误率(Sentence Error Rate)
    - corpus_mer: 批评错误率(Match Error Rate)
    - corpus_wil: 词信息丢失率(Word Information Lost)
    - corpus_wip: 词信息保留率(Word Information Preserved)

    - total_sub: 总替换次数(Substitutions)
    - total_ins: 总插入次数(Insertions)
    - total_del: 总删除次数(Deletions)
    - total_cor: 总正确次数(Correct)

    - wer_mean/median/std: 评测中每条句子的WER的平均值/中位数/标准差

    - status 任务状态: pending/running/completed/failed
    - error_message 失败时的错误信息
    - report_path 生成的pdf报告路径

    - created_at 创建时间
    - completed_at 完成时间

    - detials 关联的逐句明细(一对多)
    """
    __tablename__ = "evaluations"
    # 基本信息
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(255), nullable=False,index=True)
    dataset_name = Column(String(200), nullable=False)
    num_sentences = Column(Integer, nullable=False)

    # 语料级指标
    corpus_wer = Column(Float, nullable=True)
    corpus_cer = Column(Float, nullable=True)
    corpus_ser = Column(Float, nullable=True)
    corpus_mer = Column(Float, nullable=True)
    corpus_wil = Column(Float, nullable=True)
    corpus_wip = Column(Float, nullable=True)

    # 编辑操作统计
    total_sub = Column(Integer, nullable=True)
    total_ins = Column(Integer, nullable=True)
    total_del = Column(Integer, nullable=True)
    total_cor = Column(Integer, nullable=True)

    # wer分布统计
    wer_mean = Column(Float, nullable=True) # 平均值
    wer_median = Column(Float, nullable=True) # 中位数
    wer_std = Column(Float, nullable=True) # 标准差

    # 任务状态
    status = Column(String(20), nullable=False, default="pending", index=True)
    error_message = Column(Text, nullable=True)
    report_path = Column(String(255), nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # 关联: 一次评测 ->多条逐句明细
    # cascade="all, delete-orphan" 表示删除评测时自动删除关联的明细
    details = relationship(
        "EvaluationDetail",
        back_populates="evaluation",
        cascade="all, delete-orphan"
    )
    def __repr__(self):
        return (
            f"<Evaluation(id={self.id}, model='{self.model_name}', dataset='{self.dataset_name}', "
            f"dataset='{self.dataset_name}, wer={self.corpus_wer:.2%}, "
            f"status='{self.status}')>"
        )
    
class EvaluationDetail(Base):
    """
    evaluation_details评测明细表 - 逐句评测结果

    每次评测中的每条句子, 都会存一行记录
    用于: 查看哪些句子识别得好/差, 按长度分析, 导出明细csv.
    字段说明:
        - id: 自增主键
        - evaluation_id: 外键,关联到evaluations评测记录表的id
        - sentence_idx: 句子在数据集中的序号(从1开始)
        
        - reference: 句子的参考文本(ground truth)
        - hypothesis: 预测文本(模型输出)

        - ref_syllables: 参考文本音节数
        - hyp_syllables: 预测文本音节数
        - ref_chars: 参考文本字符数
        - hyp_chars: 预测文本字符数

        - wer/cer/mer/wil/wip: 该句子的评测指标

        - word_sub/ins/del/cor: 该句子的编辑操作计数
        - is_correct: 该句子是否完全正确识别(wer=0)
    """
    __tablename__ = "evaluation_details"
    # 基本信息
    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id",ondelete="CASCADE"), nullable=False, index=True)
    sentence_idx = Column(Integer, nullable=False)

    # 文本
    reference = Column(Text, nullable=False)
    hypothesis = Column(Text, nullable=False)

    # 长度
    ref_syllables = Column(Integer, nullable=False)
    hyp_syllables = Column(Integer, nullable=False)
    ref_chars = Column(Integer, nullable=False)
    hyp_chars = Column(Integer, nullable=False)

    # 指标
    wer = Column(Float, nullable=False)
    cer = Column(Float, nullable=False)
    mer = Column(Float, nullable=False)
    wil = Column(Float, nullable=False)
    wip = Column(Float, nullable=False)

    # 编辑操作
    word_sub = Column(Integer, nullable=False)
    word_ins = Column(Integer, nullable=False)
    word_del = Column(Integer, nullable=False)
    word_cor = Column(Integer, nullable=False)

    # 是否完成正确识别
    is_correct = Column(Boolean, nullable=False)
    # 反向关联
    evaluation = relationship("Evaluation", back_populates="details")


# ──────────────────────────────────────
# 3. 工具函数
# ──────────────────────────────────────

def init_db():
    """
    初始化数据库: 创建所有表(如果不存在)
    在FastAPI启动时 一次调用
    """
    Base.metadata.create_all(bind=engine)

def get_db():
    """
    获取数据库会话的依赖注入函数
    
    在FastAPI中这样使用
        @app.get("xxx")
        def some_api(db: Session = Depends(get_db)):
            ...
    yield 语法确保
        - 请求开始时创建 session
        - 请求结束时自动关闭 session, 避免连接泄漏
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()