"""
eval_engine.py — 多语言 ASR 评估引擎
======================================

职责：
  1. 自动检测文本语言/脚本（Unicode 范围判断）
  2. 根据语言选择分词策略（空格 / tsheg / 单字符）
  3. 逐句计算 WER / CER / MER / WIL / WIP
  4. 汇总语料级指标
  5. 生成 PDF 报告

分词策略：
  tokenize_mode="auto"（默认）：
    - 藏文 (U+0F00-0FFF)         → 按 tsheg ་ 分音节
    - 蒙古文 (U+1800-18AF)       → 按空格分词
    - 中文/日文汉字 (CJK)         → 按单字符分（业界标准）
    - 日文假名                    → 按单字符分
    - 泰文/高棉文/缅甸文          → 按单字符分（无空格语言）
    - 韩文                       → 按空格分词
    - 阿拉伯文/希伯来文           → 按空格分词
    - 天城文/孟加拉文等           → 按空格分词
    - 拉丁/西里尔等              → 按空格分词
  tokenize_mode="whisper"：
    - 用 tiktoken 的 multilingual tokenizer
    - 需要额外 pip install tiktoken
  tokenize_mode="char"：
    - 所有语言统一按字符分（最保守）
  tokenize_mode="space"：
    - 所有语言统一按空格分

接口：
  compute_all_metrics(pairs, tokenize_mode="auto")
  generate_report(sentence_metrics, corpus_metrics, output_path)
"""

import re
import unicodedata
from typing import List, Tuple, Dict, Any, Optional

import numpy as np


# ──────────────────────────────────────
# 多语言分词系统
# ──────────────────────────────────────

# Unicode 范围 → 脚本名称 → 分词策略
# 每个元组: (起始码, 结束码, 脚本名, 分词方式)
#   "tsheg"  = 按藏文音节分隔符 ་ 切
#   "char"   = 按单字符切（没有空格的语言）
#   "space"  = 按空格切
SCRIPT_RANGES = [
    # 藏文：按 tsheg 切
    (0x0F00, 0x0FFF, "tibetan",    "tsheg"),

    # CJK 汉字：按单字符切
    (0x4E00, 0x9FFF, "cjk",        "char"),   # 基本汉字
    (0x3400, 0x4DBF, "cjk",        "char"),   # 扩展A
    (0x20000, 0x2A6DF, "cjk",      "char"),   # 扩展B
    (0x2A700, 0x2B73F, "cjk",      "char"),   # 扩展C
    (0xF900, 0xFAFF, "cjk",        "char"),   # 兼容汉字

    # 日文假名：按单字符切
    (0x3040, 0x309F, "hiragana",   "char"),
    (0x30A0, 0x30FF, "katakana",   "char"),

    # 泰文/老挝文/缅甸文/高棉文：按单字符切（没有空格分词的语言）
    (0x0E00, 0x0E7F, "thai",       "char"),
    (0x0E80, 0x0EFF, "lao",        "char"),
    (0x1000, 0x109F, "myanmar",    "char"),
    (0x1780, 0x17FF, "khmer",      "char"),

    # 以下语言有空格，按空格切
    (0x1800, 0x18AF, "mongolian",  "space"),   # 蒙古文
    (0xAC00, 0xD7AF, "hangul",     "space"),   # 韩文音节
    (0x1100, 0x11FF, "hangul",     "space"),   # 韩文字母
    (0x0600, 0x06FF, "arabic",     "space"),   # 阿拉伯文
    (0x0590, 0x05FF, "hebrew",     "space"),   # 希伯来文
    (0x0900, 0x097F, "devanagari", "space"),   # 天城文（印地语等）
    (0x0980, 0x09FF, "bengali",    "space"),   # 孟加拉文
    (0x0A80, 0x0AFF, "gujarati",   "space"),   # 古吉拉特文
    (0x0B80, 0x0BFF, "tamil",      "space"),   # 泰米尔文
    (0x0C00, 0x0C7F, "telugu",     "space"),   # 泰卢固文
    (0x0C80, 0x0CFF, "kannada",    "space"),   # 卡纳达文
    (0x0D00, 0x0D7F, "malayalam",  "space"),   # 马拉雅拉姆文
    (0x0400, 0x04FF, "cyrillic",   "space"),   # 西里尔文（俄语等）
    (0x0370, 0x03FF, "greek",      "space"),   # 希腊文
    (0x10A0, 0x10FF, "georgian",   "space"),   # 格鲁吉亚文
    (0x0530, 0x058F, "armenian",   "space"),   # 亚美尼亚文
    (0x1200, 0x137F, "ethiopic",   "space"),   # 埃塞俄比亚文
]

# 拉丁文兜底：ASCII + Latin Extended 都算拉丁，按空格切
LATIN_RANGES = [(0x0041, 0x024F)]


def detect_script(text: str) -> Tuple[str, str]:
    """
    检测文本的主要脚本和对应的分词策略。

    原理：统计文本中每个字符属于哪个 Unicode 脚本区间，
    取出现次数最多的脚本作为该文本的语言。

    Returns:
        (script_name, tokenize_strategy)
        如 ("tibetan", "tsheg") 或 ("cjk", "char") 或 ("cyrillic", "space")
    """
    script_counts: Dict[str, int] = {}
    strategy_map: Dict[str, str] = {}

    for ch in text:
        cp = ord(ch)

        # 跳过空格、标点、数字
        if ch.isspace() or ch.isdigit():
            continue
        if unicodedata.category(ch).startswith('P'):
            continue

        # 在已知脚本范围里查找
        found = False
        for start, end, script, strategy in SCRIPT_RANGES:
            if start <= cp <= end:
                script_counts[script] = script_counts.get(script, 0) + 1
                strategy_map[script] = strategy
                found = True
                break

        # 拉丁文兜底
        if not found:
            for start, end in LATIN_RANGES:
                if start <= cp <= end:
                    script_counts["latin"] = script_counts.get("latin", 0) + 1
                    strategy_map["latin"] = "space"
                    found = True
                    break

        # 还没找到，归入 unknown
        if not found:
            script_counts["unknown"] = script_counts.get("unknown", 0) + 1
            strategy_map["unknown"] = "space"

    if not script_counts:
        return ("unknown", "space")

    # 取出现最多的脚本
    dominant = max(script_counts, key=script_counts.get)
    return (dominant, strategy_map[dominant])


# ──────────────────────────────────────
# 各分词策略的具体实现
# ──────────────────────────────────────

TSHEG = '\u0F0B'  # 藏文音节分隔符 ་


def _normalize(text: str) -> str:
    """通用文本规范化：去首尾空白，合并连续空格"""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def _tokenize_by_space(text: str) -> list:
    """
    按空格分词。
    适用于：英语、法语、德语、俄语、蒙古语、韩语、阿拉伯语、印地语等。
    去除标点后按空格切分。
    """
    text = _normalize(text)
    # 去除标点（保留字母、数字、空格）
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    tokens = text.split()
    return [t for t in tokens if t]


def _tokenize_by_tsheg(text: str) -> list:
    """
    按藏文 tsheg 分音节。
    适用于：藏语。
    tsheg (་ U+0F0B) 是藏文的音节分隔符。
    同时也按空格和 shad (། U+0F0D) 切分。
    """
    text = _normalize(text)
    text = text.rstrip(TSHEG).rstrip()
    if not text:
        return []
    tokens = re.split(r'[\u0F0B\s\u0F0D\u0F0E]+', text)
    return [t for t in tokens if t.strip()]


def _tokenize_by_char(text: str) -> list:
    """
    按单字符分词。
    适用于：中文、日文、泰文、缅甸文、高棉文等无空格语言。
    去除空格和标点后，每个字符算一个 token。
    对中文 ASR 来说，WER 按字计算就是业界标准的 CER。
    """
    text = _normalize(text)
    # 去除空格和标点
    chars = []
    for ch in text:
        if ch.isspace():
            continue
        if unicodedata.category(ch).startswith('P'):
            continue
        chars.append(ch)
    return chars


def _tokenize_by_whisper(text: str) -> list:
    """
    用 Whisper 的 multilingual tokenizer 分词。
    需要安装 tiktoken：pip install tiktoken
    这样分出来的 token 和模型内部一致。
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        token_ids = enc.encode(text)
        tokens = [enc.decode([tid]) for tid in token_ids]
        return [t for t in tokens if t.strip()]
    except ImportError:
        print("[WARN] tiktoken 未安装，回退到 auto 模式")
        return tokenize_for_wer(text, mode="auto")


def tokenize_for_wer(text: str, mode: str = "auto") -> list:
    """
    WER 计算用的分词入口。

    这是整个分词系统的统一入口。
    外部只需要调这一个函数，传入 mode 参数即可。

    Args:
        text: 要分词的文本
        mode: 分词模式
              "auto"    - 自动检测语言，选对应策略（默认）
              "whisper" - 用 Whisper tokenizer
              "char"    - 强制按字符分
              "space"   - 强制按空格分

    Returns:
        token 列表，如 ["བོད", "སྐད", "ཀྱི"] 或 ["你", "好", "世", "界"]
    """
    if mode == "whisper":
        return _tokenize_by_whisper(text)
    elif mode == "char":
        return _tokenize_by_char(text)
    elif mode == "space":
        return _tokenize_by_space(text)
    elif mode == "auto":
        _, strategy = detect_script(text)
        if strategy == "tsheg":
            return _tokenize_by_tsheg(text)
        elif strategy == "char":
            return _tokenize_by_char(text)
        else:
            return _tokenize_by_space(text)
    else:
        return _tokenize_by_space(text)


def tokenize_for_cer(text: str) -> list:
    """
    CER 计算用的字符分词。
    所有语言统一：去除空格和标点后按字符切。
    """
    text = _normalize(text)
    chars = []
    for ch in text:
        if ch.isspace():
            continue
        # 藏文分隔符也去掉
        if ch in ('\u0F0B', '\u0F0D', '\u0F0E', '\u0F14'):
            continue
        if unicodedata.category(ch).startswith('P'):
            continue
        chars.append(ch)
    return chars


# ──────────────────────────────────────
# 编辑距离
# ──────────────────────────────────────

def _levenshtein_ops(ref: list, hyp: list):
    """计算替换、插入、删除、正确的次数"""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[''] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
        bt[i][0] = 'D'
    for j in range(m + 1):
        dp[0][j] = j
        bt[0][j] = 'I'
    bt[0][0] = ''

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost_sub = dp[i - 1][j - 1]
            else:
                cost_sub = dp[i - 1][j - 1] + 1
            cost_del = dp[i - 1][j] + 1
            cost_ins = dp[i][j - 1] + 1
            min_cost = min(cost_sub, cost_del, cost_ins)
            dp[i][j] = min_cost
            if min_cost == cost_sub:
                bt[i][j] = 'C' if ref[i - 1] == hyp[j - 1] else 'S'
            elif min_cost == cost_del:
                bt[i][j] = 'D'
            else:
                bt[i][j] = 'I'

    i, j = n, m
    S = I = D = C = 0
    while i > 0 or j > 0:
        op = bt[i][j]
        if op == 'C':
            C += 1; i -= 1; j -= 1
        elif op == 'S':
            S += 1; i -= 1; j -= 1
        elif op == 'D':
            D += 1; i -= 1
        elif op == 'I':
            I += 1; j -= 1
        else:
            break
    return S, I, D, C


# ──────────────────────────────────────
# 核心接口
# ──────────────────────────────────────

def compute_all_metrics(
    pairs: List[Tuple[str, str]],
    tokenize_mode: str = "auto",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    计算所有指标。
    
    Args:
        pairs: [(reference, hypothesis), ...] 列表
        tokenize_mode: 分词模式 "auto"/"whisper"/"char"/"space"
    
    Returns:
        (sentence_metrics_list, corpus_metrics_dict)
    """
    sentence_metrics = []

    # 检测第一条数据的语言（用于日志）
    if pairs:
        script, strategy = detect_script(pairs[0][0])
        print(f"[INFO] 检测到脚本: {script}, 分词策略: "
              f"{tokenize_mode if tokenize_mode != 'auto' else strategy}")

    for idx, (ref_text, hyp_text) in enumerate(pairs):
        ref_words = tokenize_for_wer(ref_text, mode=tokenize_mode)
        hyp_words = tokenize_for_wer(hyp_text, mode=tokenize_mode)
        ref_chars = tokenize_for_cer(ref_text)
        hyp_chars = tokenize_for_cer(hyp_text)

        # 音节级编辑操作
        w_sub, w_ins, w_del, w_cor = _levenshtein_ops(ref_words, hyp_words)
        # 字符级编辑操作
        c_sub, c_ins, c_del, c_cor = _levenshtein_ops(ref_chars, hyp_chars)

        # WER
        n_ref = len(ref_words)
        n_hyp = len(hyp_words)
        wer = (w_sub + w_ins + w_del) / n_ref if n_ref > 0 else (0.0 if n_hyp == 0 else 1.0)

        # CER
        n_ref_c = len(ref_chars)
        n_hyp_c = len(hyp_chars)
        cer = (c_sub + c_ins + c_del) / n_ref_c if n_ref_c > 0 else (0.0 if n_hyp_c == 0 else 1.0)

        # MER
        total_ops = w_sub + w_del + w_ins + w_cor
        mer = (w_sub + w_del + w_ins) / total_ops if total_ops > 0 else 0.0

        # WIL / WIP
        if n_ref > 0 and n_hyp > 0:
            wip = (w_cor / n_ref) * (w_cor / n_hyp)
            wil = 1.0 - wip
        elif n_ref == 0 and n_hyp == 0:
            wip, wil = 1.0, 0.0
        else:
            wip, wil = 0.0, 1.0

        is_correct = (_normalize(ref_text) == _normalize(hyp_text))

        sentence_metrics.append({
            "idx": idx,
            "reference": ref_text,
            "hypothesis": hyp_text,
            "ref_words": n_ref,
            "hyp_words": n_hyp,
            "ref_chars": n_ref_c,
            "hyp_chars": n_hyp_c,
            "wer": wer,
            "cer": cer,
            "mer": mer,
            "wil": wil,
            "wip": wip,
            "word_sub": w_sub,
            "word_ins": w_ins,
            "word_del": w_del,
            "word_cor": w_cor,
            "char_sub": c_sub,
            "char_ins": c_ins,
            "char_del": c_del,
            "char_cor": c_cor,
            "is_correct": is_correct,
        })

    # ── 汇总语料级指标 ──
    n = len(sentence_metrics)
    if n == 0:
        return [], _empty_corpus()

    total_ref_w = sum(m["ref_words"] for m in sentence_metrics)
    total_hyp_w = sum(m["hyp_words"] for m in sentence_metrics)
    total_w_sub = sum(m["word_sub"] for m in sentence_metrics)
    total_w_ins = sum(m["word_ins"] for m in sentence_metrics)
    total_w_del = sum(m["word_del"] for m in sentence_metrics)
    total_w_cor = sum(m["word_cor"] for m in sentence_metrics)

    total_ref_c = sum(m["ref_chars"] for m in sentence_metrics)
    total_c_sub = sum(m["char_sub"] for m in sentence_metrics)
    total_c_ins = sum(m["char_ins"] for m in sentence_metrics)
    total_c_del = sum(m["char_del"] for m in sentence_metrics)

    correct_count = sum(1 for m in sentence_metrics if m["is_correct"])

    corpus_wer = (total_w_sub + total_w_ins + total_w_del) / total_ref_w if total_ref_w > 0 else 0
    corpus_cer = (total_c_sub + total_c_ins + total_c_del) / total_ref_c if total_ref_c > 0 else 0
    corpus_ser = 1.0 - (correct_count / n)

    total_ops = total_w_sub + total_w_del + total_w_ins + total_w_cor
    corpus_mer = (total_w_sub + total_w_del + total_w_ins) / total_ops if total_ops > 0 else 0

    if total_ref_w > 0 and total_hyp_w > 0:
        corpus_wip = (total_w_cor / total_ref_w) * (total_w_cor / total_hyp_w)
        corpus_wil = 1.0 - corpus_wip
    else:
        corpus_wip, corpus_wil = 0.0, 1.0

    wers = np.array([m["wer"] for m in sentence_metrics])

    corpus_metrics = {
        "corpus_wer": float(corpus_wer),
        "corpus_cer": float(corpus_cer),
        "corpus_ser": float(corpus_ser),
        "corpus_mer": float(corpus_mer),
        "corpus_wil": float(corpus_wil),
        "corpus_wip": float(corpus_wip),
        "total_word_sub": total_w_sub,
        "total_word_ins": total_w_ins,
        "total_word_del": total_w_del,
        "total_word_cor": total_w_cor,
        "wer_mean": float(np.mean(wers)),
        "wer_median": float(np.median(wers)),
        "wer_std": float(np.std(wers)),
        "num_sentences": n,
        "num_correct": correct_count,
    }

    return sentence_metrics, corpus_metrics


def _empty_corpus():
    """空语料的默认指标"""
    return {
        "corpus_wer": 0, "corpus_cer": 0, "corpus_ser": 0,
        "corpus_mer": 0, "corpus_wil": 0, "corpus_wip": 0,
        "total_word_sub": 0, "total_word_ins": 0,
        "total_word_del": 0, "total_word_cor": 0,
        "wer_mean": 0, "wer_median": 0, "wer_std": 0,
        "num_sentences": 0, "num_correct": 0,
    }


def generate_report(
    sentence_metrics: List[Dict],
    corpus_metrics: Dict,
    output_path: str,
):
    """
    生成 PDF 报告。
    
    Step 2 会替换为完整的中文可视化报告。
    当前版本生成一个简单的文本 PDF 占位。
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5,
                f"ASR Evaluation Report\n"
                f"WER: {corpus_metrics['corpus_wer']:.4f}\n"
                f"CER: {corpus_metrics['corpus_cer']:.4f}\n"
                f"Sentences: {corpus_metrics['num_sentences']}",
                ha='center', va='center', fontsize=16,
                transform=ax.transAxes)
        ax.axis('off')
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] 报告生成失败: {e}")
