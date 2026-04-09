"""
eval_engine.py — 评估引擎（Step 1 占位版，Step 2 替换为完整实现）
==================================================================

职责：
  1. 接收 (reference, hypothesis) 对列表
  2. 逐句计算 WER / CER / MER / WIL / WIP 等指标
  3. 汇总语料级指标
  4. 生成 PDF 报告

本文件在 Step 2 中会被替换为完整实现，
复用之前写的 tibetan_asr_eval.py 和 tibetan_asr_visualize.py。

当前版本提供：
  - compute_all_metrics(pairs) → (sentence_metrics_list, corpus_metrics_dict)
  - generate_report(sentence_metrics, corpus_metrics, output_path) → None
"""

import re
from typing import List, Tuple, Dict, Any

import numpy as np


# ──────────────────────────────────────
# 藏文分词（简化版，Step 2 用完整版替换）
# ──────────────────────────────────────

TSHEG = '\u0F0B'  # 藏文音节分隔符 ་


def _normalize(text: str) -> str:
    """基本文本规范化"""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.rstrip(TSHEG).rstrip()
    return text


def _tokenize_syllables(text: str) -> list:
    """按 tsheg 分音节"""
    text = _normalize(text)
    if not text:
        return []
    tokens = re.split(r'[\u0F0B\s\u0F0D\u0F0E]+', text)
    return [t for t in tokens if t.strip()]


def _tokenize_chars(text: str) -> list:
    """按字符分"""
    text = _normalize(text)
    text = re.sub(r'[\u0F0B\s\u0F0D\u0F0E]', '', text)
    return list(text)


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
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    计算所有指标。
    
    Args:
        pairs: [(reference, hypothesis), ...] 列表
    
    Returns:
        (sentence_metrics_list, corpus_metrics_dict)
        
        sentence_metrics_list: 每条句子的指标字典列表
        corpus_metrics_dict: 语料级汇总指标
    """
    sentence_metrics = []

    for idx, (ref_text, hyp_text) in enumerate(pairs):
        ref_words = _tokenize_syllables(ref_text)
        hyp_words = _tokenize_syllables(hyp_text)
        ref_chars = _tokenize_chars(ref_text)
        hyp_chars = _tokenize_chars(hyp_text)

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
