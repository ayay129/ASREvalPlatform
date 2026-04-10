"""
report_generator.py — ASR 评测报告生成
========================================

职责：
  1. 接收 sentence_metrics + corpus_metrics（来自 eval_engine）
  2. 生成带中文注释的多页 PDF 图表报告
  3. 导出逐句明细 CSV

与之前独立脚本的区别：
  - 不再从文件读数据，直接接收 Python 对象
  - 标题从"藏语"改为通用（显示实际语言名）
  - 可被 eval_engine.generate_report() 直接调用

依赖：matplotlib, numpy
"""

import csv
import os
from typing import List, Dict, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np


# ──────────────────────────────────────
# 配色
# ──────────────────────────────────────

COLORS = {
    'primary':    '#2563EB',
    'secondary':  '#7C3AED',
    'accent':     '#059669',
    'warning':    '#D97706',
    'danger':     '#DC2626',
    'muted':      '#6B7280',
    'bg_light':   '#F8FAFC',
    'bg_card':    '#FFFFFF',
    'text_dark':  '#1E293B',
    'text_light': '#94A3B8',
    'sub_color':  '#EF4444',
    'ins_color':  '#F59E0B',
    'del_color':  '#8B5CF6',
    'cor_color':  '#10B981',
    'note_bg':    '#F1F5F9',
    'note_border':'#CBD5E1',
}

PALETTE_5 = ['#2563EB', '#7C3AED', '#059669', '#D97706', '#DC2626']


def _setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans'],
        'axes.unicode_minus': False,
        'font.size': 10,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': COLORS['bg_light'],
        'axes.facecolor': COLORS['bg_card'],
        'axes.edgecolor': '#E2E8F0',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': '#CBD5E1',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })


# ──────────────────────────────────────
# 图表绘制函数
# ──────────────────────────────────────

def _plot_summary_card(ax, corpus, title="ASR Evaluation Report"):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    ax.text(5, 3.2, title, ha='center', va='center',
            fontsize=22, fontweight='bold', color=COLORS['text_dark'])

    n = corpus['num_sentences']
    acc = corpus['num_correct'] / n if n > 0 else 0

    cards = [
        ('WER',       f"{corpus['corpus_wer']*100:.2f}%", COLORS['primary']),
        ('CER',       f"{corpus['corpus_cer']*100:.2f}%", COLORS['secondary']),
        ('SER',       f"{corpus['corpus_ser']*100:.2f}%", COLORS['warning']),
        ('Sentences', str(n),                              COLORS['accent']),
        ('Accuracy',  f"{acc*100:.1f}%",                   COLORS['danger']),
    ]

    card_w, gap = 1.6, 0.15
    total_w = len(cards) * card_w + (len(cards) - 1) * gap
    x_start = (10 - total_w) / 2

    for i, (label, value, color) in enumerate(cards):
        x = x_start + i * (card_w + gap)
        rect = FancyBboxPatch((x, 0.5), card_w, 1.8,
                               boxstyle="round,pad=0.1",
                               facecolor=color, alpha=0.1,
                               edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + card_w/2, 1.85, value, ha='center', va='center',
                fontsize=16, fontweight='bold', color=color)
        ax.text(x + card_w/2, 0.95, label, ha='center', va='center',
                fontsize=9, color=COLORS['muted'])


def _plot_corpus_bar(ax, corpus):
    names  = ['WER', 'CER', 'SER', 'MER', 'WIL']
    descs  = ['Word Err', 'Char Err', 'Sent Err', 'Match Err', 'Word Info Lost']
    labels = [f'{d}\n({n})' for d, n in zip(descs, names)]
    keys   = ['corpus_wer', 'corpus_cer', 'corpus_ser', 'corpus_mer', 'corpus_wil']
    values = [corpus[k] * 100 for k in keys]

    bars = ax.barh(labels[::-1], values[::-1], color=PALETTE_5[::-1],
                   height=0.55, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', ha='left',
                fontsize=10, fontweight='bold', color=COLORS['text_dark'])

    ax.set_xlim(0, max(values) * 1.25 if max(values) > 0 else 10)
    ax.set_title('Corpus-level Error Rates', pad=12)
    ax.set_xlabel('Error Rate (%)')
    ax.invert_yaxis()


def _plot_edit_ops(ax, corpus):
    labels = ['Correct', 'Substitution', 'Insertion', 'Deletion']
    sizes = [corpus['total_word_cor'], corpus['total_word_sub'],
             corpus['total_word_ins'], corpus['total_word_del']]
    colors = [COLORS['cor_color'], COLORS['sub_color'],
              COLORS['ins_color'], COLORS['del_color']]
    total = sum(sizes)
    if total == 0:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color=COLORS['muted'])
        return

    wedges, _, _ = ax.pie(
        sizes, labels=None, colors=colors, autopct='',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
    ax.text(0, 0.05, f'{total:,}', ha='center', va='center',
            fontsize=16, fontweight='bold', color=COLORS['text_dark'])
    ax.text(0, -0.12, 'Total ops', ha='center', va='center',
            fontsize=8, color=COLORS['text_light'])
    legend_labels = [f'{l}: {s:,} ({s/total*100:.1f}%)' for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc='lower center',
              bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)
    ax.set_title('Edit Operation Distribution (word-level)', pad=12)


def _plot_histogram(ax, values_pct, title, xlabel, mean_val, median_val):
    bins = np.arange(0, min(max(values_pct) + 10, 210), 5)
    _, bins_out, patches = ax.hist(values_pct, bins=bins, edgecolor='white',
                                   linewidth=0.8, alpha=0.85)
    for patch, left in zip(patches, bins_out[:-1]):
        if left == 0:
            patch.set_facecolor(COLORS['accent'])
        elif left <= 10:
            patch.set_facecolor('#60A5FA')
        elif left <= 30:
            patch.set_facecolor(COLORS['warning'])
        else:
            patch.set_facecolor(COLORS['danger'])

    ax.axvline(mean_val, color=COLORS['danger'], linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_val:.1f}%')
    ax.axvline(median_val, color=COLORS['secondary'], linestyle='-.', linewidth=1.5,
               label=f'Median: {median_val:.1f}%')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Sentence Count')


def _plot_wer_bucket(ax, wers):
    buckets = [
        ('Perfect\n(WER=0)',  np.sum(wers == 0)),
        ('Low\n(0~10%)',      np.sum((wers > 0) & (wers <= 0.1))),
        ('Medium\n(10~30%)',  np.sum((wers > 0.1) & (wers <= 0.3))),
        ('High\n(30~50%)',    np.sum((wers > 0.3) & (wers <= 0.5))),
        ('Severe\n(>50%)',    np.sum(wers > 0.5)),
    ]
    labels = [b[0] for b in buckets]
    counts = [b[1] for b in buckets]
    colors = [COLORS['accent'], '#60A5FA', COLORS['warning'], '#F97316', COLORS['danger']]

    bars = ax.bar(labels, counts, color=colors, edgecolor='white',
                  linewidth=1.5, width=0.65)
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            pct = cnt / len(wers) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color=COLORS['text_dark'])
    ax.set_title('WER Bucket Distribution', pad=12)
    ax.set_ylabel('Sentence Count')
    ax.set_ylim(0, max(counts) * 1.3 if max(counts) > 0 else 10)


def _plot_wer_vs_length(ax, ref_lens, wer_vals):
    ax.scatter(ref_lens, wer_vals, alpha=0.4, s=20, c=COLORS['primary'], edgecolors='none')
    if len(ref_lens) > 5:
        z = np.polyfit(ref_lens, wer_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(ref_lens.min(), ref_lens.max(), 100)
        ax.plot(x_line, p(x_line), '--', color=COLORS['danger'], linewidth=2,
                label=f'Trend (slope={z[0]:.2f})')
        ax.legend(fontsize=9, framealpha=0.9)
    ax.set_title('WER vs. Reference Length', pad=12)
    ax.set_xlabel('Reference Length (words)')
    ax.set_ylabel('WER (%)')
    ax.set_ylim(bottom=-2)


def _plot_boxplot(ax, wer_pct, cer_pct):
    bp = ax.boxplot([wer_pct, cer_pct], tick_labels=['WER (%)', 'CER (%)'],
                    patch_artist=True, widths=0.45,
                    medianprops=dict(color=COLORS['danger'], linewidth=2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.4))
    bp['boxes'][0].set_facecolor('#DBEAFE')
    bp['boxes'][0].set_edgecolor(COLORS['primary'])
    bp['boxes'][1].set_facecolor('#EDE9FE')
    bp['boxes'][1].set_edgecolor(COLORS['secondary'])
    ax.set_title('WER & CER Box Plot', pad=12)
    ax.set_ylabel('Error Rate (%)')


def _plot_cdf(ax, wer_pct):
    wer_sorted = np.sort(wer_pct)
    cdf = np.arange(1, len(wer_sorted) + 1) / len(wer_sorted) * 100
    ax.fill_between(wer_sorted, cdf, alpha=0.15, color=COLORS['primary'])
    ax.plot(wer_sorted, cdf, color=COLORS['primary'], linewidth=2)
    for threshold, color in [(10, COLORS['accent']), (20, COLORS['warning']), (50, COLORS['danger'])]:
        pct_below = np.sum(wer_pct <= threshold) / len(wer_pct) * 100
        ax.axvline(threshold, color=color, linestyle=':', linewidth=1, alpha=0.7)
        ax.text(threshold + 1, pct_below - 3,
                f'{pct_below:.0f}% <= {threshold}%',
                fontsize=8, color=color, fontweight='bold')
    ax.set_title('WER Cumulative Distribution (CDF)', pad=12)
    ax.set_xlabel('WER (%)')
    ax.set_ylabel('Cumulative Sentences (%)')
    ax.set_xlim(left=-2)
    ax.set_ylim(0, 105)


# ──────────────────────────────────────
# 动态中文注释
# ──────────────────────────────────────

def _build_annotations(corpus, wers, cers, ref_lens):
    n = corpus['num_sentences']
    sub = corpus['total_word_sub']
    ins = corpus['total_word_ins']
    dele = corpus['total_word_del']
    cor = corpus['total_word_cor']
    total_ops = sub + ins + dele + cor

    if total_ops > 0:
        sub_pct = sub / total_ops * 100
        ins_pct = ins / total_ops * 100
        del_pct = dele / total_ops * 100
    else:
        sub_pct = ins_pct = del_pct = 0

    err_types = {'Substitution': sub_pct, 'Insertion': ins_pct, 'Deletion': del_pct}
    dominant = max(err_types, key=err_types.get)
    perfect_pct = np.sum(wers == 0) / len(wers) * 100
    severe_pct = np.sum(wers > 0.5) / len(wers) * 100

    slope = 0
    if len(ref_lens) > 5:
        z = np.polyfit(ref_lens, wers * 100, 1)
        slope = z[0]

    acc = corpus['num_correct'] / n * 100 if n > 0 else 0
    wer_mean = corpus['wer_mean'] * 100
    wer_med = corpus['wer_median'] * 100
    wer_std = corpus['wer_std'] * 100
    cer_mean = np.mean(cers)
    cer_med = np.median(cers)

    notes = {
        'summary': (
            f"This report evaluates {n} ASR results. "
            f"Corpus WER: {corpus['corpus_wer']*100:.2f}%, "
            f"CER: {corpus['corpus_cer']*100:.2f}%, "
            f"Sentence accuracy: {acc:.1f}%."
        ),
        'corpus_bar': (
            f"WER measures word-level accuracy; "
            f"CER measures character-level accuracy; "
            f"SER = {corpus['corpus_ser']*100:.1f}% of sentences have at least one error."
        ),
        'edit_ops': (
            f"Out of {total_ops:,} word-level operations: "
            f"Sub {sub_pct:.1f}%, Ins {ins_pct:.1f}%, Del {del_pct:.1f}%. "
            f"Dominant error type: {dominant}."
        ),
        'wer_hist': (
            f"Mean {wer_mean:.1f}%, Median {wer_med:.1f}%, Std {wer_std:.1f}%. "
            f"Green=perfect, Blue=low, Orange=medium, Red=high."
        ),
        'cer_hist': (
            f"CER mean {cer_mean:.1f}%, median {cer_med:.1f}%. "
            f"CER is typically lower than WER."
        ),
        'wer_bucket': (
            f"Perfect recognition: {perfect_pct:.1f}%, "
            f"Severe errors (>50%): {severe_pct:.1f}%."
        ),
        'wer_vs_length': (
            f"Trend slope: {slope:.2f}. "
            + (f"Positive slope: longer sentences have higher WER."
               if slope > 0.1 else
               f"Model performs consistently across sentence lengths."
               if slope > -0.1 else
               f"Longer sentences have lower WER.")
        ),
        'boxplot': (
            f"Box spans 25th-75th percentile, red line = median. "
            f"WER median: {wer_med:.1f}%, CER median: {cer_med:.1f}%."
        ),
        'cdf': (
            f"A curve reaching 100% earlier indicates better overall quality."
        ),
    }
    return notes


def _note_box(ax, text):
    ax.axis('off')
    ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=8.5,
            va='top', ha='left', color='#475569', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['note_bg'],
                      edgecolor=COLORS['note_border'], alpha=0.85, linewidth=0.8))


# ──────────────────────────────────────
# 公开接口
# ──────────────────────────────────────

def generate_pdf_report(
    sentence_metrics: List[Dict[str, Any]],
    corpus_metrics: Dict[str, Any],
    output_path: str,
    title: str = "ASR Evaluation Report",
):
    """
    生成带中文注释的 PDF 图表报告。

    Args:
        sentence_metrics: eval_engine.compute_all_metrics 返回的逐句指标
        corpus_metrics: eval_engine.compute_all_metrics 返回的语料级指标
        output_path: 输出 PDF 路径
        title: 报告标题，如 "蒙古语 ASR 评测报告"
    """
    _setup_style()

    # 提取数组
    wers = np.array([m['wer'] for m in sentence_metrics])
    cers = np.array([m['cer'] for m in sentence_metrics])
    wer_pct = wers * 100
    cer_pct = cers * 100
    ref_lens = np.array([m['ref_words'] for m in sentence_metrics])

    notes = _build_annotations(corpus_metrics, wers, cer_pct, ref_lens)

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(output_path) as pdf:

        # ════════ PAGE 1 ════════
        fig1 = plt.figure(figsize=(16, 22))
        gs1 = gridspec.GridSpec(
            6, 2, figure=fig1, hspace=0.12, wspace=0.30,
            left=0.07, right=0.95, top=0.96, bottom=0.02,
            height_ratios=[1.2, 0.35, 1.5, 0.35, 1.5, 0.35])

        ax_card = fig1.add_subplot(gs1[0, :])
        _plot_summary_card(ax_card, corpus_metrics, title=title)

        ax_n0 = fig1.add_subplot(gs1[1, :])
        ax_n0.axis('off')
        ax_n0.text(0.5, 0.8, notes['summary'], transform=ax_n0.transAxes,
                   fontsize=10.5, va='top', ha='center', color='#334155',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#EFF6FF',
                             edgecolor='#93C5FD', alpha=0.9, linewidth=1))

        ax1 = fig1.add_subplot(gs1[2, 0])
        _plot_corpus_bar(ax1, corpus_metrics)
        ax2 = fig1.add_subplot(gs1[2, 1])
        _plot_edit_ops(ax2, corpus_metrics)

        _note_box(fig1.add_subplot(gs1[3, 0]), notes['corpus_bar'])
        _note_box(fig1.add_subplot(gs1[3, 1]), notes['edit_ops'])

        ax3 = fig1.add_subplot(gs1[4, 0])
        _plot_histogram(ax3, wer_pct, 'WER Distribution (per sentence)', 'WER (%)',
                        wer_pct.mean(), np.median(wer_pct))
        ax4 = fig1.add_subplot(gs1[4, 1])
        _plot_histogram(ax4, cer_pct, 'CER Distribution (per sentence)', 'CER (%)',
                        cer_pct.mean(), np.median(cer_pct))

        _note_box(fig1.add_subplot(gs1[5, 0]), notes['wer_hist'])
        _note_box(fig1.add_subplot(gs1[5, 1]), notes['cer_hist'])

        pdf.savefig(fig1, dpi=150, facecolor=fig1.get_facecolor())
        plt.close(fig1)

        # ════════ PAGE 2 ════════
        fig2 = plt.figure(figsize=(16, 18))
        gs2 = gridspec.GridSpec(
            4, 2, figure=fig2, hspace=0.12, wspace=0.30,
            left=0.07, right=0.95, top=0.96, bottom=0.03,
            height_ratios=[1.5, 0.35, 1.5, 0.35])

        ax5 = fig2.add_subplot(gs2[0, 0])
        _plot_wer_bucket(ax5, wers)
        ax6 = fig2.add_subplot(gs2[0, 1])
        _plot_wer_vs_length(ax6, ref_lens, wer_pct)

        _note_box(fig2.add_subplot(gs2[1, 0]), notes['wer_bucket'])
        _note_box(fig2.add_subplot(gs2[1, 1]), notes['wer_vs_length'])

        ax7 = fig2.add_subplot(gs2[2, 0])
        _plot_boxplot(ax7, wer_pct, cer_pct)
        ax8 = fig2.add_subplot(gs2[2, 1])
        _plot_cdf(ax8, wer_pct)

        _note_box(fig2.add_subplot(gs2[3, 0]), notes['boxplot'])
        _note_box(fig2.add_subplot(gs2[3, 1]), notes['cdf'])

        pdf.savefig(fig2, dpi=150, facecolor=fig2.get_facecolor())
        plt.close(fig2)

    print(f'[INFO] PDF 报告已生成: {output_path}')


def export_detail_csv(
    sentence_metrics: List[Dict[str, Any]],
    output_path: str,
):
    """
    导出逐句评测明细 CSV。

    可以用这个文件做进一步分析，
    或者导入 Excel 筛选/排序。
    """
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'idx', 'reference', 'hypothesis',
            'ref_words', 'hyp_words', 'ref_chars', 'hyp_chars',
            'wer', 'cer', 'mer', 'wil', 'wip',
            'word_sub', 'word_ins', 'word_del', 'word_cor',
            'is_correct',
        ])
        for m in sentence_metrics:
            writer.writerow([
                m['idx'] + 1, m['reference'], m['hypothesis'],
                m['ref_words'], m['hyp_words'],
                m.get('ref_chars', 0), m.get('hyp_chars', 0),
                f"{m['wer']:.6f}", f"{m['cer']:.6f}",
                f"{m['mer']:.6f}", f"{m['wil']:.6f}", f"{m['wip']:.6f}",
                m['word_sub'], m['word_ins'], m['word_del'], m['word_cor'],
                int(m['is_correct']),
            ])

    print(f'[INFO] 逐句明细 CSV 已导出: {output_path}')
