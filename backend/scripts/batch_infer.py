"""
batch_infer.py — 批量推理脚本
================================

用途：
  加载合并后的 Whisper 模型，对测试集 manifest（JSONL）进行批量推理，
  输出 CSV 文件（transcription, predicted_string），供平台的 eval_engine 消费。

调用方式：
  python batch_infer.py \
    --model_path  output/mongolian/merged/whisper-small-finetune \
    --test_data   .../manifests/mn/test.jsonl \
    --out_csv     data/eval_outputs/eval_42.csv \
    --language    Mongolian \
    --batch_size  8

manifest 每行格式（prepare_cv_split 产出）：
  {"audio": {"path": "/abs/path.wav"}, "sentence": "参考文本", "duration": 3.2}
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor


# ──────────────────────────────────────
# CLI
# ──────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch inference with a merged Whisper model")
    p.add_argument("--model_path", type=str, required=True,
                   help="合并模型目录路径")
    p.add_argument("--test_data", type=str, required=True,
                   help="JSONL manifest 路径（每行含 audio.path + sentence）")
    p.add_argument("--out_csv", type=str, required=True,
                   help="输出 CSV 路径（transcription, predicted_string）")
    p.add_argument("--language", type=str, default="Chinese",
                   help="Whisper 语言参数")
    p.add_argument("--task", type=str, default="transcribe",
                   choices=["transcribe", "translate"],
                   help="Whisper 任务类型")
    p.add_argument("--batch_size", type=int, default=8,
                   help="推理 batch size")
    p.add_argument("--num_workers", type=int, default=4,
                   help="DataLoader workers")
    p.add_argument("--local_files_only", type=str, default="True",
                   help="是否只从本地加载模型")
    p.add_argument("--sample_rate", type=int, default=16000,
                   help="目标采样率")
    p.add_argument("--max_duration", type=float, default=30.0,
                   help="最长音频秒数")
    return p.parse_args()


# ──────────────────────────────────────
# 数据加载
# ──────────────────────────────────────

def load_manifest(path: str, max_duration: float = 30.0) -> List[dict]:
    """读取 JSONL manifest，返回 list[dict]。"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # 跳过超长音频
            dur = item.get("duration", 0)
            if dur > max_duration:
                continue
            items.append(item)
    return items


def load_audio(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """加载单个音频文件，返回 float32 numpy 数组（单通道）。"""
    sample, sr = soundfile.read(audio_path, dtype="float32")
    sample = sample.T
    # 转单通道
    if sample.ndim > 1:
        sample = librosa.to_mono(sample)
    # 重采样
    if sr != sample_rate:
        sample = librosa.resample(sample, orig_sr=sr, target_sr=sample_rate)
    return sample


# ──────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────

def main():
    args = parse_args()
    local_files_only = args.local_files_only.lower() in ("true", "1", "yes")

    print(f"[batch_infer] model_path      = {args.model_path}")
    print(f"[batch_infer] test_data       = {args.test_data}")
    print(f"[batch_infer] out_csv         = {args.out_csv}")
    print(f"[batch_infer] language        = {args.language}")
    print(f"[batch_infer] batch_size      = {args.batch_size}")
    print(f"[batch_infer] local_files_only= {local_files_only}")
    sys.stdout.flush()

    # 检查路径
    assert os.path.isdir(args.model_path), f"模型目录不存在: {args.model_path}"
    assert os.path.isfile(args.test_data), f"manifest 不存在: {args.test_data}"

    # ── 1. 加载模型 ──
    print("[batch_infer] Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        args.model_path,
        language=args.language,
        task=args.task,
        no_timestamps=True,
        local_files_only=local_files_only,
    )

    print("[batch_infer] Loading model...")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    ).to(device)
    model.generation_config.language = args.language.lower()
    model.generation_config.forced_decoder_ids = None
    model.eval()
    print(f"[batch_infer] Model loaded on {device}, dtype={torch_dtype}")
    sys.stdout.flush()

    # ── 2. 加载 manifest ──
    items = load_manifest(args.test_data, max_duration=args.max_duration)
    print(f"[batch_infer] Loaded {len(items)} items from manifest")
    sys.stdout.flush()

    # ── 3. 批量推理 ──
    results: List[Tuple[str, str]] = []  # (reference, hypothesis)
    skipped = 0

    for batch_start in tqdm(range(0, len(items), args.batch_size), desc="Inference"):
        batch_items = items[batch_start: batch_start + args.batch_size]

        # 加载音频 + 提取特征
        features_list = []
        refs = []
        for item in batch_items:
            audio_path = item["audio"]["path"]
            ref_text = item.get("sentence", "")
            try:
                audio = load_audio(audio_path, sample_rate=args.sample_rate)
                feat = processor(
                    audio=audio,
                    sampling_rate=args.sample_rate,
                    return_tensors="pt",
                ).input_features.squeeze(0)
                features_list.append(feat)
                refs.append(ref_text)
            except Exception as e:
                print(f"[batch_infer] WARN: skip {audio_path}: {e}", file=sys.stderr)
                skipped += 1
                continue

        if not features_list:
            continue

        # 组 batch tensor
        input_features = torch.stack(features_list).to(device, dtype=torch_dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features,
                max_new_tokens=255,
            )

        decoded = processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True,
        )

        for ref, hyp in zip(refs, decoded):
            results.append((ref.strip(), hyp.strip()))

        # 释放显存
        del input_features, generated_ids
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── 4. 写 CSV ──
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["transcription", "predicted_string"])
        for ref, hyp in results:
            writer.writerow([ref, hyp])

    print(f"[batch_infer] Done! {len(results)} pairs written to {out_path}")
    if skipped:
        print(f"[batch_infer] Skipped {skipped} items due to errors")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
