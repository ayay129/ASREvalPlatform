"""
prepare_manifest.py — 统一数据集转换入口
========================================

职责：
  1. 从不同来源读取语音数据集
  2. 转换成 Whisper-Finetune 可训练的 JSONL manifest
  3. 将音频物化到本地目录，统一成稳定可复用的绝对路径

当前只实现：
  - Hugging Face datasets

后续扩展：
  - CSV / TSV
  - 本地目录
  - 已经是 whisper-jsonl 的数据
"""

from __future__ import annotations

import abc
import argparse
import io
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import soundfile as sf
from datasets import Audio, Dataset, load_dataset, load_from_disk


class ManifestPreparationError(RuntimeError):
    """数据准备阶段的显式错误。"""


@dataclass
class SplitSummary:
    """单个 split 的转换统计。"""
    split_name: str
    output_file: str
    audio_dir: str
    total_rows: int = 0
    kept_rows: int = 0
    skipped_rows: int = 0
    skipped_reasons: Counter[str] = field(default_factory=Counter)

    def skip(self, reason: str) -> None:
        self.skipped_rows += 1
        self.skipped_reasons[reason] += 1

    def to_dict(self) -> dict:
        return {
            "split_name": self.split_name,
            "output_file": self.output_file,
            "audio_dir": self.audio_dir,
            "total_rows": self.total_rows,
            "kept_rows": self.kept_rows,
            "skipped_rows": self.skipped_rows,
            "skipped_reasons": dict(self.skipped_reasons),
        }


@dataclass
class PreparedSample:
    """目标 manifest 的一条样本。"""
    audio_path: str
    sentence: str
    duration: float
    language: Optional[str] = None
    sentences: Optional[list[dict[str, Any]]] = None

    def to_manifest_row(self) -> dict:
        row = {
            "audio": {"path": self.audio_path},
            "sentence": self.sentence,
            "duration": round(self.duration, 3),
            "sentences": self.sentences or [
                {"start": 0.0, "end": round(self.duration, 3), "text": self.sentence}
            ],
        }
        if self.language:
            row["language"] = self.language
        return row


class SourceAdapter(abc.ABC):
    """
    数据源适配器基类。

    当前只实现 Hugging Face，但 CLI 和处理流都按抽象接口组织，
    以后接 CSV/目录时直接新增子类即可。
    """

    @abc.abstractmethod
    def load_split(self, split_name: str) -> Dataset:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def source_metadata(self) -> dict:
        raise NotImplementedError


class HuggingFaceSourceAdapter(SourceAdapter):
    """Hugging Face datasets 适配器。"""

    def __init__(
        self,
        dataset_id: str,
        config_name: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.config_name = config_name
        self.revision = revision
        self.cache_dir = cache_dir

    def load_split(self, split_name: str) -> Dataset:
        dataset_path = Path(self.dataset_id).expanduser()
        if self._is_saved_dataset_dir(dataset_path):
            dataset_or_dict = load_from_disk(str(dataset_path))
            if isinstance(dataset_or_dict, Dataset):
                if split_name not in {"train", "test"}:
                    raise ManifestPreparationError(
                        f"本地 Hugging Face Dataset 不是 DatasetDict，不能选择 split `{split_name}`。"
                    )
                return dataset_or_dict

            if split_name not in dataset_or_dict:
                raise ManifestPreparationError(
                    f"本地 Hugging Face Dataset 不包含 split `{split_name}`。"
                    f"可用 split: {list(dataset_or_dict.keys())}"
                )
            return dataset_or_dict[split_name]

        return load_dataset(
            path=self.dataset_id,
            name=self.config_name,
            split=split_name,
            revision=self.revision,
            cache_dir=self.cache_dir,
        )

    @staticmethod
    def _is_saved_dataset_dir(dataset_path: Path) -> bool:
        return dataset_path.exists() and (
            (dataset_path / "dataset_info.json").exists()
            or (dataset_path / "state.json").exists()
            or (dataset_path / "dataset_dict.json").exists()
        )

    @property
    def source_metadata(self) -> dict:
        return {
            "source_type": "huggingface",
            "dataset_id": self.dataset_id,
            "config_name": self.config_name,
            "revision": self.revision,
            "cache_dir": self.cache_dir,
        }


class ManifestPreparer:
    """把数据源转换为 Whisper-Finetune 需要的 manifest。"""

    def __init__(
        self,
        adapter: SourceAdapter,
        output_dir: str,
        audio_column: str,
        text_column: str,
        language_column: Optional[str] = None,
        default_language: Optional[str] = None,
        max_samples: Optional[int] = None,
        min_duration: float = 0.5,
        max_duration: float = 30.0,
    ) -> None:
        self.adapter = adapter
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.audio_column = audio_column
        self.text_column = text_column
        self.language_column = language_column
        self.default_language = default_language
        self.max_samples = max_samples
        self.min_duration = min_duration
        self.max_duration = max_duration

    def prepare(self, train_split: str, test_split: str) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_summary = self._prepare_split(split_name=train_split, output_name="train")
        test_summary = self._prepare_split(split_name=test_split, output_name="test")

        summary = {
            "source": self.adapter.source_metadata,
            "output_dir": str(self.output_dir),
            "audio_column": self.audio_column,
            "text_column": self.text_column,
            "language_column": self.language_column,
            "default_language": self.default_language,
            "max_samples": self.max_samples,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "splits": {
                "train": train_summary.to_dict(),
                "test": test_summary.to_dict(),
            },
        }

        summary_path = self.output_dir / "prepare_summary.json"
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        return summary

    def _prepare_split(self, split_name: str, output_name: str) -> SplitSummary:
        dataset = self.adapter.load_split(split_name)
        if self.audio_column not in dataset.column_names:
            raise ManifestPreparationError(
                f"数据集 split `{split_name}` 不包含音频列 `{self.audio_column}`。"
                f"可用列: {dataset.column_names}"
            )
        if self.text_column not in dataset.column_names:
            raise ManifestPreparationError(
                f"数据集 split `{split_name}` 不包含文本列 `{self.text_column}`。"
                f"可用列: {dataset.column_names}"
            )

        dataset = dataset.cast_column(self.audio_column, Audio())

        split_dir = self.output_dir / "audio" / output_name
        split_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{output_name}.json"
        summary = SplitSummary(
            split_name=split_name,
            output_file=str(output_path),
            audio_dir=str(split_dir),
        )

        with open(output_path, "w", encoding="utf-8") as file:
            for index, row in enumerate(dataset):
                if self.max_samples is not None and summary.kept_rows >= self.max_samples:
                    break

                summary.total_rows += 1
                try:
                    sample = self._convert_row(
                        row=row,
                        split_dir=split_dir,
                        output_prefix=output_name,
                        index=index,
                    )
                except ManifestPreparationError as exc:
                    summary.skip(str(exc))
                    continue

                file.write(json.dumps(sample.to_manifest_row(), ensure_ascii=False) + "\n")
                summary.kept_rows += 1

        if summary.kept_rows == 0:
            raise ManifestPreparationError(
                f"split `{split_name}` 转换后没有有效样本，请检查字段映射和音频内容。"
            )

        return summary

    def _convert_row(
        self,
        row: dict,
        split_dir: Path,
        output_prefix: str,
        index: int,
    ) -> PreparedSample:
        audio_value = row.get(self.audio_column)
        if audio_value is None:
            raise ManifestPreparationError("missing_audio")

        text_value = row.get(self.text_column)
        if text_value is None:
            raise ManifestPreparationError("missing_text")

        sentence = str(text_value).strip()
        if not sentence:
            raise ManifestPreparationError("empty_text")

        language = self.default_language
        if self.language_column:
            language_value = row.get(self.language_column)
            if language_value is not None:
                language = str(language_value).strip() or language

        audio_path, duration = self._materialize_audio(
            audio_value=audio_value,
            split_dir=split_dir,
            output_prefix=output_prefix,
            index=index,
        )

        return PreparedSample(
            audio_path=str(audio_path),
            sentence=sentence,
            duration=duration,
            language=language,
        )

    def _materialize_audio(
        self,
        audio_value: Any,
        split_dir: Path,
        output_prefix: str,
        index: int,
    ) -> tuple[Path, float]:
        output_path = split_dir / f"{output_prefix}_{index:08d}.wav"

        # datasets.Audio() decode 后通常会给出 array / sampling_rate / path
        if isinstance(audio_value, dict):
            audio_array = audio_value.get("array")
            audio_bytes = audio_value.get("bytes")
            sampling_rate = audio_value.get("sampling_rate")
            source_path = audio_value.get("path")

            if audio_array is not None and sampling_rate:
                sf.write(output_path, audio_array, sampling_rate)
                duration = len(audio_array) / float(sampling_rate)
                return self._validate_duration(output_path.resolve(), duration)

            if audio_bytes:
                audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                sf.write(output_path, audio_array, sampling_rate)
                duration = len(audio_array) / float(sampling_rate)
                return self._validate_duration(output_path.resolve(), duration)

            if source_path and os.path.exists(source_path):
                audio_array, sampling_rate = sf.read(source_path, dtype="float32")
                sf.write(output_path, audio_array, sampling_rate)
                duration = len(audio_array) / float(sampling_rate)
                return self._validate_duration(output_path.resolve(), duration)

            raise ManifestPreparationError("unsupported_audio_dict")

        if isinstance(audio_value, str) and os.path.exists(audio_value):
            audio_array, sampling_rate = sf.read(audio_value, dtype="float32")
            sf.write(output_path, audio_array, sampling_rate)
            duration = len(audio_array) / float(sampling_rate)
            return self._validate_duration(output_path.resolve(), duration)

        raise ManifestPreparationError("unsupported_audio_value")

    def _validate_duration(self, audio_path: Path, duration: float) -> tuple[Path, float]:
        if duration < self.min_duration:
            if audio_path.exists():
                audio_path.unlink()
            raise ManifestPreparationError("duration_too_short")
        if self.max_duration != -1 and duration > self.max_duration:
            if audio_path.exists():
                audio_path.unlink()
            raise ManifestPreparationError("duration_too_long")
        return audio_path, duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare training manifests for Whisper-Finetune")
    parser.add_argument("--source-type", choices=["huggingface"], default="huggingface")
    parser.add_argument("--dataset-id", required=True, help="Hugging Face dataset repo id")
    parser.add_argument("--config-name", default=None, help="Hugging Face dataset config name")
    parser.add_argument("--revision", default=None, help="Hugging Face dataset revision")
    parser.add_argument("--cache-dir", default=None, help="datasets cache dir")
    parser.add_argument("--train-split", required=True, help="Train split name")
    parser.add_argument("--test-split", required=True, help="Test split name")
    parser.add_argument("--audio-column", default="audio", help="Audio column name")
    parser.add_argument("--text-column", default="sentence", help="Text column name")
    parser.add_argument("--language-column", default=None, help="Language column name")
    parser.add_argument("--default-language", default=None, help="Fallback language value")
    parser.add_argument("--output-dir", required=True, help="Output directory for manifests and audio")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples per split for debugging")
    parser.add_argument("--min-duration", type=float, default=0.5, help="Minimum audio duration in seconds")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Maximum audio duration in seconds, use -1 to disable")
    return parser.parse_args()


def build_adapter(args: argparse.Namespace) -> SourceAdapter:
    if args.source_type == "huggingface":
        return HuggingFaceSourceAdapter(
            dataset_id=args.dataset_id,
            config_name=args.config_name,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )

    raise ManifestPreparationError(f"不支持的数据源类型: {args.source_type}")


def main() -> None:
    args = parse_args()
    adapter = build_adapter(args)
    preparer = ManifestPreparer(
        adapter=adapter,
        output_dir=args.output_dir,
        audio_column=args.audio_column,
        text_column=args.text_column,
        language_column=args.language_column,
        default_language=args.default_language,
        max_samples=args.max_samples,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
    try:
        summary = preparer.prepare(train_split=args.train_split, test_split=args.test_split)
    except ManifestPreparationError as exc:
        print(f"Manifest preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("Manifest prepared successfully")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
