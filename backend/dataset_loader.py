"""
dataset_loader.py — 数据集扫描与加载
=====================================

职责：
  1. 扫描 GPU 服务器上的数据集根目录
  2. 识别不同格式的数据集（CSV、HuggingFace、目录）
  3. 返回数据集元信息（名称、路径、大小、行数）
  4. 加载指定数据集为 (reference, hypothesis) 对

支持的数据集格式：
  - 单个 CSV 文件：包含 transcription + predicted_string 列
  - HuggingFace 格式目录：包含 dataset_info.json
  - 普通目录：内含多个 CSV 文件

数据集根目录可通过环境变量 ASR_DATASET_DIR 配置，
默认为 /data/datasets/
"""

import csv
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

from schemas import DatasetInfo


# ──────────────────────────────────────
# 1. 配置
# ──────────────────────────────────────

# 数据集存放的根目录
# GPU 服务器上通常是 /data/datasets/ 或 /home/user/datasets/
DATASET_BASE_DIR = os.environ.get(
    "ASR_DATASET_DIR",
    os.path.join(os.path.dirname(__file__), "datasets")
)

# Hugging Face cache 根目录下常见的非数据集目录
SKIP_SCAN_DIR_NAMES = {"downloads", "downloads_extracted", "locks", "__pycache__"}


# ──────────────────────────────────────
# 2. 数据集扫描
# ──────────────────────────────────────

def scan_datasets(base_dir: Optional[str] = None) -> List[DatasetInfo]:
    """
    扫描数据集根目录，返回所有可用数据集的信息列表。
    
    扫描逻辑：
      1. 遍历根目录下的一级子项（文件和目录）
      2. 如果是 .csv 文件 → 当作单文件数据集
      3. 如果是目录且包含（或下层包含）dataset_info.json → HuggingFace 格式
      4. 如果是目录且包含（或下层包含）.csv 文件 → 普通目录数据集
      5. 其他跳过
    
    Args:
        base_dir: 数据集根目录，默认使用环境变量配置
    
    Returns:
        DatasetInfo 列表
    """
    root = Path(base_dir or DATASET_BASE_DIR)
    
    if not root.exists():
        # 目录不存在时创建它（方便首次使用）
        root.mkdir(parents=True, exist_ok=True)
        return []

    datasets = []

    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name in SKIP_SCAN_DIR_NAMES:
            continue

        try:
            if entry.is_file() and entry.suffix.lower() == '.csv':
                # ── 单 CSV 文件 ──
                info = _scan_csv_file(entry)
                if info:
                    datasets.append(info)

            elif entry.is_dir():
                # ── 目录型数据集 ──
                info = _scan_directory(entry)
                if info:
                    datasets.append(info)
        except Exception as e:
            # 单个数据集扫描失败不影响其他数据集
            print(f"[WARN] 扫描 {entry} 时出错: {e}")
            continue

    return datasets


def _display_dataset_name(path: Path) -> str:
    """把 Hugging Face cache 目录名转成更可读的显示名。"""
    name = path.name
    if name.startswith("datasets--"):
        return name[len("datasets--"):].replace("--", "/")
    if "___" in name:
        return name.replace("___", "/")
    return name


def _directory_stats(path: Path) -> Tuple[int, float]:
    """统计目录内文件数量和总大小。"""
    file_count = 0
    total_size = 0

    for file_path in path.rglob('*'):
        if file_path.is_file():
            file_count += 1
            total_size += file_path.stat().st_size

    size_mb = round(total_size / (1024 * 1024), 2)
    return file_count, size_mb


def _find_hf_dataset_dir(path: Path) -> Optional[Path]:
    """
    在目录中定位 Hugging Face 数据集的实际叶子目录。

    `datasets` 的本地 cache 一般是多层结构：
      owner___dataset/config/version/hash/dataset_info.json
    当前扫描入口通常只会拿到 owner___dataset 这一层，所以这里要往下找。
    """
    direct_info = path / "dataset_info.json"
    if direct_info.exists():
        return path

    candidates = [info_file.parent for info_file in path.rglob("dataset_info.json")]
    if not candidates:
        return None

    # 取最近一次生成、且层级最深的那个目录，避免扫到旧 cache 版本。
    candidates.sort(
        key=lambda p: (p.stat().st_mtime, len(p.parts)),
        reverse=True,
    )
    return candidates[0]


def _find_csv_files(path: Path) -> List[Path]:
    """递归查找目录里的 CSV 文件。"""
    return sorted(file_path for file_path in path.rglob("*.csv") if file_path.is_file())


def _scan_csv_file(path: Path) -> Optional[DatasetInfo]:
    """
    扫描单个 CSV 文件，检查是否包含必要的列。
    
    必要列：transcription（或 reference）+ predicted_string（或 hypothesis）
    """
    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        
        # 读取表头，检查列名
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return None
            
            try:
                _resolve_text_columns(header)
            except ValueError:
                return None  # 不是 ASR 评测用的 CSV
            
            # 计算行数（跳过表头）
            row_count = sum(1 for _ in reader)

        return DatasetInfo(
            name=path.stem,        # 文件名（不含扩展名）
            path=str(path),
            file_count=1,
            total_rows=row_count,
            format="csv",
            size_mb=round(size_mb, 2),
        )
    except Exception:
        return None


def _scan_directory(path: Path) -> Optional[DatasetInfo]:
    """
    扫描目录型数据集。
    
    优先级：
      1. 如果有 dataset_info.json（允许在子目录）→ HuggingFace 格式
      2. 如果有 CSV 文件（允许在子目录）→ 普通目录
      3. 否则跳过
    """
    # 检查 HuggingFace 格式
    hf_dir = _find_hf_dataset_dir(path)
    if hf_dir:
        file_count, size_mb = _directory_stats(hf_dir)
        return DatasetInfo(
            name=_display_dataset_name(path),
            path=str(hf_dir),
            file_count=file_count,
            total_rows=_count_hf_rows(hf_dir),
            format="huggingface",
            size_mb=size_mb,
        )

    # 检查是否有 CSV 文件
    csv_files = _find_csv_files(path)
    if csv_files:
        # 统计所有 CSV 的总行数
        valid_infos = []
        for csv_file in csv_files:
            info = _scan_csv_file(csv_file)
            if info:
                valid_infos.append(info)

        if valid_infos:
            total_rows = sum(info.total_rows or 0 for info in valid_infos)
            file_count, size_mb = _directory_stats(path)
            return DatasetInfo(
                name=path.name,
                path=str(valid_infos[0].path) if len(valid_infos) == 1 else str(path),
                file_count=file_count,
                total_rows=total_rows or None,
                format="csv",
                size_mb=size_mb,
            )

    return None


def _count_hf_rows(path: Path) -> Optional[int]:
    """
    尝试从 HuggingFace dataset_info.json 中读取行数。
    如果读不到就返回 None。
    """
    try:
        info_file = path / "dataset_info.json"
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        # HuggingFace 格式中行数可能在不同位置
        if 'splits' in info:
            total = 0
            for split_info in info['splits'].values() if isinstance(info['splits'], dict) else info['splits']:
                if isinstance(split_info, dict):
                    total += split_info.get('num_examples', 0)
            return total if total > 0 else None
        
        return None
    except Exception:
        return None


# ──────────────────────────────────────
# 3. 数据集加载
# ──────────────────────────────────────

# 支持的列名映射（统一到 ref / hyp）
REF_COLUMNS = {'transcription', 'reference', 'ref', 'ground_truth', 'gt', 'label', 'sentence'}
HYP_COLUMNS = {'predicted_string', 'prediction', 'pred', 'hypothesis', 'hyp', 'asr_output'}


def load_dataset(dataset_path: str) -> List[Tuple[str, str]]:
    """
    加载数据集，返回 (reference, hypothesis) 对的列表。
    
    支持：
      - CSV 文件路径 → 直接读取
      - 目录路径 → 查找其中的 CSV 文件
    
    Args:
        dataset_path: CSV 文件路径或包含 CSV 的目录路径
    
    Returns:
        [(ref_text, hyp_text), ...] 列表
    
    Raises:
        FileNotFoundError: 路径不存在
        ValueError: 找不到必要的列
    """
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

    if path.is_file() and path.suffix.lower() == '.csv':
        return _load_csv(path)
    
    if path.is_dir():
        # 先递归查找 CSV 文件
        csv_files = _find_csv_files(path)
        if csv_files:
            # 合并所有 CSV
            all_pairs = []
            for csv_file in csv_files:
                try:
                    pairs = _load_csv(csv_file)
                    all_pairs.extend(pairs)
                except ValueError:
                    continue  # 跳过格式不对的文件
            
            if all_pairs:
                return all_pairs

        # 再尝试当作 Hugging Face datasets cache 目录读取
        hf_dir = _find_hf_dataset_dir(path)
        if hf_dir:
            return _load_huggingface_directory(hf_dir)

        raise FileNotFoundError(f"目录中没有找到可加载的数据文件: {dataset_path}")

    raise ValueError(f"不支持的数据集格式: {dataset_path}")


def _load_csv(path: Path) -> List[Tuple[str, str]]:
    """
    从 CSV 文件加载 (reference, hypothesis) 对。
    
    自动识别列名（支持多种命名方式）。
    """
    pairs = []

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        if reader.fieldnames is None:
            raise ValueError(f"CSV 文件为空: {path}")

        # 找到参考文本列和预测文本列
        ref_col, hyp_col = _resolve_text_columns(reader.fieldnames)

        for row in reader:
            ref = (row.get(ref_col) or "").strip()
            hyp = (row.get(hyp_col) or "").strip()
            if ref:  # 跳过空行
                pairs.append((ref, hyp))

    return pairs


def _resolve_text_columns(fieldnames: List[str]) -> Tuple[str, str]:
    """在列名列表里定位 reference / hypothesis 两列。"""
    ref_col = None
    hyp_col = None

    for col in fieldnames:
        col_lower = col.strip().lower()
        if ref_col is None and col_lower in REF_COLUMNS:
            ref_col = col
        elif hyp_col is None and col_lower in HYP_COLUMNS:
            hyp_col = col

    if ref_col is None:
        raise ValueError(
            f"找不到参考文本列。CSV/HF 列名: {fieldnames}。"
            f"需要以下之一: {sorted(REF_COLUMNS)}"
        )
    if hyp_col is None:
        raise ValueError(
            f"找不到预测文本列。CSV/HF 列名: {fieldnames}。"
            f"需要以下之一: {sorted(HYP_COLUMNS)}"
        )

    return ref_col, hyp_col


def _load_huggingface_directory(path: Path) -> List[Tuple[str, str]]:
    """
    读取 Hugging Face datasets cache 目录中的 Arrow 文件。

    这里兼容的是 `datasets.load_dataset(...)` 生成的本地 cache，
    不是 `save_to_disk()` 的目录结构。
    """
    try:
        from datasets import Dataset as HFDataset
    except ImportError as exc:
        raise ImportError(
            "检测到 Hugging Face 数据集缓存目录，但当前环境未安装 `datasets`。"
            "请先安装 `datasets`，或把数据导出成包含 reference + hypothesis 的 CSV。"
        ) from exc

    arrow_files = sorted(
        file_path
        for file_path in path.glob("*.arrow")
        if file_path.is_file() and not file_path.name.startswith("cache-")
    )
    if not arrow_files:
        raise FileNotFoundError(f"Hugging Face 数据集目录中没有找到 Arrow 文件: {path}")

    all_pairs = []
    last_column_error = None

    for arrow_file in arrow_files:
        dataset = HFDataset.from_file(str(arrow_file))

        try:
            ref_col, hyp_col = _resolve_text_columns(dataset.column_names)
        except ValueError as exc:
            last_column_error = exc
            continue

        for row in dataset:
            ref = str(row.get(ref_col) or "").strip()
            hyp = str(row.get(hyp_col) or "").strip()
            if ref:
                all_pairs.append((ref, hyp))

    if all_pairs:
        return all_pairs

    if last_column_error is not None:
        raise last_column_error

    raise ValueError(f"Hugging Face 数据集中没有找到有效的 ASR 文本对: {path}")
