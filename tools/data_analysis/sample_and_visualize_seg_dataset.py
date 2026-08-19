#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【中文使用说明】
本脚本用于对 AgenticSeg-RL 的分割类数据集进行抽样可视化，并导出逐条样本的原始文本信息。

每条采样样本会生成：
1) 一张可视化图（左右两栏）：左=原图，右=原图+GT掩码 overlay
2) 一个同名 JSON（例如 0001_xxx.png 对应 0001_xxx.json），用于保存该条样本的文本字段原值

另外会自动生成 summary.json 统计抽样与导出信息。

支持范围：
- ./data 下除 agenticRL* 以外的数据集（通过 --dataset-path 指定）
- 支持常见字段风格：image + mask / solution_mask + text / problem

常用命令（可直接复制）：
1. 默认：采样 ReasonSegX_train/train，输出到 vis/ReasonSegX
   python tools/data_analysis/sample_and_visualize_seg_dataset.py

2. 随机采样 80 条（固定随机种子）
   python tools/data_analysis/sample_and_visualize_seg_dataset.py \
     --dataset-path ./data/ReasonSegX_train/train \
     --num-samples 80 \
     --strategy random \
     --seed 42 \
     --output-dir ./vis/ReasonSegX

3. 按 reasoning_type 分层采样
   python tools/data_analysis/sample_and_visualize_seg_dataset.py \
     --dataset-path ./data/ReasonSegX_test \
     --strategy stratified-group \
     --group-field reasoning_type \
     --num-samples 120 \
     --output-dir ./vis/ReasonSegX_test

4. 按 mask 面积桶分层采样（更关注大小目标分布）
   python tools/data_analysis/sample_and_visualize_seg_dataset.py \
     --dataset-path ./data/MUSE/val \
     --strategy stratified-mask-area \
     --num-samples 120 \
     --output-dir ./vis/MUSE_val
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.ipc as ipc
from PIL import Image, ImageDraw


DEFAULT_DATASET_PATH = Path("./data/ReasonSegX_train/train")
DEFAULT_OUTPUT_DIR = Path("./vis/ReasonSegX")
DATA_ROOT = Path("./data")
MASK_CANDIDATE_FIELDS = ("solution_mask", "mask", "gt_mask", "masks")
TEXT_CANDIDATE_FIELDS = ("problem", "text", "question", "instruction")
IMAGE_CANDIDATE_FIELDS = ("image", "img", "source_image")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample and visualize segmentation-style datasets in AgenticSeg-RL")
    p.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH, help="Path to split-dir or dataset root")
    p.add_argument("--split", type=str, default="train", help="Split name when dataset-path points to DatasetDict root")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--strategy", choices=["random", "first-n", "stratified-group", "stratified-mask-area"], default="random")
    p.add_argument("--group-field", type=str, default="reasoning_type", help="Used by stratified-group")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overlay-alpha", type=float, default=0.35)
    p.add_argument("--overlay-color", type=str, default="0,255,0", help="R,G,B")
    p.add_argument("--save-per-sample-manifest", action="store_true", help="Also save manifest.jsonl for sampled rows")
    return p.parse_args()


def parse_rgb(text: str) -> Tuple[int, int, int]:
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid RGB format: {text}")
    rgb = tuple(int(v) for v in parts)
    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError(f"RGB values must be in [0,255], got: {rgb}")
    return rgb  # type: ignore[return-value]


def ensure_not_agenticrl(path: Path) -> None:
    low = str(path).lower()
    if "agenticrl" in low or "agentic_rl" in low:
        raise ValueError("This script is for non-agenticRL datasets only. Please use a non-agenticRL dataset path.")


def resolve_split_dir(dataset_path: Path, split: str) -> Path:
    dataset_path = dataset_path.resolve()
    if any(dataset_path.glob("data-*.arrow")) and (dataset_path / "state.json").exists():
        return dataset_path
    split_dir = dataset_path / split
    if split_dir.exists() and any(split_dir.glob("data-*.arrow")):
        return split_dir

    child_split_dirs = []
    for child in sorted(dataset_path.iterdir()) if dataset_path.exists() else []:
        if child.is_dir() and (child / "state.json").exists() and any(child.glob("data-*.arrow")):
            child_split_dirs.append(child)
    if len(child_split_dirs) == 1:
        return child_split_dirs[0]

    if len(child_split_dirs) > 1:
        names = ", ".join([x.name for x in child_split_dirs])
        raise FileNotFoundError(
            f"Multiple split dirs found under {dataset_path}: {names}. "
            f"Please specify --split or point --dataset-path directly to one split dir."
        )

    raise FileNotFoundError(
        f"Cannot resolve split dir from dataset-path={dataset_path}. "
        f"Expected either <path>/data-*.arrow or <path>/<split>/data-*.arrow"
    )


def get_arrow_files(split_dir: Path) -> List[Path]:
    files = sorted(split_dir.glob("data-*.arrow"))
    if not files:
        raise FileNotFoundError(f"No data-*.arrow found in {split_dir}")
    return files


def _load_total_rows_hint(split_dir: Path) -> Optional[int]:
    info_path = split_dir / "dataset_info.json"
    if not info_path.exists():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    splits = info.get("splits", {})
    if not isinstance(splits, dict):
        return None
    for split_name in ("train", "val", "validation", "test"):
        split_obj = splits.get(split_name)
        if isinstance(split_obj, dict):
            n = split_obj.get("num_examples")
            if isinstance(n, int):
                return n
    return None


def _print_progress(prefix: str, cur: int, total: Optional[int]) -> None:
    if total is not None and total > 0:
        width = 28
        ratio = max(0.0, min(1.0, float(cur) / float(total)))
        done = int(width * ratio)
        bar = "#" * done + "-" * (width - done)
        print(f"\r[{prefix}] |{bar}| {cur}/{total} ({ratio * 100:.1f}%)", end="", flush=True)
    else:
        print(f"\r[{prefix}] {cur}", end="", flush=True)


def sanitize_filename(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in name)


def decode_image_struct(image_struct: Any) -> Image.Image:
    if not isinstance(image_struct, dict):
        raise ValueError(f"image field must be dict, got {type(image_struct)}")
    img_bytes = image_struct.get("bytes")
    img_path = image_struct.get("path")
    if isinstance(img_bytes, (bytes, bytearray)) and len(img_bytes) > 0:
        with Image.open(io.BytesIO(img_bytes)) as im:
            return im.convert("RGB")
    if isinstance(img_path, str) and img_path:
        with Image.open(img_path) as im:
            return im.convert("RGB")
    raise ValueError("image struct has neither valid bytes nor path")


def _to_bool_2d(mask_like: Any) -> Optional[np.ndarray]:
    arr = np.array(mask_like)
    if arr.ndim != 2:
        return None
    if arr.dtype == bool:
        return arr
    if np.issubdtype(arr.dtype, np.number):
        return arr > 0
    try:
        return arr.astype(np.bool_)
    except Exception:
        return None


def union_mask(mask_value: Any) -> Optional[np.ndarray]:
    if mask_value is None:
        return None

    arr = np.array(mask_value)
    if arr.ndim == 2:
        m = _to_bool_2d(arr)
        return m if m is not None else None
    if arr.ndim == 3:
        if np.issubdtype(arr.dtype, np.number) or arr.dtype == bool:
            return np.any(arr > 0, axis=0)

    if isinstance(mask_value, list):
        merged = None
        for part in mask_value:
            m = _to_bool_2d(part)
            if m is None:
                continue
            merged = m if merged is None else np.logical_or(merged, m)
        return merged
    return None


def find_image_field(schema_names: Sequence[str]) -> str:
    for k in IMAGE_CANDIDATE_FIELDS:
        if k in schema_names:
            return k
    raise KeyError(f"Cannot find image field in schema. candidates={IMAGE_CANDIDATE_FIELDS}, actual={list(schema_names)}")


def find_mask_field(schema_names: Sequence[str]) -> Optional[str]:
    for k in MASK_CANDIDATE_FIELDS:
        if k in schema_names:
            return k
    return None


def mask_area_bucket(ratio: Optional[float]) -> str:
    if ratio is None:
        return "unknown"
    if ratio < 0.01:
        return "tiny(<1%)"
    if ratio < 0.05:
        return "small(1%-5%)"
    if ratio < 0.20:
        return "medium(5%-20%)"
    return "large(>=20%)"


def sample_round_robin(groups: Dict[str, List[int]], total: int, rng: random.Random) -> List[int]:
    keys = sorted(groups.keys())
    for k in keys:
        rng.shuffle(groups[k])
    chosen: List[int] = []
    while len(chosen) < total and keys:
        next_keys = []
        for k in keys:
            arr = groups[k]
            if arr:
                chosen.append(arr.pop())
                if len(chosen) >= total:
                    break
            if arr:
                next_keys.append(k)
        keys = next_keys
    return chosen


def make_overlay(image: Image.Image, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    out = base.copy()
    c = np.asarray(color, dtype=np.float32)
    out[m] = (1.0 - alpha) * out[m] + alpha * c
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def make_panel(left: Image.Image, right: Image.Image, left_title: str = "Original", right_title: str = "Original + GT Overlay") -> Image.Image:
    left = left.convert("RGB")
    right = right.convert("RGB").resize(left.size, Image.Resampling.BILINEAR)
    w, h = left.size
    title_h = 28
    canvas = Image.new("RGB", (w * 2, h + title_h), (255, 255, 255))
    canvas.paste(left, (0, title_h))
    canvas.paste(right, (w, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), left_title, fill=(0, 0, 0))
    draw.text((w + 8, 6), right_title, fill=(0, 0, 0))
    return canvas


def to_jsonable_record(row: Dict[str, Any], image_field: str, mask_field: Optional[str]) -> Dict[str, Any]:
    out = {}
    for k, v in row.items():
        if k == image_field:
            continue
        if mask_field is not None and k == mask_field:
            continue
        try:
            json.dumps(v, ensure_ascii=False)
            out[k] = v
        except TypeError:
            out[k] = str(v)
    return out


def collect_scan_meta(split_dir: Path, group_field: str) -> Dict[str, Any]:
    arrow_files = get_arrow_files(split_dir)
    metas = []
    global_idx = 0
    image_field: Optional[str] = None
    mask_field: Optional[str] = None
    schema_names: Optional[List[str]] = None

    total_rows_hint = _load_total_rows_hint(split_dir)
    print("[INFO] 开始扫描数据元信息...", flush=True)
    for fp in arrow_files:
        reader = ipc.open_stream(fp)
        for batch in reader:
            if schema_names is None:
                schema_names = list(batch.schema.names)
                image_field = find_image_field(schema_names)
                mask_field = find_mask_field(schema_names)
            cols = {name: batch.column(name) for name in batch.schema.names}
            for i in range(batch.num_rows):
                row_id = cols["id"][i].as_py() if "id" in cols else cols.get("image_id", [f"row_{global_idx}"])[i].as_py()
                group_val = cols[group_field][i].as_py() if group_field in cols else "ungrouped"
                ratio = None
                if mask_field is not None:
                    mask_val = cols[mask_field][i].as_py()
                    m = union_mask(mask_val)
                    if m is not None and m.size > 0:
                        ratio = float(np.count_nonzero(m) / float(m.size))
                metas.append(
                    {
                        "global_index": global_idx,
                        "id": str(row_id),
                        "group": str(group_val) if group_val is not None else "ungrouped",
                        "mask_ratio": ratio,
                        "mask_bucket": mask_area_bucket(ratio),
                    }
                )
                global_idx += 1
                if global_idx == 1 or global_idx % 100 == 0:
                    _print_progress("scan", global_idx, total_rows_hint)
    if global_idx > 0:
        _print_progress("scan", global_idx, total_rows_hint)
        print("", flush=True)
    return {
        "metas": metas,
        "arrow_files": arrow_files,
        "image_field": image_field,
        "mask_field": mask_field,
        "schema_names": schema_names or [],
    }


def choose_indices(metas: List[Dict[str, Any]], num_samples: int, strategy: str, seed: int) -> List[int]:
    rng = random.Random(seed)
    total = min(max(0, num_samples), len(metas))
    all_indices = [int(x["global_index"]) for x in metas]
    if strategy == "first-n":
        return all_indices[:total]
    if strategy == "random":
        rng.shuffle(all_indices)
        return all_indices[:total]

    if strategy == "stratified-group":
        groups: Dict[str, List[int]] = {}
        for m in metas:
            groups.setdefault(str(m["group"]), []).append(int(m["global_index"]))
        return sample_round_robin(groups=groups, total=total, rng=rng)

    if strategy == "stratified-mask-area":
        groups = {}
        for m in metas:
            groups.setdefault(str(m["mask_bucket"]), []).append(int(m["global_index"]))
        return sample_round_robin(groups=groups, total=total, rng=rng)

    raise ValueError(f"Unsupported strategy: {strategy}")


def count_by_key(items: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in items:
        k = str(x.get(key, "unknown"))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def process_selected(
    split_dir: Path,
    selected_indices: List[int],
    image_field: str,
    mask_field: Optional[str],
    output_dir: Path,
    color: Tuple[int, int, int],
    alpha: float,
    save_per_sample_manifest: bool,
) -> Dict[str, Any]:
    index_rank = {gidx: rank for rank, gidx in enumerate(selected_indices)}
    selected_set = set(selected_indices)
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    missing_mask = 0
    manifest_lines: List[Dict[str, Any]] = []

    global_idx = 0
    print("[INFO] 开始导出可视化与JSON...", flush=True)
    target_total = len(selected_indices)
    for fp in get_arrow_files(split_dir):
        reader = ipc.open_stream(fp)
        for batch in reader:
            cols = {name: batch.column(name) for name in batch.schema.names}
            for i in range(batch.num_rows):
                if global_idx not in selected_set:
                    global_idx += 1
                    continue

                rank = index_rank[global_idx]
                row = {name: cols[name][i].as_py() for name in batch.schema.names}
                sample_id = row.get("id", row.get("image_id", f"row_{global_idx}"))
                sample_id = str(sample_id)
                base_name = f"{rank:04d}_{sanitize_filename(sample_id)}"
                out_png = output_dir / f"{base_name}.png"
                out_json = output_dir / f"{base_name}.json"

                try:
                    image = decode_image_struct(row[image_field])
                except Exception:
                    global_idx += 1
                    continue

                overlay = image.copy()
                mask_ratio = None
                if mask_field is not None:
                    m = union_mask(row.get(mask_field))
                    if m is not None and m.size > 0:
                        h, w = image.size[1], image.size[0]
                        if m.shape != (h, w):
                            m_img = Image.fromarray((m.astype(np.uint8) * 255), mode="L")
                            m_img = m_img.resize((w, h), Image.Resampling.NEAREST)
                            m = np.array(m_img, dtype=np.uint8) > 0
                        mask_ratio = float(np.count_nonzero(m) / float(m.size))
                        overlay = make_overlay(image=image, mask=m, color=color, alpha=alpha)
                    else:
                        missing_mask += 1
                else:
                    missing_mask += 1

                panel = make_panel(image, overlay)
                panel.save(out_png)

                text_fields_raw = {k: v for k, v in row.items() if isinstance(v, str)}
                json_obj = {
                    "global_index": global_idx,
                    "sample_rank": rank,
                    "sample_id": sample_id,
                    "text_fields_raw": text_fields_raw,
                    "raw_record_without_image_and_mask": to_jsonable_record(
                        row=row,
                        image_field=image_field,
                        mask_field=mask_field,
                    ),
                    "mask_field": mask_field,
                    "mask_area_ratio": mask_ratio,
                    "image_size": {"width": image.size[0], "height": image.size[1]},
                    "output_image": str(out_png),
                }
                out_json.write_text(json.dumps(json_obj, ensure_ascii=False, indent=2), encoding="utf-8")
                rendered += 1
                if rendered == 1 or rendered % 10 == 0 or rendered == target_total:
                    _print_progress("render", rendered, target_total)

                manifest_lines.append(
                    {
                        "global_index": global_idx,
                        "sample_rank": rank,
                        "sample_id": sample_id,
                        "output_image": str(out_png),
                        "output_json": str(out_json),
                        "mask_area_ratio": mask_ratio,
                    }
                )
                global_idx += 1

    if target_total > 0:
        _print_progress("render", rendered, target_total)
        print("", flush=True)

    if save_per_sample_manifest:
        manifest_path = output_dir / "sample_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as f:
            for row in sorted(manifest_lines, key=lambda x: int(x["sample_rank"])):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ratios = [float(x["mask_area_ratio"]) for x in manifest_lines if x.get("mask_area_ratio") is not None]
    return {
        "rendered_samples": rendered,
        "missing_or_invalid_mask_samples": missing_mask,
        "mean_mask_area_ratio": safe_mean(ratios),
        "min_mask_area_ratio": (float(min(ratios)) if ratios else None),
        "max_mask_area_ratio": (float(max(ratios)) if ratios else None),
    }


def main() -> None:
    args = parse_args()
    ensure_not_agenticrl(args.dataset_path)
    split_dir = resolve_split_dir(args.dataset_path, args.split)
    color = parse_rgb(args.overlay_color)
    alpha = max(0.0, min(1.0, float(args.overlay_alpha)))

    if DATA_ROOT in split_dir.parents and "agenticrl" in str(split_dir).lower():
        raise ValueError(f"Refuse agenticRL dataset: {split_dir}")

    scan = collect_scan_meta(split_dir=split_dir, group_field=args.group_field)
    metas = scan["metas"]
    if not metas:
        raise ValueError(f"No rows found in {split_dir}")

    selected_indices = choose_indices(
        metas=metas,
        num_samples=args.num_samples,
        strategy=args.strategy,
        seed=args.seed,
    )
    selected_meta_by_idx = {int(x["global_index"]): x for x in metas}
    selected_meta = [selected_meta_by_idx[i] for i in selected_indices if i in selected_meta_by_idx]

    stats_export = process_selected(
        split_dir=split_dir,
        selected_indices=selected_indices,
        image_field=str(scan["image_field"]),
        mask_field=scan["mask_field"],
        output_dir=args.output_dir,
        color=color,
        alpha=alpha,
        save_per_sample_manifest=args.save_per_sample_manifest,
    )

    summary = {
        "dataset_path_arg": str(args.dataset_path),
        "resolved_split_dir": str(split_dir),
        "split_arg": args.split,
        "strategy": args.strategy,
        "group_field": args.group_field,
        "seed": args.seed,
        "requested_num_samples": args.num_samples,
        "scanned_total_rows": len(metas),
        "actual_selected_rows": len(selected_indices),
        "scan_group_distribution": count_by_key(metas, "group"),
        "sampled_group_distribution": count_by_key(selected_meta, "group"),
        "scan_mask_bucket_distribution": count_by_key(metas, "mask_bucket"),
        "sampled_mask_bucket_distribution": count_by_key(selected_meta, "mask_bucket"),
        "schema_names": scan["schema_names"],
        "image_field": scan["image_field"],
        "mask_field": scan["mask_field"],
        "overlay_alpha": alpha,
        "overlay_color_rgb": list(color),
        "output_dir": str(args.output_dir),
        "export_stats": stats_export,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] dataset={split_dir}")
    print(f"[OK] strategy={args.strategy}")
    print(f"[OK] selected={len(selected_indices)} / scanned={len(metas)}")
    print(f"[OK] output={args.output_dir}")
    print(f"[OK] summary={summary_path}")


if __name__ == "__main__":
    main()
