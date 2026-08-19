#!/usr/bin/env python3
"""【中文说明】构建 AgenticRL Task-3（自我纠正）数据集。

目标：
- 输入：原图 + 问题 + GT 并集掩码
- 过程：对 GT 人工制造“多画/少画”错误
- 输出：点监督样本，训练模型定位掩码错误区域

构建逻辑：
1. 将多实例 GT 合并成 `gt_union_mask`。
2. 随机决定生成：
   - `false_positive_addition`（多画，点标签=0）
   - `false_negative_deletion`（少画，点标签=1）
3. 用凸多边形制造扰动并得到 `corrupted_union_mask`。
4. 以“实际变化区域 changed_region”作为监督基础：
   - `target_point_2d` = changed_region 中心（并吸附到有效像素）
   - 避免“多边形中心与真实变化区域偏离”的监督噪声

删除与增加的细节：
- 删除（false_negative_deletion）：
  - 中心点从 GT 内部采样，生成多边形后做减法；
  - 监督点取“实际删除区域”中心。
- 增加（false_positive_addition）支持 `--addition-mode`：
  - `boundary`（默认）：从 GT 边缘点生长，视觉更像“边缘多画一块”；
  - `random`：随机外部点生长；
  - `mixed`：两者混合采样。
  - 监督点取“实际新增区域”中心。

输出结构：
- `task3_self_correction.jsonl`
  - 主键与映射：`id/source_id/global_index/variant_index`
  - 监督字段：`target_point_2d/target_point_label/target_answer`
  - 可追溯字段：`seed_point_2d/addition_mode_used/polygon_xy`
  - 资源路径：`mask_npz/overlay_image/overlay_with_center`
- `masks/*.npz`
  - `corrupted_union_mask`, `gt_union_mask`
  - `changed_region_mask`, `gt_mask_stack`
- `overlays/false_positive_addition/*.png`
- `overlays/false_negative_deletion/*.png`

参数说明（重点）：
- 数据：`--dataset-path --split --output-dir`
- 扰动分布：`--add-prob --addition-mode --variants-per-question`
- 扰动几何：`--min-change-pixels --min/max-vertices --radius-min/max-ratio`
- 工程：`--sample-ratio --max-rows --seed --log-every --summary-every --save-center-debug`

最简配置命令：
python tools/datasets/build_task3_self_correction_dataset.py \
  --dataset-path data/base_segmentation_train \
  --split train

常用配置命令：
python tools/datasets/build_task3_self_correction_dataset.py \
  --dataset-path data/base_segmentation_train \
  --split train \
  --output-dir data/agenticrl/task3_self_correction/base_dataset \
  --add-prob 0.5 \
  --addition-mode boundary \
  --variants-per-question 1 \
  --seed 42

完整配置命令（常见全量）：
python tools/datasets/build_task3_self_correction_dataset.py \
  --dataset-path data/base_segmentation_train \
  --split train \
  --output-dir data/agenticrl/task3_self_correction/base_dataset \
  --seed 42 \
  --add-prob 0.5 \
  --addition-mode boundary \
  --variants-per-question 1 \
  --sample-ratio 1.0 \
  --min-change-pixels 50 \
  --min-vertices 8 \
  --max-vertices 18 \
  --radius-min-ratio 0.01 \
  --radius-max-ratio 0.15 \
  --max-rows 0 \
  --log-every 20 \
  --summary-every 100

# 2026 0520
# 最大的ratio太大了，掩码被删除的部分太多，剩下一堆零碎的掩码，被模型认为是多出来的噪声
python tools/datasets/build_task3_self_correction_dataset.py \
  --dataset-path data/base_segmentation_train \
  --split train \
  --output-dir data/agenticrl/task3_self_correction/base_dataset_refined \
  --seed 42 \
  --add-prob 0.5 \
  --addition-mode boundary \
  --variants-per-question 1 \
  --min-change-pixels 50 \
  --min-vertices 8 \
  --max-vertices 18 \
  --radius-min-ratio 0.01 \
  --radius-max-ratio 0.08 \
  --log-every 50 \
  --summary-every 100

  
python tools/datasets/build_task3_self_correction_dataset.py \
  --dataset-path data/ReasonSegX_train \
  --split train \
  --output-dir data/agenticrl/task3_self_correction/ReasonSegX \
  --seed 42 \
  --add-prob 0.5 \
  --addition-mode boundary \
  --variants-per-question 1 \
  --sample-ratio 1.0 \
  --min-change-pixels 50 \
  --min-vertices 8 \
  --max-vertices 18 \
  --radius-min-ratio 0.01 \
  --radius-max-ratio 0.15 \
  --max-rows 0 \
  --log-every 20 \
  --summary-every 100
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.ipc as ipc
from PIL import Image, ImageDraw

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.datasets.agenticrl_dataset_common import (  # noqa: E402
    decode_image_from_arrow_struct,
    iter_dataset_rows,
    mask_boundary,
    mask_center_xy_on_true,
    make_mask_overlay,
    resize_mask_to_shape,
    resolve_split_dir,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build AgenticRL Task-3 self-correction dataset")
    p.add_argument("--dataset-path", type=Path, default=Path("data/base_segmentation_train"))
    p.add_argument("--split", type=str, default="train")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir. Omit to use data/agenticrl/task3_self_correction/<dataset_name>/",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--add-prob", type=float, default=0.5)
    p.add_argument(
        "--addition-mode",
        type=str,
        default="boundary",
        choices=["boundary", "random", "mixed"],
        help="How to sample center for false_positive_addition",
    )
    p.add_argument("--variants-per-question", type=int, default=1)
    p.add_argument("--sample-ratio", type=float, default=1.0)
    p.add_argument("--min-change-pixels", type=int, default=50)
    p.add_argument("--min-vertices", type=int, default=8)
    p.add_argument("--max-vertices", type=int, default=18)
    p.add_argument("--radius-min-ratio", type=float, default=0.02)
    p.add_argument("--radius-max-ratio", type=float, default=0.12)
    p.add_argument("--max-rows", type=int, default=0, help="0 means full dataset")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--summary-every", type=int, default=100)
    p.add_argument("--save-center-debug", action="store_true")
    return p.parse_args()


def estimate_total_rows(split_dir: Path, split: str) -> int:
    info_path = split_dir / "dataset_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            n = info.get("splits", {}).get(split, {}).get("num_examples", None)
            if isinstance(n, int) and n > 0:
                return int(n)
        except Exception:
            pass
    total = 0
    for fp in sorted(split_dir.glob("data-*.arrow")):
        reader = ipc.open_stream(fp)
        for b in reader:
            total += int(b.num_rows)
    return total


def progress_bar(cur: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return f"{cur}"
    ratio = min(max(float(cur) / float(total), 0.0), 1.0)
    fill = int(round(width * ratio))
    return f"[{'#' * fill}{'-' * (width - fill)}] {cur}/{total} ({ratio * 100:.1f}%)"


def sanitize_stem(text: str) -> str:
    s = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in str(text))
    s = s.strip("_.")
    return s or "sample"


def infer_dataset_name(dataset_path: Path, split_dir: Path) -> str:
    if split_dir.name in {"train", "val", "validation", "test"}:
        return split_dir.parent.name
    if dataset_path.name:
        return dataset_path.name
    return split_dir.name


def random_point_from_mask(mask: np.ndarray, want_inside: bool, rng: random.Random) -> Optional[Tuple[int, int]]:
    ys, xs = np.where(mask if want_inside else ~mask)
    if len(xs) == 0:
        return None
    k = rng.randrange(len(xs))
    return int(xs[k]), int(ys[k])


def random_point_from_boundary(mask: np.ndarray, rng: random.Random) -> Optional[Tuple[int, int]]:
    bnd = mask_boundary(mask)
    ys, xs = np.where(bnd)
    if len(xs) == 0:
        return None
    k = rng.randrange(len(xs))
    return int(xs[k]), int(ys[k])


def random_polygon(
    cx: int,
    cy: int,
    width: int,
    height: int,
    rng: random.Random,
    n_vertices: int,
    r_min: int,
    r_max: int,
) -> List[Tuple[int, int]]:
    def cross(o: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def convex_hull(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        pts = sorted(set(points))
        if len(pts) <= 2:
            return pts
        lower: List[Tuple[int, int]] = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper: List[Tuple[int, int]] = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    samples = max(n_vertices * 3, 24)
    cloud: List[Tuple[int, int]] = []
    for _ in range(samples):
        a = rng.uniform(0.0, 2.0 * math.pi)
        r = rng.uniform(r_min, r_max)
        x = int(round(cx + r * math.cos(a)))
        y = int(round(cy + r * math.sin(a)))
        x = min(max(0, x), width - 1)
        y = min(max(0, y), height - 1)
        cloud.append((x, y))

    hull = convex_hull(cloud)
    if len(hull) <= n_vertices:
        return hull
    idx = np.linspace(0, len(hull) - 1, num=n_vertices, dtype=int)
    return [hull[i] for i in idx]


def polygon_to_mask(width: int, height: int, polygon_xy: Sequence[Tuple[int, int]]) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    draw.polygon(list(polygon_xy), fill=255)
    return np.array(canvas, dtype=np.uint8) > 0


def draw_center_debug(image: Image.Image, center_xy: Sequence[int], label: str) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x, y = int(center_xy[0]), int(center_xy[1])
    r = 5
    draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=2)
    draw.rectangle([x + 6, y - 10, x + 120, y + 10], fill=(0, 0, 0))
    draw.text((x + 8, y - 8), label, fill=(255, 255, 255))
    return out


def build_one_corruption(
    gt_union_mask: np.ndarray,
    add_prob: float,
    addition_mode: str,
    min_change_pixels: int,
    rng: random.Random,
    min_vertices: int,
    max_vertices: int,
    radius_min_ratio: float,
    radius_max_ratio: float,
    max_try: int = 30,
) -> Optional[Dict[str, Any]]:
    h, w = gt_union_mask.shape
    min_side = min(h, w)
    r_min = max(4, int(round(min_side * radius_min_ratio)))
    r_max = max(r_min + 1, int(round(min_side * radius_max_ratio)))
    for _ in range(max_try):
        do_add = rng.random() < add_prob
        add_mode_used = None
        if do_add:
            if addition_mode == "boundary":
                center = random_point_from_boundary(gt_union_mask, rng=rng)
                add_mode_used = "boundary"
            elif addition_mode == "random":
                center = random_point_from_mask(gt_union_mask, want_inside=False, rng=rng)
                add_mode_used = "random"
            else:
                add_mode_used = "boundary" if rng.random() < 0.5 else "random"
                if add_mode_used == "boundary":
                    center = random_point_from_boundary(gt_union_mask, rng=rng)
                    if center is None:
                        center = random_point_from_mask(gt_union_mask, want_inside=False, rng=rng)
                        add_mode_used = "random_fallback"
                else:
                    center = random_point_from_mask(gt_union_mask, want_inside=False, rng=rng)
        else:
            center = random_point_from_mask(gt_union_mask, want_inside=True, rng=rng)
            add_mode_used = None
        if center is None:
            continue
        cx, cy = center
        n_vertices = rng.randint(min_vertices, max_vertices)
        poly = random_polygon(cx=cx, cy=cy, width=w, height=h, rng=rng, n_vertices=n_vertices, r_min=r_min, r_max=r_max)
        if len(poly) < max(3, min_vertices):
            continue

        poly_mask = polygon_to_mask(width=w, height=h, polygon_xy=poly)
        if do_add:
            corrupted = np.logical_or(gt_union_mask, poly_mask)
            changed = np.logical_and(corrupted, ~gt_union_mask)
            corruption_type = "false_positive_addition"
            target_label = 0
        else:
            corrupted = np.logical_and(gt_union_mask, ~poly_mask)
            changed = np.logical_and(gt_union_mask, ~corrupted)
            corruption_type = "false_negative_deletion"
            target_label = 1
        changed_pixels = int(changed.sum())
        if changed_pixels < min_change_pixels:
            continue
        target_point = mask_center_xy_on_true(changed)
        if target_point is None:
            continue

        return {
            "corrupted_mask": corrupted.astype(bool),
            "corruption_type": corruption_type,
            "target_point": [int(target_point[0]), int(target_point[1])],
            "target_label": int(target_label),
            "changed_pixels": changed_pixels,
            "polygon": [[int(x), int(y)] for x, y in poly],
            "num_vertices": int(len(poly)),
            "seed_point": [int(cx), int(cy)],
            "addition_mode_used": add_mode_used,
            "changed_region_mask": changed.astype(bool),
        }
    return None


def main() -> None:
    args = parse_args()
    args.sample_ratio = min(max(float(args.sample_ratio), 0.0), 1.0)
    rng = random.Random(args.seed)

    split_dir = resolve_split_dir(args.dataset_path, args.split)
    total_rows_all = estimate_total_rows(split_dir=split_dir, split=args.split)
    process_total_rows = total_rows_all if args.max_rows <= 0 else min(total_rows_all, args.max_rows)

    out_dir = args.output_dir
    if out_dir is None:
        dataset_name = infer_dataset_name(args.dataset_path, split_dir)
        out_dir = Path("data/agenticrl/task3_self_correction") / dataset_name
    overlays_root = out_dir / "overlays"
    overlays_fp_dir = overlays_root / "false_positive_addition"
    overlays_fn_dir = overlays_root / "false_negative_deletion"
    masks_dir = out_dir / "masks"
    debug_root = out_dir / "overlays_with_center"
    debug_fp_dir = debug_root / "false_positive_addition"
    debug_fn_dir = debug_root / "false_negative_deletion"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_fp_dir.mkdir(parents=True, exist_ok=True)
    overlays_fn_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if args.save_center_debug:
        debug_fp_dir.mkdir(parents=True, exist_ok=True)
        debug_fn_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "task3_self_correction.jsonl"
    summary_path = out_dir / "summary.json"
    stats = {
        "total_records": 0,
        "false_positive_addition": 0,
        "false_negative_deletion": 0,
        "skipped_by_sampling": 0,
        "skipped_invalid": 0,
        "skipped_failed_corruption": 0,
    }
    rows_out: List[Dict[str, Any]] = []
    row_count = 0

    def write_summary() -> None:
        summary = {
            "dataset_path": str(args.dataset_path),
            "split": args.split,
            "output_dir": str(out_dir),
            "unit": "one image-question pair (union of all object masks)",
            "sampling": {
                "sample_ratio": args.sample_ratio,
                "seed": args.seed,
                "process_total_rows": process_total_rows,
            },
            "construction": {
                "add_prob": args.add_prob,
                "addition_mode": args.addition_mode,
                "variants_per_question": args.variants_per_question,
                "min_change_pixels": args.min_change_pixels,
                "min_vertices": args.min_vertices,
                "max_vertices": args.max_vertices,
                "radius_min_ratio": args.radius_min_ratio,
                "radius_max_ratio": args.radius_max_ratio,
                "target_label_policy": {
                    "false_positive_addition": 0,
                    "false_negative_deletion": 1,
                },
            },
            "progress": {
                "processed_rows": row_count,
                "progress_bar": progress_bar(row_count, process_total_rows),
            },
            "stats": stats,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in iter_dataset_rows(split_dir):
        if args.max_rows > 0 and row_count >= args.max_rows:
            break
        row_count += 1
        if args.log_every > 0 and row_count % args.log_every == 0:
            print(f"[INFO] progress {progress_bar(row_count, process_total_rows)} | kept={stats['total_records']}", flush=True)
        if args.summary_every > 0 and row_count % args.summary_every == 0:
            write_summary()

        if rng.random() > args.sample_ratio:
            stats["skipped_by_sampling"] += 1
            continue

        gt_masks = row["solution_mask"]
        if not isinstance(gt_masks, list):
            stats["skipped_invalid"] += 1
            continue

        try:
            image = decode_image_from_arrow_struct(row["image_struct"])
        except Exception:
            stats["skipped_invalid"] += 1
            continue

        h, w = image.size[1], image.size[0]
        gt_masks_norm: List[np.ndarray] = []
        for one_mask in gt_masks:
            m = np.array(one_mask, dtype=bool)
            if m.ndim != 2:
                continue
            m = resize_mask_to_shape(m, target_h=h, target_w=w)
            if int(m.sum()) == 0:
                continue
            gt_masks_norm.append(m)
        if not gt_masks_norm:
            stats["skipped_invalid"] += 1
            continue

        gt_union = np.zeros((h, w), dtype=bool)
        for one_mask in gt_masks_norm:
            gt_union |= one_mask
        if int(gt_union.sum()) == 0:
            stats["skipped_invalid"] += 1
            continue

        for k in range(args.variants_per_question):
            one = build_one_corruption(
                gt_union_mask=gt_union,
                add_prob=args.add_prob,
                addition_mode=args.addition_mode,
                min_change_pixels=args.min_change_pixels,
                rng=rng,
                min_vertices=args.min_vertices,
                max_vertices=args.max_vertices,
                radius_min_ratio=args.radius_min_ratio,
                radius_max_ratio=args.radius_max_ratio,
            )
            if one is None:
                stats["skipped_failed_corruption"] += 1
                continue

            base_stem = sanitize_stem(str(row["id"]))
            record_id = f"{base_stem}_q{int(row['global_index'])}"
            if args.variants_per_question > 1:
                record_id = f"{record_id}_v{k:02d}"
            file_name = f"{record_id}.png"
            npz_name = f"{record_id}.npz"
            npz_path = masks_dir / npz_name
            group_overlay_dir = overlays_fp_dir if one["corruption_type"] == "false_positive_addition" else overlays_fn_dir
            overlay_path = group_overlay_dir / file_name

            corrupted_mask = one["corrupted_mask"]
            np.savez_compressed(
                npz_path,
                corrupted_union_mask=corrupted_mask.astype(np.uint8),
                gt_union_mask=gt_union.astype(np.uint8),
                changed_region_mask=one["changed_region_mask"].astype(np.uint8),
                gt_mask_stack=np.stack(gt_masks_norm, axis=0).astype(np.uint8),
            )
            overlay = make_mask_overlay(image, corrupted_mask, color=(0, 255, 0), alpha=0.35)
            overlay.save(overlay_path)

            debug_overlay_rel = None
            if args.save_center_debug:
                debug_overlay = draw_center_debug(
                    overlay,
                    center_xy=one["target_point"],
                    label=f"{one['corruption_type']} p={one['target_label']}",
                )
                debug_dir = debug_fp_dir if one["corruption_type"] == "false_positive_addition" else debug_fn_dir
                debug_path = debug_dir / file_name
                debug_overlay.save(debug_path)
                debug_overlay_rel = str(debug_path.relative_to(out_dir))

            rec: Dict[str, Any] = {
                "id": record_id,
                "source_id": row["id"],
                "global_index": row["global_index"],
                "variant_index": k,
                "problem": row["problem"],
                "reasoning_type": row.get("reasoning_type"),
                "img_height": row["img_height"],
                "img_width": row["img_width"],
                "num_objects": len(gt_masks_norm),
                "source_text_field": row.get("source_text_field"),
                "source_mask_field": row.get("source_mask_field"),
                "source_id_field": row.get("source_id_field"),
                "corruption_type": one["corruption_type"],
                "changed_pixels": one["changed_pixels"],
                "num_vertices": one["num_vertices"],
                "target_point_2d": one["target_point"],
                "target_point_label": one["target_label"],
                "target_answer": [{"point_2d": one["target_point"], "point_label": one["target_label"]}],
                "polygon_xy": one["polygon"],
                "seed_point_2d": one["seed_point"],
                "addition_mode_used": one["addition_mode_used"],
                "mask_npz": str(npz_path.relative_to(out_dir)),
                "overlay_image": str(overlay_path.relative_to(out_dir)),
                "overlay_with_center": debug_overlay_rel,
            }
            rows_out.append(rec)
            stats["total_records"] += 1
            stats[one["corruption_type"]] += 1

    with manifest_path.open("w", encoding="utf-8") as fout:
        for rec in rows_out:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_summary()
    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] summary : {summary_path}")
    print(f"[OK] records : {stats['total_records']}")


if __name__ == "__main__":
    main()
