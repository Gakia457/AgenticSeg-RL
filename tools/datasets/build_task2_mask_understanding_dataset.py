#!/usr/bin/env python3
"""【中文说明】构建 AgenticRL Task-2（掩码理解）数据集。

目标：
- 输入：原图 + 问题 + GT 多实例掩码
- 过程：用 GT 提示（bbox/可选点）驱动 SAM 生成初始掩码
- 输出：用于“是否继续细化”判断的数据资产（jsonl + npz + overlay）

核心流程：
1. 读取样本（自动适配 `id|image_id`、`problem|text`、`solution_mask|mask`）。
2. 对每个 GT 实例：
   - bbox 优先取 `solution[*].bbox_2d`，缺失则由 mask 自动提取；
   - 可选点提示：`--use-gt-point` 时优先 `solution[*].point_2d`，否则 mask 中心。
3. 逐实例跑 SAM，得到每实例预测掩码并汇总为 `pred_union_mask`。
4. 汇总 GT 为 `gt_union_mask`，计算两类 IoU：
   - `raw_iou`: 普通 IoU
   - `effective_iou`: 真空环带外 IoU（用于阈值分类）
5. 按 `good-enough-threshold` 分类：
   - `effective_iou >= threshold` → `good_enough / not_need_refine`
   - `effective_iou < threshold`  → `need_refine`

真空环带（边界忽略）：
- 对 GT 并集做 Canny 边界，再膨胀 `vacuum-ring-width` 像素；
- 该环带内像素不参与阈值 IoU 的分子/分母；
- 设计目的：弱化人工边界标注的“外沿毛边误差”。

输出结构：
- `task2_mask_understanding.jsonl`
  - 主键与映射：`id/source_id/global_index`
  - 任务字段：`decision/decision_class/target_action`
  - 指标字段：`init_vs_gt_iou_raw/effective`
  - 兼容字段：`solution`（写入分类与指标），`problem/img_height/img_width`
  - 资源路径：`mask_npz/overlay_image/input_overlay_image`
- `masks/*.npz`
  - `pred_union_mask`, `gt_union_mask`
  - `pred_mask_stack`, `gt_mask_stack`（每实例掩码）
- `overlays/good_enough/*.png`, `overlays/need_refine/*.png`

参数说明（重点）：
- 数据：`--dataset-path --split --output-dir`
- SAM：`--sam-model-cfg --sam-checkpoint --sam-device --use-gt-point`
- 分类：`--good-enough-threshold`
- 真空环带：`--use-vacuum-ring --vacuum-ring-width --vacuum-canny-low --vacuum-canny-high`
- 采样与工程：`--sample-ratio --balance-sample --max-rows --seed --log-every --summary-every`
- 索引重划分：`--enable-overlay-index --keep-all-overlays --force-recompute`

最简配置命令：
python tools/datasets/build_task2_mask_understanding_dataset.py \
  --dataset-path data/base_segmentation_train \
  --split train

常用配置命令：
python tools/datasets/build_task2_mask_understanding_dataset.py \
  --dataset-path data/base_segmentation_train \
  --split train \
  --output-dir data/agenticrl/task2_mask_understanding/base_dataset \
  --sam-device cuda:0 \
  --use-gt-point \
  --good-enough-threshold 0.8 \
  --vacuum-ring-width 10 \
  --seed 42

完整配置命令（常见全量）：
python tools/datasets/build_task2_mask_understanding_dataset.py \
  --dataset-path data/base_segmentation_train \
  --split train \
  --output-dir data/agenticrl/task2_mask_understanding/base_dataset \
  --sam-model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam-checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam-device cuda:0 \
  --use-gt-point \
  --use-vacuum-ring \
  --vacuum-ring-width 10 \
  --vacuum-canny-low 100 \
  --vacuum-canny-high 200 \
  --good-enough-threshold 0.9 \
  --sample-ratio 1.0 \
  --max-rows 0 \
  --seed 42 \
  --log-every 20 \
  --summary-every 100 \
  --enable-overlay-index
  
  
# ReasonSegX_train需要关掉真空环带
python tools/datasets/build_task2_mask_understanding_dataset.py \
  --dataset-path data/ReasonSegX_train \
  --split train \
  --output-dir data/agenticrl/task2_mask_understanding/ReasonSegX \
  --sam-model-cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam-checkpoint sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam-device cuda:0 \
  --use-gt-point \
  --no-use-vacuum-ring \
  --good-enough-threshold 0.9 \
  --sample-ratio 1.0 \
  --max-rows 0 \
  --seed 42 \
  --log-every 20 \
  --summary-every 100 \
  --enable-overlay-index \
  --force-recompute
  
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.ipc as ipc

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.datasets.agenticrl_dataset_common import (  # noqa: E402
    build_canny_vacuum_ring,
    clip_box_xyxy,
    decode_image_from_arrow_struct,
    iou,
    iou_excluding_ignore,
    iter_dataset_rows,
    mask_center_xy,
    mask_to_bbox_xyxy,
    make_mask_overlay,
    parse_solution,
    resize_mask_to_shape,
    resolve_split_dir,
)


class SAMPromptRunner:
    """SAM2 wrapper for box(+optional point) prompted mask generation."""

    def __init__(self, sam_cfg: str, sam_ckpt: str, device: str = "cuda:0"):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.predictor = SAM2ImagePredictor(build_sam2(sam_cfg, sam_ckpt))
        self.predictor.model.to(self.device).eval()
        for p in self.predictor.model.parameters():
            p.requires_grad_(False)

    def predict_one(self, image_rgb: np.ndarray, bbox_xyxy: List[int], point_xy: Optional[List[int]] = None) -> np.ndarray:
        self.predictor.set_image(image_rgb)

        box = np.array(bbox_xyxy, dtype=np.float32)
        point_coords = None
        point_labels = None
        if point_xy is not None and len(point_xy) == 2:
            point_coords = np.array([[float(point_xy[0]), float(point_xy[1])]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)

        if self.device.startswith("cuda"):
            with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=True,
                )
        else:
            with self.torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=True,
                )

        masks_np = np.array(masks)
        scores_np = np.array(scores)
        if masks_np.ndim == 4:
            masks_np = masks_np[0]
            scores_np = scores_np[0]
        best = int(np.argmax(scores_np))
        return np.array(masks_np[best], dtype=bool)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build AgenticRL Task-2 mask-understanding dataset")
    p.add_argument("--dataset-path", type=Path, default=Path("data/base_segmentation_train"))
    p.add_argument("--split", type=str, default="train")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir. Omit to use data/agenticrl/task2_mask_understanding/<dataset_name>/",
    )
    p.add_argument("--sam-model-cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam-checkpoint", type=Path, default=Path("sam2/checkpoints/sam2.1_hiera_large.pt"))
    p.add_argument("--sam-device", type=str, default="cuda:0")
    p.add_argument("--use-gt-point", action="store_true")
    p.add_argument(
        "--use-vacuum-ring",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore boundary vacuum ring when thresholding IoU",
    )
    p.add_argument("--vacuum-ring-width", type=int, default=10, help="Canny edge dilation radius (pixels)")
    p.add_argument("--vacuum-canny-low", type=int, default=100)
    p.add_argument("--vacuum-canny-high", type=int, default=200)
    p.add_argument("--good-enough-threshold", type=float, default=0.8)
    p.add_argument("--sample-ratio", type=float, default=1.0)
    p.add_argument("--balance-sample", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=0, help="0 means full dataset")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--summary-every", type=int, default=100)
    p.add_argument("--enable-overlay-index", action="store_true", help="Generate overlays/overlay_iou_index.json")
    p.add_argument("--keep-all-overlays", action="store_true", help="Also keep overlays/all/*.png as full pool")
    p.add_argument("--force-recompute", action="store_true", help="Force SAM recompute even if overlay index exists")
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
    # If user points to split dir like .../ReasonSegX_train/train, prefer parent dataset name.
    if split_dir.name in {"train", "val", "validation", "test"}:
        return split_dir.parent.name
    # If user points to root like .../ReasonSegX_train, use its basename.
    if dataset_path.name:
        return dataset_path.name
    return split_dir.name


def iou_bucket_name(v: float) -> str:
    vv = min(max(float(v), 0.0), 1.0)
    lo = int(vv * 10) / 10
    hi = min(lo + 0.1, 1.0)
    if lo >= 0.9:
        lo, hi = 0.9, 1.0
    return f"{lo:.1f}-{hi:.1f}"


def safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_iou_summary(rows: List[Dict[str, Any]], metric_key: str) -> Dict[str, Any]:
    vals = [float(r[metric_key]) for r in rows if r.get(metric_key) is not None]
    bins = {"0.9-1.0": 0, "0.8-0.9": 0, "0.7-0.8": 0, "0.6-0.7": 0}
    for v in vals:
        if 0.9 <= v <= 1.0:
            bins["0.9-1.0"] += 1
        elif 0.8 <= v < 0.9:
            bins["0.8-0.9"] += 1
        elif 0.7 <= v < 0.8:
            bins["0.7-0.8"] += 1
        elif 0.6 <= v < 0.7:
            bins["0.6-0.7"] += 1
    if not vals:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "count": 0,
            "distribution": {k: {"count": 0, "ratio": 0.0} for k in bins},
        }
    n = len(vals)
    return {
        "min": float(min(vals)),
        "max": float(max(vals)),
        "mean": float(sum(vals) / n),
        "count": n,
        "distribution": {k: {"count": int(v), "ratio": float(v / n)} for k, v in bins.items()},
    }


def write_overlay_index(
    index_path: Path,
    records: List[Dict[str, Any]],
    use_vacuum_ring: bool,
    vacuum_ring_width: int,
) -> None:
    bins: Dict[str, List[str]] = {f"{i/10:.1f}-{(i+1)/10:.1f}": [] for i in range(9)}
    bins["0.9-1.0"] = []
    for r in records:
        b = iou_bucket_name(float(r["iou_effective"]))
        bins.setdefault(b, []).append(str(r["file_name"]))
    payload = {
        "version": 1,
        "metric": "init_vs_gt_iou_effective",
        "use_vacuum_ring": bool(use_vacuum_ring),
        "vacuum_ring_width": int(vacuum_ring_width),
        "records": records,
        "bins": bins,
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def relabel_from_overlay_index(
    manifest_path: Path,
    index_path: Path,
    overlays_good_dir: Path,
    overlays_need_dir: Path,
    keep_all_overlays: bool,
    overlays_all_dir: Optional[Path],
    threshold: float,
    summary_path: Path,
    args: argparse.Namespace,
) -> None:
    rows = read_jsonl(manifest_path)
    if not rows:
        raise FileNotFoundError(f"Manifest not found or empty: {manifest_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Overlay index not found: {index_path}")

    idx = json.loads(index_path.read_text(encoding="utf-8"))
    idx_map = {str(r["record_id"]): r for r in idx.get("records", [])}

    overlays_root = manifest_path.parent / "overlays"
    tmp_root = overlays_root / ".relabel_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_good_dir = tmp_root / "good_enough"
    tmp_need_dir = tmp_root / "need_refine"
    tmp_good_dir.mkdir(parents=True, exist_ok=True)
    tmp_need_dir.mkdir(parents=True, exist_ok=True)
    tmp_all_dir: Optional[Path] = None
    if keep_all_overlays and overlays_all_dir is not None:
        tmp_all_dir = tmp_root / "all"
        tmp_all_dir.mkdir(parents=True, exist_ok=True)

    out_index_records: List[Dict[str, Any]] = []
    stats = {
        "total_records": 0,
        "good_enough": 0,
        "need_refine": 0,
        "skipped_by_sampling": 0,
        "skipped_by_balance": 0,
        "skipped_invalid": 0,
        "skipped_empty_prompt": 0,
    }

    for rec in rows:
        rid = str(rec.get("id", ""))
        iou_v = rec.get("init_vs_gt_iou_effective", None)
        idx_rec = idx_map.get(rid)
        if idx_rec is not None:
            iou_v = idx_rec.get("iou_effective", iou_v)
        if iou_v is None:
            continue
        iou_v = float(iou_v)

        if iou_v >= threshold:
            decision = "not_need_refine"
            decision_class = "good_enough"
            target_action = None
            group_dir = overlays_good_dir
            stats["good_enough"] += 1
        else:
            decision = "need_refine"
            decision_class = "need_refine"
            target_action = {"type": "continue_refine"}
            group_dir = tmp_need_dir
            stats["need_refine"] += 1
        if decision_class == "good_enough":
            group_dir = tmp_good_dir

        file_name = f"{rid}.png"
        if idx_rec is not None and isinstance(idx_rec.get("file_name"), str):
            file_name = idx_rec["file_name"]

        src_candidates = []
        if isinstance(rec.get("overlay_image"), str):
            src_candidates.append(manifest_path.parent / rec["overlay_image"])
        if isinstance(rec.get("overlay_image_all"), str):
            src_candidates.append(manifest_path.parent / rec["overlay_image_all"])
        if idx_rec is not None and isinstance(idx_rec.get("overlay_image"), str):
            src_candidates.append(manifest_path.parent / idx_rec["overlay_image"])
        if idx_rec is not None and isinstance(idx_rec.get("overlay_image_all"), str):
            src_candidates.append(manifest_path.parent / idx_rec["overlay_image_all"])
        src_candidates.extend([overlays_good_dir / file_name, overlays_need_dir / file_name])
        if overlays_all_dir is not None:
            src_candidates.append(overlays_all_dir / file_name)

        src = next((p for p in src_candidates if p.exists()), None)
        dst = group_dir / file_name
        if src is not None:
            safe_link_or_copy(src, dst)
            rec["overlay_image"] = str(dst.relative_to(manifest_path.parent))
            if keep_all_overlays and tmp_all_dir is not None:
                all_dst = tmp_all_dir / file_name
                safe_link_or_copy(dst, all_dst)
                rec["overlay_image_all"] = str(all_dst.relative_to(manifest_path.parent))
            elif "overlay_image_all" in rec:
                rec.pop("overlay_image_all", None)

        rec["decision"] = decision
        rec["decision_class"] = decision_class
        rec["target_action"] = target_action
        rec["init_vs_gt_iou_effective"] = iou_v
        stats["total_records"] += 1

        out_index_records.append(
            {
                "record_id": rid,
                "file_name": file_name,
                "overlay_image": str(dst.relative_to(manifest_path.parent)),
                "overlay_image_all": (
                    str((tmp_all_dir / file_name).relative_to(manifest_path.parent))
                    if keep_all_overlays and tmp_all_dir is not None
                    else None
                ),
                "iou_effective": iou_v,
                "decision_class": decision_class,
            }
        )

    if overlays_good_dir.exists():
        shutil.rmtree(overlays_good_dir)
    if overlays_need_dir.exists():
        shutil.rmtree(overlays_need_dir)
    shutil.move(str(tmp_good_dir), str(overlays_good_dir))
    shutil.move(str(tmp_need_dir), str(overlays_need_dir))

    if overlays_all_dir is not None and overlays_all_dir.exists():
        shutil.rmtree(overlays_all_dir)
    if keep_all_overlays and tmp_all_dir is not None and overlays_all_dir is not None:
        shutil.move(str(tmp_all_dir), str(overlays_all_dir))
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    write_jsonl(manifest_path, rows)
    write_overlay_index(
        index_path=index_path,
        records=out_index_records,
        use_vacuum_ring=bool(args.use_vacuum_ring),
        vacuum_ring_width=int(args.vacuum_ring_width),
    )

    summary = {
        "mode": "relabel_from_overlay_index",
        "dataset_path": str(args.dataset_path),
        "split": args.split,
        "output_dir": str(manifest_path.parent),
        "label_policy": {
            "good_enough": f"effective_iou >= {threshold}",
            "need_refine": f"effective_iou < {threshold}",
            "use_vacuum_ring": args.use_vacuum_ring,
            "vacuum_ring_width": args.vacuum_ring_width,
            "vacuum_canny_low": args.vacuum_canny_low,
            "vacuum_canny_high": args.vacuum_canny_high,
            "use_gt_point": args.use_gt_point,
        },
        "sampling": {
            "sample_ratio": args.sample_ratio,
            "balance_sample": args.balance_sample,
            "seed": args.seed,
        },
        "unit": "one image-question pair (union of all object masks)",
        "progress": {"processed_rows": len(rows)},
        "stats": stats,
        "iou_stats_effective": build_iou_summary(rows, metric_key="init_vs_gt_iou_effective"),
        "overlay_index": str(index_path.relative_to(manifest_path.parent)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.sample_ratio = min(max(float(args.sample_ratio), 0.0), 1.0)

    split_dir = resolve_split_dir(args.dataset_path, args.split)
    total_rows_all = estimate_total_rows(split_dir=split_dir, split=args.split)
    process_total_rows = total_rows_all if args.max_rows <= 0 else min(total_rows_all, args.max_rows)
    balance_target_total = int(process_total_rows * args.sample_ratio)
    balance_target_per_class = balance_target_total // 2
    rng = np.random.default_rng(args.seed)

    out_dir = args.output_dir
    if out_dir is None:
        dataset_name = infer_dataset_name(args.dataset_path, split_dir)
        out_dir = Path("data/agenticrl/task2_mask_understanding") / dataset_name
    overlays_root = out_dir / "overlays"
    overlays_all_dir = overlays_root / "all" if args.keep_all_overlays else None
    overlays_good_dir = overlays_root / "good_enough"
    overlays_need_dir = overlays_root / "need_refine"
    masks_dir = out_dir / "masks"
    manifest_path = out_dir / "task2_mask_understanding.jsonl"
    summary_path = out_dir / "summary.json"
    overlay_index_path = overlays_root / "overlay_iou_index.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    if overlays_all_dir is not None:
        overlays_all_dir.mkdir(parents=True, exist_ok=True)
    overlays_good_dir.mkdir(parents=True, exist_ok=True)
    overlays_need_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    if args.enable_overlay_index and overlay_index_path.exists() and not args.force_recompute:
        print("[INFO] 发现 overlay index，进入快速重划分模式（不调用 SAM）", flush=True)
        relabel_from_overlay_index(
            manifest_path=manifest_path,
            index_path=overlay_index_path,
            overlays_good_dir=overlays_good_dir,
            overlays_need_dir=overlays_need_dir,
            keep_all_overlays=bool(args.keep_all_overlays),
            overlays_all_dir=overlays_all_dir,
            threshold=float(args.good_enough_threshold),
            summary_path=summary_path,
            args=args,
        )
        print(f"[OK] manifest: {manifest_path}")
        print(f"[OK] summary : {summary_path}")
        print("[OK] mode    : relabel_from_overlay_index")
        return

    runner = SAMPromptRunner(
        sam_cfg=args.sam_model_cfg,
        sam_ckpt=str(args.sam_checkpoint),
        device=args.sam_device,
    )

    stats = {
        "total_records": 0,
        "good_enough": 0,
        "need_refine": 0,
        "skipped_by_sampling": 0,
        "skipped_by_balance": 0,
        "skipped_invalid": 0,
        "skipped_empty_prompt": 0,
    }
    rows_out: List[Dict[str, Any]] = []
    index_records: List[Dict[str, Any]] = []
    row_count = 0

    def write_summary() -> None:
        summary = {
            "mode": "full_recompute",
            "dataset_path": str(args.dataset_path),
            "split": args.split,
            "output_dir": str(out_dir),
            "label_policy": {
                "good_enough": f"effective_iou >= {args.good_enough_threshold}",
                "need_refine": f"effective_iou < {args.good_enough_threshold}",
                "use_vacuum_ring": args.use_vacuum_ring,
                "vacuum_ring_width": args.vacuum_ring_width,
                "vacuum_canny_low": args.vacuum_canny_low,
                "vacuum_canny_high": args.vacuum_canny_high,
                "use_gt_point": args.use_gt_point,
            },
            "sampling": {
                "sample_ratio": args.sample_ratio,
                "balance_sample": args.balance_sample,
                "seed": args.seed,
                "process_total_rows": process_total_rows,
                "balance_target_total": balance_target_total,
                "balance_target_per_class": balance_target_per_class,
            },
            "unit": "one image-question pair (union of all object masks)",
            "progress": {"processed_rows": row_count, "progress_bar": progress_bar(row_count, process_total_rows)},
            "stats": stats,
            "iou_stats_effective": build_iou_summary(rows_out, metric_key="init_vs_gt_iou_effective"),
            "overlay_index": (
                str(overlay_index_path.relative_to(out_dir))
                if args.enable_overlay_index
                else None
            ),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in iter_dataset_rows(split_dir):
        if args.balance_sample and stats["good_enough"] >= balance_target_per_class and stats["need_refine"] >= balance_target_per_class:
            break
        if args.max_rows > 0 and row_count >= args.max_rows:
            break
        row_count += 1

        if args.log_every > 0 and row_count % args.log_every == 0:
            print(
                f"[INFO] progress {progress_bar(row_count, process_total_rows)} | kept={stats['total_records']}",
                flush=True,
            )
        if args.summary_every > 0 and row_count % args.summary_every == 0:
            write_summary()

        if not args.balance_sample and rng.random() > args.sample_ratio:
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

        image_np = np.array(image.convert("RGB"), dtype=np.uint8)
        h, w = image_np.shape[:2]
        solution = parse_solution(row["solution"])

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
        pred_union = np.zeros((h, w), dtype=bool)
        pred_masks_by_object: List[np.ndarray] = [np.zeros((h, w), dtype=bool) for _ in range(len(gt_masks_norm))]
        prompt_items: List[Dict[str, Any]] = []
        valid_prompts = 0

        for obj_idx, gt_mask in enumerate(gt_masks_norm):
            gt_union |= gt_mask
            obj = solution[obj_idx] if obj_idx < len(solution) and isinstance(solution[obj_idx], dict) else None
            bbox_raw = obj.get("bbox_2d") if isinstance(obj, dict) else None
            point_raw = obj.get("point_2d") if isinstance(obj, dict) else None

            if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
                bbox_xyxy = clip_box_xyxy(bbox_raw, width=w, height=h)
            else:
                bbox_xyxy = mask_to_bbox_xyxy(gt_mask)
            if bbox_xyxy is None:
                continue

            point_xy = None
            if args.use_gt_point:
                if isinstance(point_raw, list) and len(point_raw) == 2:
                    point_xy = [int(round(point_raw[0])), int(round(point_raw[1]))]
                else:
                    point_xy = mask_center_xy(gt_mask)

            try:
                pred_mask = runner.predict_one(image_np, bbox_xyxy=bbox_xyxy, point_xy=point_xy)
            except Exception:
                continue

            pred_masks_by_object[obj_idx] = pred_mask
            pred_union |= pred_mask
            valid_prompts += 1
            prompt_items.append({"obj_index": obj_idx, "bbox_2d": bbox_xyxy, "point_2d": point_xy})

        if valid_prompts == 0:
            stats["skipped_empty_prompt"] += 1
            continue

        raw_iou = iou(pred_union, gt_union)
        vacuum_ring = (
            build_canny_vacuum_ring(
                gt_union,
                ring_width=int(args.vacuum_ring_width),
                canny_low=int(args.vacuum_canny_low),
                canny_high=int(args.vacuum_canny_high),
            )
            if args.use_vacuum_ring
            else np.zeros((h, w), dtype=bool)
        )
        effective_iou = iou_excluding_ignore(pred_union, gt_union, vacuum_ring)

        if effective_iou >= args.good_enough_threshold:
            decision = "not_need_refine"
            decision_class = "good_enough"
            target_action = None
        else:
            decision = "need_refine"
            decision_class = "need_refine"
            target_action = {"type": "continue_refine"}

        if args.balance_sample:
            if decision_class == "good_enough" and stats["good_enough"] >= balance_target_per_class:
                stats["skipped_by_balance"] += 1
                continue
            if decision_class == "need_refine" and stats["need_refine"] >= balance_target_per_class:
                stats["skipped_by_balance"] += 1
                continue

        base_stem = sanitize_stem(str(row["id"]))
        record_id = f"{base_stem}_q{int(row['global_index'])}"
        file_name = f"{record_id}.png"
        npz_name = f"{record_id}.npz"
        npz_path = masks_dir / npz_name
        overlay_all_path: Optional[Path] = None
        overlay_group_path = (overlays_good_dir if decision_class == "good_enough" else overlays_need_dir) / file_name

        np.savez_compressed(
            npz_path,
            pred_union_mask=pred_union.astype(np.uint8),
            gt_union_mask=gt_union.astype(np.uint8),
            pred_mask_stack=np.stack(pred_masks_by_object, axis=0).astype(np.uint8),
            gt_mask_stack=np.stack(gt_masks_norm, axis=0).astype(np.uint8),
        )
        overlay = make_mask_overlay(image, pred_union, color=(0, 255, 0), alpha=0.35)
        overlay.save(overlay_group_path)
        if overlays_all_dir is not None:
            overlay_all_path = overlays_all_dir / file_name
            safe_link_or_copy(overlay_group_path, overlay_all_path)
        else:
            overlay_all_path = None

        rec: Dict[str, Any] = {
            "id": record_id,
            "source_id": row["id"],
            "global_index": row["global_index"],
            "problem": row["problem"],
            "reasoning_type": row.get("reasoning_type"),
            "img_height": row["img_height"],
            "img_width": row["img_width"],
            "num_objects": len(gt_masks_norm),
            "sam_prompts": prompt_items,
            "decision": decision,
            "decision_class": decision_class,
            "target_action": target_action,
            "solution": json.dumps(
                {
                    "task2_decision": decision,
                    "task2_decision_class": decision_class,
                    "task2_target_action": target_action,
                    "iou_effective": float(effective_iou),
                    "iou_raw": float(raw_iou),
                },
                ensure_ascii=False,
            ),
            "init_vs_gt_iou_raw": float(raw_iou),
            "init_vs_gt_iou_effective": float(effective_iou),
            "vacuum_ring_pixels": int(vacuum_ring.sum()),
            "vacuum_ring_ratio": float(vacuum_ring.mean()),
            "source_text_field": row.get("source_text_field"),
            "source_mask_field": row.get("source_mask_field"),
            "source_id_field": row.get("source_id_field"),
            "mask_npz": str(npz_path.relative_to(out_dir)),
            "overlay_image": str(overlay_group_path.relative_to(out_dir)),
            "input_overlay_image": str(overlay_group_path.relative_to(out_dir)),
            "solution_mask_keys": {
                "gt_solution_mask_list": "gt_mask_stack",
                "pred_solution_mask_list": "pred_mask_stack",
                "gt_union_mask": "gt_union_mask",
                "pred_union_mask": "pred_union_mask",
            },
        }
        if overlay_all_path is not None:
            rec["overlay_image_all"] = str(overlay_all_path.relative_to(out_dir))
        rows_out.append(rec)
        stats["total_records"] += 1
        if decision_class == "good_enough":
            stats["good_enough"] += 1
        else:
            stats["need_refine"] += 1

        if args.enable_overlay_index:
            index_records.append(
                {
                    "record_id": record_id,
                    "file_name": file_name,
                    "overlay_image": str(overlay_group_path.relative_to(out_dir)),
                    "overlay_image_all": (
                        str(overlay_all_path.relative_to(out_dir)) if overlay_all_path is not None else None
                    ),
                    "iou_effective": float(effective_iou),
                    "decision_class": decision_class,
                }
            )

    write_jsonl(manifest_path, rows_out)
    if args.enable_overlay_index:
        write_overlay_index(
            index_path=overlay_index_path,
            records=index_records,
            use_vacuum_ring=bool(args.use_vacuum_ring),
            vacuum_ring_width=int(args.vacuum_ring_width),
        )
    write_summary()

    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] summary : {summary_path}")
    print(f"[OK] records : {stats['total_records']}")


if __name__ == "__main__":
    main()
