#!/usr/bin/env python3
"""构建 AgenticRL 资产到 HuggingFace Arrow 数据集（HF save_to_disk）。

脚本功能：
1. 读取 Task2/Task3 离线资产目录（manifest jsonl + npz + overlay）。
2. 按统一字段契约组装 HF Dataset，并保存为 arrow 目录。
3. 支持可配置采样、Task2 类别平衡采样、mask 保留策略、去重策略。

输出字段契约：
- id: string
- problem: string
- solution: string
- solution_mask: LargeList[Array2D(bool, 840x840)]
- image: Image（仅使用 overlay 图）
- img_height: int64
- img_width: int64
- reasoning_type: string（无该字段时写空串）

关键约束：
- id 默认取 source_id（可改为 id），并规范化为“图片名去后缀/去 q 后缀”。
- 默认同图去重（保留第一条），避免一图多问重复进入 HF 数据。
- mask 不做 resize；严格模式下要求每张 mask 为 840x840（默认开启）。
- Task2 默认 `--task2-mask-policy need_refine`（仅 need_refine 保留 mask list）。
- Task3 默认 `--task3-mask-policy none`（默认不保留 mask list）。
- Task3 可选 `--task3-region-polygon`（把 changed_region_mask 外轮廓及对应中心点写入 solution）。

采样说明：
- 支持 `--max-rows`（先截处理池）、`--sample-ratio`（再按比例取）、`--random-sample`。
- Task2 平衡采样已实现：`--balance-sample` 开启后，按 good_enough/need_refine 等量采样。
- `--balance-sample` 默认关闭（仅在你显式传入时开启）。
- Task2 平衡采样排序策略：
  - good_enough 默认 `iou_desc`（高 IoU 优先）
  - need_refine 默认 `iou_asc`（低 IoU 优先）
  - 两类都支持 `iou_asc/iou_desc/random`

solution 约定：
- Task2: `{"label":"good_enough|need_refine"}`（json string）
- Task3: 点集 json string，元素形如 `{"point_2d":[x,y], "point_label":0|1}`；
  传入 `--task3-region-polygon` 时额外写入 `polygon`、`polygon_centers`、`polygon_areas`。

命令行配置总览：
- 任务与路径：
  `--task` `--asset-dir` `--output-dir` `--split`
- id 与去重：
  `--id-source` `--drop-duplicate-image/--no-drop-duplicate-image`
- mask 策略：
  `--task2-mask-policy` `--task3-mask-policy` `--strict-shape/--no-strict-shape`
- 采样：
  `--sample-ratio` `--max-rows` `--random-sample/--no-random-sample` `--seed`
- Task2 平衡采样：
  `--balance-sample`
  `--task2-good-enough-sampling`
  `--task2-need-refine-sampling`
- 日志：
  `--log-every`

Task2 最简命令：
python tools/datasets/build_hf_dataset_from_agentic_assets.py \
  --task task2 \
  --asset-dir data/agenticrl/task2_mask_understanding/base_dataset \
  --output-dir data/hf_agentic/task2_base_dataset

Task2 常用命令（含平衡采样）：
python tools/datasets/build_hf_dataset_from_agentic_assets.py \
  --task task2 \
  --asset-dir data/agenticrl/task2_mask_understanding/base_dataset \
  --output-dir data/hf_agentic/task2_base_dataset \
  --split train \
  --task2-mask-policy need_refine \
  --sample-ratio 0.5 \
  --max-rows 0 \
  --balance-sample \
  --task2-good-enough-sampling iou_desc \
  --task2-need-refine-sampling iou_asc \
  --random-sample \
  --seed 42

Task2 全量配置命令： （实际使用）
python tools/datasets/build_hf_dataset_from_agentic_assets.py \
  --task task2 \
  --asset-dir data/agenticrl/task2_mask_understanding/base_dataset \
  --output-dir data/hf_agentic/task2_base_dataset \
  --split train \
  --id-source source_id \
  --drop-duplicate-image \
  --task2-mask-policy all \
  --strict-shape \
  --sample-ratio 0.2 \
  --max-rows 0 \
  --balance-sample \
  --task2-good-enough-sampling iou_desc \
  --task2-need-refine-sampling iou_asc \
  --random-sample \
  --seed 42 \
  --log-every 200
  
python tools/datasets/build_hf_dataset_from_agentic_assets.py \
  --task task2 \
  --asset-dir data/agenticrl/task2_mask_understanding/ReasonSegX \
  --output-dir data/hf_agentic/task2_ReasonSegX \
  --split train \
  --id-source source_id \
  --drop-duplicate-image \
  --task2-mask-policy all \
  --strict-shape \
  --sample-ratio 1.0 \
  --max-rows 0 \
  --seed 42 \
  --log-every 200


Task3 最简命令：
python tools/datasets/build_hf_dataset_from_agentic_assets.py \
  --task task3 \
  --asset-dir data/agenticrl/task3_self_correction/base_dataset \
  --output-dir data/hf_agentic/task3_base_dataset

Task3 常用命令：
python tools/datasets/build_hf_dataset_from_agentic_assets.py \
  --task task3 \
  --asset-dir data/agenticrl/task3_self_correction/base_dataset \
  --output-dir data/hf_agentic/task3_base_dataset \
  --split train \
  --task3-mask-policy none \
  --sample-ratio 0.5 \
  --random-sample \
  --seed 42

Task3 全量配置命令： （实际使用）
  python tools/datasets/build_hf_dataset_from_agentic_assets.py \
    --task task3 \
    --asset-dir data/agenticrl/task3_self_correction/base_dataset \
    --output-dir data/hf_agentic/base_dataset/task3 \
    --split train \
    --id-source source_id \
    --drop-duplicate-image \
    --task3-mask-policy none \
    --strict-shape \
    --sample-ratio 1.0 \
    --max-rows 0 \
    --random-sample \
    --seed 42 \
    --task3-region-polygon \
    --log-every 200

# 2026 0520
# 最大的ratio太大了，掩码被删除的部分太多，剩下一堆零碎的掩码，被模型认为是多出来的噪声
python tools/datasets/build_hf_dataset_from_agentic_assets.py \
  --task task3 \
  --asset-dir data/agenticrl/task3_self_correction/base_dataset_refined \
  --output-dir data/hf_agentic/base_dataset/task3 \
  --split train \
  --id-source source_id \
  --drop-duplicate-image \
  --task3-mask-policy none \
  --task3-region-polygon \
  --strict-shape \
  --sample-ratio 1.0 \
  --seed 42 \
  --log-every 200



python tools/datasets/build_hf_dataset_from_agentic_assets.py \
  --task task3 \
  --asset-dir data/agenticrl/task3_self_correction/base_dataset \
  --output-dir data/hf_agentic/base_dataset/task3 \
  --split train \
  --id-source source_id \
  --drop-duplicate-image \
  --task3-mask-policy all \
  --task3-region-polygon \
  --strict-shape \
  --sample-ratio 1.0 \
  --max-rows 0 \
  --random-sample \
  --seed 42 \
  --log-every 200
  
python tools/datasets/build_hf_dataset_from_agentic_assets.py \
  --task task3 \
  --asset-dir data/agenticrl/task3_self_correction/ReasonSegX \
  --output-dir data/hf_agentic/task3_ReasonSegX \
  --split train \
  --id-source source_id \
  --drop-duplicate-image \
  --task3-mask-policy all \
  --task3-region-polygon \
  --strict-shape \
  --sample-ratio 1.0 \
  --max-rows 0 \
  --seed 42 \
  --log-every 200
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build HF dataset from AgenticRL Task2/Task3 assets")
    p.add_argument("--task", type=str, required=True, choices=["task2", "task3"])
    p.add_argument("--asset-dir", type=Path, required=True, help="Task asset dir containing jsonl/masks/overlays")
    p.add_argument("--output-dir", type=Path, required=True, help="HF save_to_disk output path")
    p.add_argument("--split", type=str, default="train")
    p.add_argument(
        "--id-source",
        type=str,
        default="source_id",
        choices=["source_id", "id"],
        help="Which field to derive final image id from",
    )
    p.add_argument(
        "--drop-duplicate-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep first sample per image id, drop later duplicates",
    )
    p.add_argument(
        "--task2-mask-policy",
        type=str,
        default="need_refine",
        choices=["need_refine", "all", "none"],
        help="Task2 solution_mask keep policy",
    )
    p.add_argument(
        "--task3-mask-policy",
        type=str,
        default="none",
        choices=["all", "none"],
        help="Task3 solution_mask keep policy",
    )
    p.add_argument(
        "--task3-region-polygon",
        action="store_true",
        help="Task3 only: encode changed_region_mask contours and centers into solution polygon fields",
    )
    p.add_argument("--strict-shape", action=argparse.BooleanOptionalAction, default=True, help="Require mask shape 840x840")
    p.add_argument("--sample-ratio", type=float, default=1.0, help="Sampling ratio over processed rows")
    p.add_argument("--balance-sample", action="store_true", help="Task2 only: balance good_enough/need_refine counts")
    p.add_argument(
        "--task2-good-enough-sampling",
        type=str,
        default="iou_desc",
        choices=["iou_asc", "iou_desc", "random"],
        help="Task2 balance mode: how to pick good_enough rows (default iou_desc)",
    )
    p.add_argument(
        "--task2-need-refine-sampling",
        type=str,
        default="iou_asc",
        choices=["iou_asc", "iou_desc", "random"],
        help="Task2 balance mode: how to pick need_refine rows (default iou_asc)",
    )
    p.add_argument(
        "--random-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize row order before sampling/balancing",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=0, help="Rows to process before sampling; 0 means full manifest")
    p.add_argument("--log-every", type=int, default=200)
    return p.parse_args()


def manifest_name(task: str) -> str:
    return "task2_mask_understanding.jsonl" if task == "task2" else "task3_self_correction.jsonl"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def iterate_with_progress(items: Sequence[Dict[str, Any]], task: str) -> Iterable[Dict[str, Any]]:
    try:
        from tqdm import tqdm  # type: ignore

        yield from tqdm(items, total=len(items), desc=f"[{task}] converting")
        return
    except Exception:
        pass
    total = len(items)
    for i, x in enumerate(items, start=1):
        if i == 1 or i % max(1, total // 20) == 0 or i == total:
            print(f"[INFO] progress {i}/{total}")
        yield x


_Q_SUFFIX_RE = re.compile(r"_q\d+(?:_v\d+)?$")


def normalize_image_id(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return "unknown"
    # Remove extension if present.
    s = Path(s).stem
    # Remove task-specific suffix.
    s = _Q_SUFFIX_RE.sub("", s)
    return s or "unknown"


def _to_bool_2d(mask_like: Any) -> Optional[np.ndarray]:
    arr = np.array(mask_like)
    if arr.ndim != 2:
        return None
    return np.array(arr > 0, dtype=bool)


def _extract_mask_list(npz: Any, stack_key: str, union_key: str) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    if stack_key in npz.files:
        arr = np.array(npz[stack_key])
        if arr.ndim == 3:
            for i in range(arr.shape[0]):
                one = _to_bool_2d(arr[i])
                if one is not None:
                    out.append(one)
            if out:
                return out
    if union_key in npz.files:
        one = _to_bool_2d(npz[union_key])
        if one is not None:
            out.append(one)
    return out


def _mask_center_xy_on_true(mask: np.ndarray) -> Optional[List[int]]:
    m = np.array(mask, dtype=bool)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    cx_f = float(xs.mean())
    cy_f = float(ys.mean())
    cx = int(round(cx_f))
    cy = int(round(cy_f))
    if 0 <= cy < m.shape[0] and 0 <= cx < m.shape[1] and bool(m[cy, cx]):
        return [cx, cy]
    d2 = (xs.astype(np.float32) - cx_f) ** 2 + (ys.astype(np.float32) - cy_f) ** 2
    k = int(np.argmin(d2))
    return [int(xs[k]), int(ys[k])]


def mask_to_polygons(mask: np.ndarray) -> tuple[List[List[List[int]]], List[List[int]], List[int]]:
    mask_u8 = (np.array(mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: List[tuple[int, List[List[int]], List[int]]] = []
    for contour in contours:
        if len(contour) < 3:
            x, y, w, h = cv2.boundingRect(contour)
            polygon = [[int(x), int(y)], [int(x + w), int(y)], [int(x + w), int(y + h)], [int(x), int(y + h)]]
        else:
            polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in contour]

        contour_mask = np.zeros_like(mask_u8)
        cv2.drawContours(contour_mask, [contour], -1, color=255, thickness=-1)
        component_mask = np.logical_and(mask_u8 > 0, contour_mask > 0)
        center = _mask_center_xy_on_true(component_mask)
        area = int(component_mask.sum())
        if center is None or area <= 0:
            continue
        regions.append((area, polygon, center))

    regions.sort(key=lambda item: item[0], reverse=True)
    polygons = [polygon for _, polygon, _ in regions]
    centers = [center for _, _, center in regions]
    areas = [area for area, _, _ in regions]
    return polygons, centers, areas


def build_solution_and_masks(
    rec: Dict[str, Any],
    asset_dir: Path,
    task: str,
    task2_mask_policy: str,
    task3_mask_policy: str,
    task3_region_polygon: bool,
    strict_shape: bool,
) -> Optional[Dict[str, Any]]:
    npz_rel = rec.get("mask_npz", None)
    if not isinstance(npz_rel, str):
        return None
    npz_path = asset_dir / npz_rel
    if not npz_path.exists():
        return None

    npz = np.load(npz_path)
    if task == "task2":
        decision = str(rec.get("decision_class") or rec.get("decision") or "")
        if decision == "not_need_refine":
            decision = "good_enough"
        if decision not in {"good_enough", "need_refine"}:
            decision = "need_refine"
        solution = json.dumps({"label": decision}, ensure_ascii=False)

        keep_mask = (
            (task2_mask_policy == "all")
            or (task2_mask_policy == "need_refine" and decision == "need_refine")
        )
        if not keep_mask:
            return {"solution": solution, "solution_mask": []}

        gt_list = _extract_mask_list(npz, stack_key="gt_mask_stack", union_key="gt_union_mask")
        pred_list = _extract_mask_list(npz, stack_key="pred_mask_stack", union_key="pred_union_mask")
        masks = gt_list + pred_list
    else:
        points = rec.get("target_answer", None)
        out_points: List[Dict[str, Any]] = []
        if isinstance(points, list):
            for it in points:
                if not isinstance(it, dict):
                    continue
                p = it.get("point_2d")
                lb = it.get("point_label")
                if isinstance(p, list) and len(p) == 2 and lb is not None:
                    out_points.append({"point_2d": [int(p[0]), int(p[1])], "point_label": int(lb)})
        if not out_points:
            p = rec.get("target_point_2d")
            lb = rec.get("target_point_label")
            if isinstance(p, list) and len(p) == 2 and lb is not None:
                out_points.append({"point_2d": [int(p[0]), int(p[1])], "point_label": int(lb)})
        if task3_region_polygon and out_points and "changed_region_mask" in npz.files:
            polygons, centers, areas = mask_to_polygons(npz["changed_region_mask"])
            if polygons:
                out_points[0]["polygon"] = polygons
                out_points[0]["polygon_centers"] = centers
                out_points[0]["polygon_areas"] = areas
        solution = json.dumps(out_points, ensure_ascii=False)

        if task3_mask_policy == "none":
            return {"solution": solution, "solution_mask": []}
        gt_list = _extract_mask_list(npz, stack_key="gt_mask_stack", union_key="gt_union_mask")
        pred_list = _extract_mask_list(npz, stack_key="corrupted_mask_stack", union_key="corrupted_union_mask")
        masks = gt_list + pred_list

    if strict_shape:
        for m in masks:
            if m.shape != (840, 840):
                return None
    return {
        "solution": solution,
        "solution_mask": [m.astype(bool).tolist() for m in masks],
    }


def normalize_task2_decision(rec: Dict[str, Any]) -> str:
    x = str(rec.get("decision_class") or rec.get("decision") or "").strip()
    if x == "not_need_refine":
        x = "good_enough"
    if x not in {"good_enough", "need_refine"}:
        x = "need_refine"
    return x


def get_task2_iou_score(rec: Dict[str, Any]) -> float:
    for key in ("init_vs_gt_iou_effective", "init_vs_gt_iou_denoised", "init_vs_gt_iou_raw"):
        v = rec.get(key, None)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue
    return 1.0


def order_task2_pool_by_iou(pool: List[Dict[str, Any]], mode: str, rng: random.Random) -> List[Dict[str, Any]]:
    out = list(pool)
    if mode == "random":
        rng.shuffle(out)
        return out
    if mode == "iou_desc":
        out.sort(key=get_task2_iou_score, reverse=True)
        return out
    out.sort(key=get_task2_iou_score)
    return out


def select_rows_by_sampling(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    sample_ratio = min(max(float(args.sample_ratio), 0.0), 1.0)
    process_total = len(rows) if args.max_rows <= 0 else min(len(rows), int(args.max_rows))
    work = list(rows[:process_total])
    rng = random.Random(args.seed)
    if args.random_sample:
        rng.shuffle(work)

    sampling_stats: Dict[str, int] = {
        "process_total_rows": int(process_total),
        "selected_rows": 0,
        "skipped_by_sampling": 0,
        "skipped_by_balance": 0,
    }

    if bool(args.balance_sample) and str(args.task) == "task2":
        target_total = int(process_total * sample_ratio)
        if target_total <= 0:
            sampling_stats["selected_rows"] = 0
            sampling_stats["skipped_by_sampling"] = int(process_total)
            return [], sampling_stats
        target_per_class = target_total // 2
        good_pool = [r for r in work if normalize_task2_decision(r) == "good_enough"]
        need_pool = [r for r in work if normalize_task2_decision(r) == "need_refine"]
        good_pool = order_task2_pool_by_iou(good_pool, str(args.task2_good_enough_sampling), rng)
        need_pool = order_task2_pool_by_iou(need_pool, str(args.task2_need_refine_sampling), rng)

        k = min(target_per_class, len(good_pool), len(need_pool))
        out = good_pool[:k] + need_pool[:k]
        if args.random_sample:
            rng.shuffle(out)
        sampling_stats["skipped_by_balance"] = int(process_total - len(out))
        sampling_stats["selected_rows"] = len(out)
        return out, sampling_stats

    if sample_ratio >= 1.0:
        out = work
    elif args.random_sample:
        out = []
        for rec in work:
            if rng.random() <= sample_ratio:
                out.append(rec)
            else:
                sampling_stats["skipped_by_sampling"] += 1
    else:
        keep_n = int(process_total * sample_ratio)
        out = work[:keep_n]
        sampling_stats["skipped_by_sampling"] = int(process_total - keep_n)

    sampling_stats["selected_rows"] = len(out)
    return out, sampling_stats


def main() -> None:
    args = parse_args()
    manifest_path = args.asset_dir / manifest_name(args.task)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    all_rows = read_jsonl(manifest_path)
    if args.task != "task2" and args.balance_sample:
        print("[WARN] --balance-sample only applies to task2; ignored for this task.", flush=True)
    rows, sampling_stats = select_rows_by_sampling(all_rows, args)

    stats: Dict[str, int] = {
        "input_rows": len(all_rows),
        "process_total_rows": sampling_stats["process_total_rows"],
        "selected_rows": sampling_stats["selected_rows"],
        "skipped_by_sampling": sampling_stats["skipped_by_sampling"],
        "skipped_by_balance": sampling_stats["skipped_by_balance"],
        "kept_rows": 0,
        "skipped_duplicate_image": 0,
        "skipped_missing_overlay": 0,
        "skipped_missing_mask_asset": 0,
        "skipped_invalid_mask_shape": 0,
    }

    buffers: Dict[str, List[Any]] = {
        "id": [],
        "problem": [],
        "solution": [],
        "solution_mask": [],
        "image": [],
        "img_height": [],
        "img_width": [],
        "reasoning_type": [],
    }
    seen_image_ids = set()

    for i, rec in enumerate(iterate_with_progress(rows, args.task), start=1):
        rid_raw = rec.get(args.id_source) if args.id_source in rec else rec.get("source_id", rec.get("id"))
        image_id = normalize_image_id(rid_raw)
        if args.drop_duplicate_image and image_id in seen_image_ids:
            stats["skipped_duplicate_image"] += 1
            continue

        overlay_rel = rec.get("overlay_image", None)
        if not isinstance(overlay_rel, str):
            stats["skipped_missing_overlay"] += 1
            continue
        overlay_path = args.asset_dir / overlay_rel
        if not overlay_path.exists():
            stats["skipped_missing_overlay"] += 1
            continue

        packed = build_solution_and_masks(
            rec=rec,
            asset_dir=args.asset_dir,
            task=args.task,
            task2_mask_policy=args.task2_mask_policy,
            task3_mask_policy=args.task3_mask_policy,
            task3_region_polygon=bool(args.task3_region_polygon),
            strict_shape=bool(args.strict_shape),
        )
        if packed is None:
            # Missing npz or shape mismatch.
            npz_rel = rec.get("mask_npz", None)
            if isinstance(npz_rel, str) and (args.asset_dir / npz_rel).exists():
                stats["skipped_invalid_mask_shape"] += 1
            else:
                stats["skipped_missing_mask_asset"] += 1
            continue

        problem = str(rec.get("problem", ""))
        reasoning_type = rec.get("reasoning_type", "")
        if reasoning_type is None:
            reasoning_type = ""

        buffers["id"].append(image_id)
        buffers["problem"].append(problem)
        buffers["solution"].append(packed["solution"])
        buffers["solution_mask"].append(packed["solution_mask"])
        buffers["image"].append(str(overlay_path))
        buffers["img_height"].append(int(rec.get("img_height") or 0))
        buffers["img_width"].append(int(rec.get("img_width") or 0))
        buffers["reasoning_type"].append(str(reasoning_type))
        seen_image_ids.add(image_id)
        stats["kept_rows"] += 1

        if args.log_every > 0 and i % args.log_every == 0:
            print(f"[INFO] scanned={i} kept={stats['kept_rows']}", flush=True)

    from datasets import Array2D, Dataset, DatasetDict, Features, Image, LargeList, Value

    features = Features(
        {
            "id": Value("string"),
            "problem": Value("string"),
            "solution": Value("string"),
            "solution_mask": LargeList(Array2D(shape=(840, 840), dtype="bool")),
            "image": Image(),
            "img_height": Value("int64"),
            "img_width": Value("int64"),
            "reasoning_type": Value("string"),
        }
    )

    ds = Dataset.from_dict(buffers, features=features)
    dsd = DatasetDict({args.split: ds})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(args.output_dir))

    summary = {
        "task": args.task,
        "asset_dir": str(args.asset_dir),
        "output_dir": str(args.output_dir),
        "split": args.split,
        "id_source": args.id_source,
        "drop_duplicate_image": bool(args.drop_duplicate_image),
        "sample_ratio": float(min(max(float(args.sample_ratio), 0.0), 1.0)),
        "random_sample": bool(args.random_sample),
        "balance_sample": bool(args.balance_sample if args.task == "task2" else False),
        "task2_good_enough_sampling": str(args.task2_good_enough_sampling),
        "task2_need_refine_sampling": str(args.task2_need_refine_sampling),
        "seed": int(args.seed),
        "max_rows": int(args.max_rows),
        "task2_mask_policy": args.task2_mask_policy,
        "task3_mask_policy": args.task3_mask_policy,
        "task3_region_polygon": bool(args.task3_region_polygon),
        "strict_shape": bool(args.strict_shape),
        "stats": stats,
    }
    (args.output_dir / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] saved HF dataset: {args.output_dir}")
    print(f"[OK] kept rows      : {stats['kept_rows']}")


if __name__ == "__main__":
    main()
