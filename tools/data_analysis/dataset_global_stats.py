#!/usr/bin/env python3
"""base_dataset 全局统计脚本（含抽样样本属性导出）。

用途：
1) 对全量 Arrow 分片做“全局统计”（对象数分布、尺寸分布、bbox/point 合法性等）
2) 抽样导出一小批样本的属性信息（用于文档举例，不渲染图片）

适配数据格式：
    data/base_segmentation_train
    └── train/data-*.arrow

常用命令（可直接复制）：
1. 用默认路径跑统计
   python tools/data_analysis/dataset_global_stats.py

2. 指定数据路径 + 输出目录
   python tools/data_analysis/dataset_global_stats.py \
     --dataset-path data/base_segmentation_train \
     --split train \
     --output-dir outputs/dataset_stats/base_dataset

3. 多导出一些“样本属性举例”
   python tools/data_analysis/dataset_global_stats.py --sample-count 100

输出文件：
- global_stats.json
- sample_image_attributes.jsonl


我的运行：
    python tools/data_analysis/dataset_global_stats.py \
    --dataset-path data/base_segmentation_train \
    --split train \
    --output-dir outputs/dataset_stats/base_dataset \
    --sample-count 100
"""


from __future__ import annotations

import argparse
import io
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.ipc as ipc
from PIL import Image


@dataclass
class RunningStat:
    count: int = 0
    total: float = 0.0
    min_val: float = float("inf")
    max_val: float = float("-inf")

    def update(self, x: float) -> None:
        self.count += 1
        self.total += x
        if x < self.min_val:
            self.min_val = x
        if x > self.max_val:
            self.max_val = x

    def as_dict(self) -> Dict[str, Any]:
        if self.count == 0:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {
            "count": self.count,
            "mean": self.total / self.count,
            "min": self.min_val,
            "max": self.max_val,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute global stats for base_dataset arrow dataset",
        epilog=(
            "Quick start:\n"
            "  python tools/data_analysis/dataset_global_stats.py\n\n"
            "Custom output:\n"
            "  python tools/data_analysis/dataset_global_stats.py "
            "--output-dir outputs/dataset_stats/base_dataset\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/base_segmentation_train"),
        help="Dataset root path (HF save_to_disk directory).",
    )
    p.add_argument("--split", type=str, default="train", help="Split name")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_stats/base_dataset"),
        help="Output directory for stats JSON and sample attribute JSONL",
    )
    p.add_argument(
        "--sample-count",
        type=int,
        default=30,
        help="Number of sampled rows to export image/sample attributes",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_split_dir(dataset_path: Path, split: str) -> Path:
    split_dir = dataset_path / split
    if split_dir.exists():
        return split_dir
    if (dataset_path / "state.json").exists():
        return dataset_path
    raise FileNotFoundError(f"Cannot find split dir: {split_dir} (or state.json under {dataset_path})")


def _safe_json_load(x: str) -> Optional[Any]:
    try:
        return json.loads(x)
    except Exception:
        return None


def _object_summary(one_obj: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    bbox = one_obj.get("bbox_2d")
    point = one_obj.get("point_2d")
    if isinstance(bbox, list) and len(bbox) == 4:
        out["bbox_2d"] = bbox
    if isinstance(point, list) and len(point) == 2:
        out["point_2d"] = point
    return out


def _image_attr_from_struct(image_struct: Dict[str, Any]) -> Dict[str, Any]:
    img_bytes = image_struct.get("bytes") if isinstance(image_struct, dict) else None
    img_path = image_struct.get("path") if isinstance(image_struct, dict) else None

    attr: Dict[str, Any] = {
        "image_bytes_len": len(img_bytes) if isinstance(img_bytes, (bytes, bytearray)) else None,
        "image_path_field": img_path,
        "decoded_width": None,
        "decoded_height": None,
        "decoded_mode": None,
        "decoded_format": None,
    }
    if isinstance(img_bytes, (bytes, bytearray)) and len(img_bytes) > 0:
        try:
            with Image.open(io.BytesIO(img_bytes)) as im:
                attr["decoded_width"], attr["decoded_height"] = im.size
                attr["decoded_mode"] = im.mode
                attr["decoded_format"] = im.format
        except Exception:
            pass
    return attr


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_dir = resolve_split_dir(args.dataset_path, args.split)
    arrow_files = sorted(split_dir.glob("data-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No data-*.arrow found in {split_dir}")
    print(f"[INFO] scanning split={args.split}, shards={len(arrow_files)}", flush=True)

    dataset_info_path = split_dir / "dataset_info.json"
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8")) if dataset_info_path.exists() else None

    id_set = set()
    duplicate_id_count = 0
    row_count = 0
    bad_solution_json_count = 0
    empty_solution_count = 0
    solution_mask_mismatch_count = 0

    problem_len_stat = RunningStat()
    object_count_stat = RunningStat()
    bbox_width_stat = RunningStat()
    bbox_height_stat = RunningStat()
    bbox_area_stat = RunningStat()

    object_count_hist = Counter()
    solution_mask_count_hist = Counter()
    image_size_hist = Counter()
    object_key_hist = Counter()

    bbox_invalid_count = 0
    bbox_total_count = 0
    point_total_count = 0
    point_inside_bbox_count = 0

    # Reservoir sample: store sample_count rows with rich attributes for examples.
    sample_rows: List[Dict[str, Any]] = []

    for shard_idx, fp in enumerate(arrow_files, start=1):
        print(f"[INFO] shard {shard_idx}/{len(arrow_files)}: {fp.name}", flush=True)
        reader = ipc.open_stream(fp)
        for batch_idx, batch in enumerate(reader):
            ids = batch.column("id")
            probs = batch.column("problem")
            sols = batch.column("solution")
            imgs = batch.column("image")
            hs = batch.column("img_height")
            ws = batch.column("img_width")
            mask_len_arr = batch.column("solution_mask").value_lengths()
            masks = batch.column("solution_mask")

            for i in range(batch.num_rows):
                row_count += 1
                rid = ids[i].as_py()
                if rid in id_set:
                    duplicate_id_count += 1
                else:
                    id_set.add(rid)

                problem = probs[i].as_py()
                solution_str = sols[i].as_py()
                image_h = int(hs[i].as_py())
                image_w = int(ws[i].as_py())
                mask_count = int(mask_len_arr[i].as_py())

                problem_len_stat.update(len(problem))
                image_size_hist[(image_h, image_w)] += 1
                solution_mask_count_hist[mask_count] += 1

                solution = _safe_json_load(solution_str)
                if not isinstance(solution, list):
                    bad_solution_json_count += 1
                    continue
                if len(solution) == 0:
                    empty_solution_count += 1

                obj_count = len(solution)
                object_count_hist[obj_count] += 1
                object_count_stat.update(obj_count)
                if obj_count != mask_count:
                    solution_mask_mismatch_count += 1

                for obj in solution:
                    if not isinstance(obj, dict):
                        continue
                    object_key_hist[tuple(sorted(obj.keys()))] += 1
                    bbox = obj.get("bbox_2d")
                    point = obj.get("point_2d")
                    if isinstance(bbox, list) and len(bbox) == 4:
                        bbox_total_count += 1
                        try:
                            x1, y1, x2, y2 = [float(v) for v in bbox]
                            bw, bh = x2 - x1, y2 - y1
                            bbox_width_stat.update(bw)
                            bbox_height_stat.update(bh)
                            bbox_area_stat.update(max(0.0, bw) * max(0.0, bh))
                            valid = (x2 >= x1) and (y2 >= y1)
                            if not valid:
                                bbox_invalid_count += 1
                            if isinstance(point, list) and len(point) == 2:
                                point_total_count += 1
                                px, py = float(point[0]), float(point[1])
                                if x1 <= px <= x2 and y1 <= py <= y2:
                                    point_inside_bbox_count += 1
                        except Exception:
                            bbox_invalid_count += 1

                # reservoir sampling for example attributes
                if args.sample_count > 0:
                    take = False
                    slot = None
                    if len(sample_rows) < args.sample_count:
                        take = True
                        slot = len(sample_rows)
                    else:
                        j = random.randint(1, row_count)
                        if j <= args.sample_count:
                            take = True
                            slot = j - 1
                    if take and slot is not None:
                        image_struct = imgs[i].as_py()
                        image_attr = _image_attr_from_struct(image_struct)

                        # A tiny mask-area example for illustration.
                        first_mask_area = None
                        try:
                            one_masks = masks[i].as_py()
                            if isinstance(one_masks, list) and len(one_masks) > 0:
                                first = one_masks[0]
                                area = 0
                                for row in first:
                                    area += sum(1 for v in row if v)
                                first_mask_area = area
                        except Exception:
                            pass

                        example = {
                            "global_row_index": row_count - 1,
                            "shard_index": shard_idx,
                            "batch_index": batch_idx,
                            "batch_row_index": i,
                            "id": rid,
                            "problem": problem,
                            "problem_preview": (problem[:120] + "...") if len(problem) > 120 else problem,
                            "solution": solution_str,
                            "img_height_field": image_h,
                            "img_width_field": image_w,
                            "solution_object_count": obj_count,
                            "solution_mask_count": mask_count,
                            "first_object_preview": _object_summary(solution[0]) if len(solution) > 0 else None,
                            "first_mask_area_example": first_mask_area,
                            **image_attr,
                        }
                        if slot == len(sample_rows):
                            sample_rows.append(example)
                        else:
                            sample_rows[slot] = example

    stats = {
        "dataset_path": str(args.dataset_path),
        "split": args.split,
        "resolved_split_dir": str(split_dir),
        "dataset_fields": (
            list(dataset_info.get("features", {}).keys())
            if isinstance(dataset_info, dict)
            else None
        ),
        "num_shards": len(arrow_files),
        "num_rows": row_count,
        "dataset_info_num_examples": (
            dataset_info.get("splits", {}).get(args.split, {}).get("num_examples")
            if isinstance(dataset_info, dict)
            else None
        ),
        "id_unique_count": len(id_set),
        "id_duplicate_count": duplicate_id_count,
        "bad_solution_json_count": bad_solution_json_count,
        "empty_solution_count": empty_solution_count,
        "solution_mask_mismatch_count": solution_mask_mismatch_count,
        "problem_char_len": problem_len_stat.as_dict(),
        "object_count": {
            **object_count_stat.as_dict(),
            "hist_top20": object_count_hist.most_common(20),
        },
        "solution_mask_count_hist_top20": solution_mask_count_hist.most_common(20),
        "image_size_hist_top20": [([h, w], c) for (h, w), c in image_size_hist.most_common(20)],
        "object_key_patterns": [(list(k), c) for k, c in object_key_hist.most_common(20)],
        "bbox": {
            "total": bbox_total_count,
            "invalid_count": bbox_invalid_count,
            "invalid_ratio": (bbox_invalid_count / bbox_total_count) if bbox_total_count > 0 else None,
            "width": bbox_width_stat.as_dict(),
            "height": bbox_height_stat.as_dict(),
            "area": bbox_area_stat.as_dict(),
        },
        "point_in_bbox": {
            "total": point_total_count,
            "inside_count": point_inside_bbox_count,
            "inside_ratio": (point_inside_bbox_count / point_total_count) if point_total_count > 0 else None,
        },
    }

    stats_path = args.output_dir / "global_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    sample_path = args.output_dir / "sample_image_attributes.jsonl"
    with sample_path.open("w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] rows={row_count}, shards={len(arrow_files)}")
    print(f"[OK] global stats -> {stats_path}")
    print(f"[OK] sample attributes ({len(sample_rows)}) -> {sample_path}")


if __name__ == "__main__":
    main()
