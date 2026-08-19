#!/usr/bin/env python3
"""方案C工具：先抽样，再按需渲染（不依赖 datasets 包）。

核心思路：
1) sample 子命令：只生成抽样清单 manifest（不渲染图片）
2) render 子命令：从 manifest 中选一部分按需渲染

这样做的好处：
- 不会一次性把大数据集全部转图
- 可以先小规模检查，再逐步扩大
- 支持按 id 精确渲染

常用命令（可直接复制）：
1. 先抽样 200 条（分层抽样）
   python tools/data_analysis/dataset_sample_and_render.py sample \
     --dataset-path data/base_segmentation_train \
     --split train \
     --total 200 \
     --strategy stratified \
     --manifest outputs/dataset_samples/base_dataset_sample_manifest.jsonl

2. 先渲染前 20 条看看
   python tools/data_analysis/dataset_sample_and_render.py render \
     --dataset-path data/base_segmentation_train \
     --split train \
     --manifest outputs/dataset_samples/base_dataset_sample_manifest.jsonl \
     --output-dir outputs/dataset_samples/rendered \
     --limit 20

3. 按 id 定向渲染
   python tools/data_analysis/dataset_sample_and_render.py render \
     --dataset-path data/base_segmentation_train \
     --split train \
     --manifest outputs/dataset_samples/base_dataset_sample_manifest.jsonl \
     --output-dir outputs/dataset_samples/rendered_selected \
     --ids refcocog_28421,refcoco_12345
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.ipc as ipc
from PIL import Image, ImageDraw


def encode_mask_rle(mask: Any) -> Dict[str, Any]:
    """Encode a 2D bool mask to reversible RLE (row-major, counts start with 0-run)."""
    arr = np.array(mask, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={arr.shape}")
    flat = arr.reshape(-1)
    counts: List[int] = []
    prev = 0
    run = 0
    for v in flat:
        cur = int(v)
        if cur == prev:
            run += 1
        else:
            counts.append(run)
            run = 1
            prev = cur
    counts.append(run)
    return {"size": [int(arr.shape[0]), int(arr.shape[1])], "counts": counts}


def parse_solution(solution_str: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(solution_str)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def resolve_split_dir(dataset_path: Path, split: str) -> Path:
    split_dir = dataset_path / split
    if split_dir.exists():
        return split_dir
    if (dataset_path / "state.json").exists():
        return dataset_path
    raise FileNotFoundError(f"Cannot find split dir: {split_dir} (or state.json under {dataset_path})")


def get_arrow_files(split_dir: Path) -> List[Path]:
    files = sorted(split_dir.glob("data-*.arrow"))
    if not files:
        raise FileNotFoundError(f"No data-*.arrow in {split_dir}")
    return files


def get_shard_lengths(split_dir: Path, arrow_files: List[Path], split: str) -> List[int]:
    info_path = split_dir / "dataset_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            lens = info.get("splits", {}).get(split, {}).get("shard_lengths")
            if isinstance(lens, list) and len(lens) == len(arrow_files):
                return [int(x) for x in lens]
        except Exception:
            pass

    lengths = []
    for fp in arrow_files:
        n = 0
        reader = ipc.open_stream(fp)
        for b in reader:
            n += b.num_rows
        lengths.append(n)
    return lengths


def build_offsets(lengths: List[int]) -> List[int]:
    offsets = [0]
    s = 0
    for x in lengths:
        s += x
        offsets.append(s)
    return offsets


def global_to_shard_row(global_idx: int, offsets: List[int]) -> Tuple[int, int]:
    # offsets: [0, len0, len0+len1, ...]
    shard = bisect_right(offsets, global_idx) - 1
    if shard < 0 or shard >= len(offsets) - 1:
        raise IndexError(f"global index out of range: {global_idx}")
    local = global_idx - offsets[shard]
    return shard, local


def fetch_row_by_global_index(
    split_dir: Path,
    split: str,
    arrow_files: List[Path],
    offsets: List[int],
    global_idx: int,
) -> Dict[str, Any]:
    shard_idx, local_idx = global_to_shard_row(global_idx, offsets)
    fp = arrow_files[shard_idx]
    reader = ipc.open_stream(fp)
    remain = local_idx
    for batch in reader:
        if remain < batch.num_rows:
            i = remain
            return {
                "id": batch.column("id")[i].as_py(),
                "problem": batch.column("problem")[i].as_py(),
                "solution": batch.column("solution")[i].as_py(),
                "solution_mask": batch.column("solution_mask")[i].as_py(),
                "image_struct": batch.column("image")[i].as_py(),
                "img_height": int(batch.column("img_height")[i].as_py()),
                "img_width": int(batch.column("img_width")[i].as_py()),
                "global_index": global_idx,
                "split": split,
                "split_dir": str(split_dir),
            }
        remain -= batch.num_rows
    raise IndexError(f"Cannot locate row {global_idx} in shard {fp}")


def scan_rows_basic(split_dir: Path, show_progress: bool = True) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    files = get_arrow_files(split_dir)
    gidx = 0
    if show_progress:
        print(f"[INFO] scanning shards: {len(files)}", flush=True)
    for si, fp in enumerate(files, start=1):
        if show_progress:
            print(f"[INFO] shard {si}/{len(files)}: {fp.name}", flush=True)
        reader = ipc.open_stream(fp)
        for b in reader:
            ids = b.column("id")
            probs = b.column("problem")
            sols = b.column("solution")
            hs = b.column("img_height")
            ws = b.column("img_width")
            for i in range(b.num_rows):
                solution = parse_solution(sols[i].as_py())
                rows.append(
                    {
                        "id": ids[i].as_py(),
                        "index": gidx,
                        "object_count": len(solution),
                        "problem_preview": (probs[i].as_py()[:140] + "...") if len(probs[i].as_py()) > 140 else probs[i].as_py(),
                        "img_height": int(hs[i].as_py()),
                        "img_width": int(ws[i].as_py()),
                    }
                )
                gidx += 1
    return rows


def choose_indices_stratified(rows: List[Dict[str, Any]], total: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    buckets: Dict[int, List[int]] = {}
    for r in rows:
        buckets.setdefault(int(r["object_count"]), []).append(int(r["index"]))
    for v in buckets.values():
        rng.shuffle(v)

    keys = sorted(buckets.keys())
    chosen: List[int] = []
    while len(chosen) < total and keys:
        next_keys = []
        for k in keys:
            arr = buckets[k]
            if arr:
                chosen.append(arr.pop())
                if len(chosen) >= total:
                    break
            if arr:
                next_keys.append(k)
        keys = next_keys
    return chosen


def choose_indices_random(rows: List[Dict[str, Any]], total: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    all_idx = [int(r["index"]) for r in rows]
    rng.shuffle(all_idx)
    return all_idx[: min(total, len(all_idx))]


def cmd_sample(args: argparse.Namespace) -> None:
    # Step 1) 扫描每条样本的轻量信息（id/index/object_count 等）
    split_dir = resolve_split_dir(args.dataset_path, args.split)
    rows = scan_rows_basic(split_dir, show_progress=True)
    total = min(args.total, len(rows))
    if total <= 0:
        raise ValueError("total must be > 0")

    row_by_idx = {int(r["index"]): r for r in rows}
    # Step 2) 选择抽样策略：分层抽样 / 随机抽样
    if args.strategy == "stratified":
        chosen_idx = choose_indices_stratified(rows, total, args.seed)
    else:
        chosen_idx = choose_indices_random(rows, total, args.seed)

    records = [row_by_idx[i] for i in chosen_idx]
    obj_hist: Dict[int, int] = {}
    for r in records:
        k = int(r["object_count"])
        obj_hist[k] = obj_hist.get(k, 0) + 1

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    # Step 3) 仅保存 manifest，不渲染（方案C的关键）
    with args.manifest.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "dataset_path": str(args.dataset_path),
        "split": args.split,
        "strategy": args.strategy,
        "seed": args.seed,
        "requested_total": args.total,
        "actual_total": len(records),
        "object_count_hist": dict(sorted(obj_hist.items(), key=lambda x: x[0])),
        "manifest": str(args.manifest),
    }
    summary_path = args.manifest.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] sampled {len(records)} rows")
    print(f"[OK] manifest -> {args.manifest}")
    print(f"[OK] summary  -> {summary_path}")


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def sanitize_filename(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in name)


def color_for_idx(i: int) -> np.ndarray:
    palette = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 120, 255],
            [255, 120, 0],
            [180, 0, 255],
            [0, 200, 200],
            [255, 220, 0],
            [255, 0, 120],
        ],
        dtype=np.float32,
    )
    return palette[i % len(palette)]


def overlay_masks(image: Image.Image, masks: List[Any], alpha: float) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = base.shape[:2]
    out = base.copy()
    for i, mask in enumerate(masks):
        m = np.array(mask, dtype=bool)
        if m.shape != (h, w):
            m_img = Image.fromarray((m.astype(np.uint8) * 255), mode="L")
            m_img = m_img.resize((w, h), Image.Resampling.NEAREST)
            m = np.array(m_img, dtype=np.uint8) > 0
        c = color_for_idx(i)
        out[m] = (1 - alpha) * out[m] + alpha * c
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def draw_geom(image: Image.Image, solution: List[Dict[str, Any]]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for i, obj in enumerate(solution):
        bbox = obj.get("bbox_2d")
        point = obj.get("point_2d")
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=2)
            draw.text((x1 + 2, y1 + 2), str(i + 1), fill=(255, 255, 255))
        if isinstance(point, list) and len(point) == 2:
            px, py = point
            r = 3
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(255, 255, 255))
    return image


def cmd_render(args: argparse.Namespace) -> None:
    # Step 1) 读 manifest，得到“待渲染候选集”
    split_dir = resolve_split_dir(args.dataset_path, args.split)
    arrow_files = get_arrow_files(split_dir)
    shard_lengths = get_shard_lengths(split_dir, arrow_files, args.split)
    offsets = build_offsets(shard_lengths)

    records = load_manifest(args.manifest)
    if not records:
        raise ValueError(f"Empty manifest: {args.manifest}")

    selected = records
    # Step 2) 支持按 id 精确筛选，或先只渲染前 N 条
    if args.ids:
        wanted = set([x.strip() for x in args.ids.split(",") if x.strip()])
        selected = [r for r in selected if str(r.get("id")) in wanted]
    if args.limit is not None:
        selected = selected[: max(0, args.limit)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_meta_path = args.output_dir / "rendered_manifest.jsonl"
    print(f"[INFO] rendering {len(selected)} samples", flush=True)

    rendered = 0
    with render_meta_path.open("w", encoding="utf-8") as mf:
        for idx_render, rec in enumerate(selected, start=1):
            global_idx = int(rec["index"])
            row = fetch_row_by_global_index(split_dir, args.split, arrow_files, offsets, global_idx)

            img_struct = row["image_struct"]
            img_bytes = img_struct.get("bytes") if isinstance(img_struct, dict) else None
            if not isinstance(img_bytes, (bytes, bytearray)) or len(img_bytes) == 0:
                continue

            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            solution = parse_solution(row["solution"])
            masks = row["solution_mask"]
            problem = row["problem"]

            mask_areas_preview = []
            mask_shape_preview = None
            solution_mask_rle = []
            if isinstance(masks, list):
                for m in masks:
                    arr = np.array(m, dtype=bool)
                    if mask_shape_preview is None and arr.ndim == 2:
                        mask_shape_preview = [int(arr.shape[0]), int(arr.shape[1])]
                    solution_mask_rle.append(encode_mask_rle(arr))
                for m in masks[:5]:
                    arr = np.array(m, dtype=bool)
                    mask_areas_preview.append(int(arr.sum()))

            # Step 3) 叠加可视化：mask 半透明 + bbox/point
            vis = image.copy()
            if args.overlay_mask:
                vis = overlay_masks(vis, masks, alpha=args.alpha)
            if args.overlay_geom:
                vis = draw_geom(vis, solution)

            rid = str(row["id"])
            stem = sanitize_filename(rid)
            out_img = args.output_dir / f"{stem}.png"
            out_json = args.output_dir / f"{stem}.json"
            vis.save(out_img)

            meta = {
                "id": rid,
                "index": global_idx,
                "problem": problem,
                "problem_preview": (problem[:140] + "...") if len(problem) > 140 else problem,
                "solution": row["solution"],
                "solution_mask_encoding": "rle_row_major_full",
                "solution_mask_rle": solution_mask_rle,
                "output_image": str(out_img),
                "output_meta": str(out_json),
                "object_count": len(solution),
                "mask_count": len(masks) if isinstance(masks, list) else None,
                "mask_shape_preview": mask_shape_preview,
                "mask_areas_preview_top5": mask_areas_preview,
                "img_height_field": int(row["img_height"]),
                "img_width_field": int(row["img_width"]),
                "decoded_size": list(image.size),
                "image_struct": {
                    "path": img_struct.get("path") if isinstance(img_struct, dict) else None,
                    "bytes_len": (len(img_bytes) if isinstance(img_bytes, (bytes, bytearray)) else None),
                    "bytes_base64": (
                        base64.b64encode(img_bytes).decode("ascii")
                        if isinstance(img_bytes, (bytes, bytearray))
                        else None
                    ),
                },
            }
            out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
            rendered += 1
            if idx_render == 1 or idx_render % 20 == 0 or idx_render == len(selected):
                print(f"[INFO] rendered {idx_render}/{len(selected)}", flush=True)

    print(f"[OK] rendered {rendered} samples")
    print(f"[OK] outputs -> {args.output_dir}")
    print(f"[OK] rendered manifest -> {render_meta_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sample then render base_dataset dataset on-demand",
        epilog=(
            "Quick start:\n"
            "  python tools/data_analysis/dataset_sample_and_render.py sample --total 200\n"
            "  python tools/data_analysis/dataset_sample_and_render.py render --limit 20\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="Build sample manifest (no rendering)")
    p_sample.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/base_segmentation_train"),
    )
    p_sample.add_argument("--split", type=str, default="train")
    p_sample.add_argument("--total", type=int, default=200, help="How many rows to sample")
    p_sample.add_argument("--seed", type=int, default=42)
    p_sample.add_argument(
        "--strategy",
        choices=["stratified", "random"],
        default="stratified",
        help="stratified=round-robin by object_count; random=uniform random",
    )
    p_sample.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/dataset_samples/base_dataset_sample_manifest.jsonl"),
    )
    p_sample.set_defaults(func=cmd_sample)

    p_render = sub.add_parser("render", help="Render rows from manifest")
    p_render.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/base_segmentation_train"),
    )
    p_render.add_argument("--split", type=str, default="train")
    p_render.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/dataset_samples/base_dataset_sample_manifest.jsonl"),
    )
    p_render.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dataset_samples/rendered"),
    )
    p_render.add_argument(
        "--ids",
        type=str,
        default="",
        help="Comma-separated ids to render. Empty means render by manifest order.",
    )
    p_render.add_argument("--limit", type=int, default=None, help="Render first N selected rows")
    p_render.add_argument("--overlay-mask", action="store_true", default=True)
    p_render.add_argument("--no-overlay-mask", dest="overlay_mask", action="store_false")
    p_render.add_argument("--overlay-geom", action="store_true", default=True)
    p_render.add_argument("--no-overlay-geom", dest="overlay_geom", action="store_false")
    p_render.add_argument("--alpha", type=float, default=0.40, help="Mask overlay alpha in [0,1]")
    p_render.set_defaults(func=cmd_render)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
