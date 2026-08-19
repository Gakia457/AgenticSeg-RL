#!/usr/bin/env python3
"""【中文说明】可视化 AgenticRL Task2/Task3 数据质量（原图 / GT overlay / 生成overlay 三联图）。

脚本作用：
1) 从 Task2/Task3 manifest 采样样本（支持 max_rows/sample_ratio/random/balance）。
2) 读取 source 数据集原图，并加载当前记录的 GT 掩码与 overlay。
3) 输出三联对比图，便于快速检查“原图-掩码-overlay”是否对齐。
4) 默认把文本问题打在三联图顶部（大字体），可通过命令行关闭。
5) Task2 输出文件名自动追加 IoU（如 `_iou0p7342`），并按 IoU 排名渲染，便于按质量浏览样本。

适配当前目录结构（你现在使用的）：
- Task2: data/agenticrl/task2_mask_understanding/<dataset_name>/
- Task3: data/agenticrl/task3_self_correction/<dataset_name>/
其中 overlays 下按类别分子目录（task2: good_enough/need_refine；task3: false_positive_addition/false_negative_deletion）。

常用命令：
1) Task2（base_dataset）： 
# 实际使用
python tools/datasets/visualize_agenticrl_dataset.py \
  --task task2 \
  --dataset-path data/agenticrl/task2_mask_understanding/base_dataset \
  --sample-count 400 \
  --balance-sample \
  --task2-good-enough-sampling iou_desc \
  --task2-need-refine-sampling iou_asc \
  --output-dir vis \
  --seed 42

2) Task2（ReasonSegX_train）：
python tools/datasets/visualize_agenticrl_dataset.py \
  --task task2 \
  --dataset-path data/agenticrl/task2_mask_understanding/ReasonSegX \
  --source-dataset-path data/ReasonSegX_train \
  --sample-count 240 \
  --output-dir vis/ReasonSegX \
  --seed 42

3) Task3（base_dataset）：
python tools/datasets/visualize_agenticrl_dataset.py \
  --task task3 \
  --dataset-path data/agenticrl/task3_self_correction/base_dataset \
  --sample-count 200 \
  --sample-ratio 0.5 \
  --output-dir vis \
  --seed 42
  
  
python tools/datasets/visualize_agenticrl_dataset.py \
  --task task3 \
  --dataset-path data/agenticrl/task3_self_correction/ReasonSegX \
  --sample-count 200 \
  --sample-ratio 0.5 \
  --output-dir vis/ReasonSegX \
  --seed 42

4) 关闭问题文本显示（默认开启）：
python tools/datasets/visualize_agenticrl_dataset.py \
  --task task2 \
  --dataset-path data/agenticrl/task2_mask_understanding/base_dataset \
  --no-show-problem-text

说明：
- 若不传 --source-dataset-path，会优先从 <dataset-path>/summary.json 自动推断 source dataset/split。
- 若 --dataset-path 传的是 task 根目录（如 data/agenticrl/task2_mask_understanding），脚本会自动尝试定位其中可用的子数据集目录。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.ipc as ipc
from PIL import Image, ImageDraw, ImageFont

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.datasets.agenticrl_dataset_common import (  # noqa: E402
    decode_image_from_arrow_struct,
    make_mask_overlay,
    resolve_split_dir,
)


DEFAULT_TASK2_PATH = Path("./data/agenticrl/task2_mask_understanding/base_dataset")
DEFAULT_TASK3_PATH = Path("./data/agenticrl/task3_self_correction/base_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize task2/task3 dataset quality")
    p.add_argument("--task", type=str, choices=["task2", "task3"], required=True)
    p.add_argument("--dataset-path", type=Path, default=None, help="Task dataset path. If omitted, task default is used.")
    p.add_argument(
        "--source-dataset-path",
        type=Path,
        default=None,
        help="Source dataset path containing original images. If omitted, auto-infer from <dataset-path>/summary.json.",
    )
    p.add_argument("--source-split", type=str, default=None, help="If omitted, auto-infer from summary.json or fallback to train.")
    p.add_argument("--sample-count", type=int, default=100, help="Total sampled records.")
    p.add_argument("--sample-ratio", type=float, default=1.0, help="Sampling ratio over processed rows")
    p.add_argument("--max-rows", type=int, default=0, help="Rows to process before sampling; 0 means full manifest")
    p.add_argument(
        "--balance-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Class-balanced sampling before rendering (default: on)",
    )
    p.add_argument(
        "--task2-good-enough-sampling",
        type=str,
        default="iou_desc",
        choices=["iou_asc", "iou_desc", "random"],
        help="Task2 balance mode: how to pick good_enough rows",
    )
    p.add_argument(
        "--task2-need-refine-sampling",
        type=str,
        default="iou_asc",
        choices=["iou_asc", "iou_desc", "random"],
        help="Task2 balance mode: how to pick need_refine rows",
    )
    p.add_argument(
        "--random-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize row order before sampling/balancing",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("vis"))
    p.add_argument("--overlay-alpha", type=float, default=0.35)
    p.add_argument(
        "--show-problem-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay problem text on top of board (default: on)",
    )
    p.add_argument("--problem-font-size", type=int, default=34, help="Problem text font size on board")
    p.add_argument("--log-every", type=int, default=50, help="Progress print interval.")
    return p.parse_args()


def manifest_name(task: str) -> str:
    return "task2_mask_understanding.jsonl" if task == "task2" else "task3_self_correction.jsonl"


def resolve_task_dataset_dir(dataset_path: Path, task: str) -> Path:
    dataset_path = Path(dataset_path)
    mf = manifest_name(task)
    if (dataset_path / mf).exists():
        return dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if dataset_path.is_dir():
        cands: List[Path] = []
        for child in sorted(dataset_path.iterdir()):
            if child.is_dir() and (child / mf).exists():
                cands.append(child)
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            for preferred in ("base_dataset", "ReasonSegX_train"):
                hit = next((p for p in cands if p.name == preferred), None)
                if hit is not None:
                    return hit
            names = ", ".join(p.name for p in cands[:8])
            raise ValueError(
                f"Multiple dataset dirs found under {dataset_path}. "
                f"Please pass --dataset-path to a concrete one. candidates: {names}"
            )
    raise FileNotFoundError(f"Manifest not found under dataset path: {dataset_path}")


def infer_source_from_summary(dataset_path: Path, source_dataset_path: Optional[Path], source_split: Optional[str]) -> Tuple[Path, str]:
    if source_dataset_path is not None:
        return Path(source_dataset_path), (source_split or "train")

    summary_path = dataset_path / "summary.json"
    if summary_path.exists():
        try:
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            inferred_path = s.get("dataset_path", None)
            inferred_split = s.get("split", None)
            if isinstance(inferred_path, str) and inferred_path:
                return Path(inferred_path), (source_split or inferred_split or "train")
        except Exception:
            pass
    return Path("data/base_segmentation_train"), (source_split or "train")


def estimate_total_rows(split_dir: Path, split: str) -> Optional[int]:
    info_path = split_dir / "dataset_info.json"
    if not info_path.exists():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        n = info.get("splits", {}).get(split, {}).get("num_examples", None)
        if isinstance(n, int) and n > 0:
            return int(n)
    except Exception:
        return None
    return None


def iter_source_image_rows(split_dir: Path):
    global_idx = 0
    for fp in sorted(split_dir.glob("data-*.arrow")):
        reader = ipc.open_stream(fp)
        for batch in reader:
            cols = {name: batch.column(name) for name in batch.schema.names}
            if "image" not in cols:
                raise KeyError(f"Source dataset missing image field, shard={fp}")
            for i in range(batch.num_rows):
                yield global_idx, cols["image"][i].as_py()
                global_idx += 1


def load_images_by_global_index(
    source_dataset_path: Path,
    split: str,
    needed_indices: Sequence[int],
    log_every: int,
) -> Dict[int, Image.Image]:
    split_dir = resolve_split_dir(source_dataset_path, split)
    needed = set(int(x) for x in needed_indices)
    out: Dict[int, Image.Image] = {}
    if not needed:
        return out
    total_rows = estimate_total_rows(split_dir=split_dir, split=split)
    scanned = 0
    print(
        f"[INFO] loading source images: need={len(needed)}"
        + (f", source_total≈{total_rows}" if total_rows is not None else ""),
        flush=True,
    )
    for gidx, image_struct in iter_source_image_rows(split_dir):
        scanned += 1
        if gidx in needed:
            out[gidx] = decode_image_from_arrow_struct(image_struct)
            if log_every > 0 and (len(out) == 1 or len(out) % log_every == 0):
                print(f"[INFO] source loaded {len(out)}/{len(needed)}", flush=True)
            if len(out) == len(needed):
                break
        if log_every > 0 and scanned % (log_every * 20) == 0:
            if total_rows is not None:
                print(f"[INFO] source scan {scanned}/{total_rows}", flush=True)
            else:
                print(f"[INFO] source scan {scanned}", flush=True)
    print(f"[INFO] source images ready: {len(out)}/{len(needed)}", flush=True)
    return out


def load_manifest(dataset_path: Path, task: str) -> List[Dict[str, Any]]:
    fname = manifest_name(task)
    path = dataset_path / fname
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _task2_decision(rec: Dict[str, Any]) -> str:
    x = str(rec.get("decision_class") or rec.get("decision") or "").strip()
    if x == "not_need_refine":
        x = "good_enough"
    if x not in {"good_enough", "need_refine"}:
        x = "need_refine"
    return x


def _task2_iou(rec: Dict[str, Any]) -> float:
    for k in ("init_vs_gt_iou_effective", "init_vs_gt_iou_denoised", "init_vs_gt_iou_raw"):
        v = rec.get(k, None)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue
    return 1.0


def _order_task2_pool_by_iou(pool: List[Dict[str, Any]], mode: str, rng: random.Random) -> List[Dict[str, Any]]:
    out = list(pool)
    if mode == "random":
        rng.shuffle(out)
        return out
    if mode == "iou_desc":
        out.sort(key=_task2_iou, reverse=True)
        return out
    out.sort(key=_task2_iou)
    return out


def preprocess_rows(
    rows: List[Dict[str, Any]],
    sample_ratio: float,
    max_rows: int,
    random_sample: bool,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rng = random.Random(seed)
    ratio = min(max(float(sample_ratio), 0.0), 1.0)
    process_total = len(rows) if max_rows <= 0 else min(len(rows), int(max_rows))
    work = list(rows[:process_total])
    if random_sample:
        rng.shuffle(work)

    stat = {
        "process_total_rows": int(process_total),
        "selected_rows": 0,
        "skipped_by_sampling": 0,
    }
    if ratio >= 1.0:
        picked = work
    elif random_sample:
        picked = []
        for r in work:
            if rng.random() <= ratio:
                picked.append(r)
            else:
                stat["skipped_by_sampling"] += 1
    else:
        keep_n = int(process_total * ratio)
        picked = work[:keep_n]
        stat["skipped_by_sampling"] = int(process_total - keep_n)
    stat["selected_rows"] = len(picked)
    return picked, stat


def pick_balanced(
    rows: List[Dict[str, Any]],
    task: str,
    sample_count: int,
    seed: int,
    task2_good_enough_sampling: str,
    task2_need_refine_sampling: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rng = random.Random(seed)
    if task == "task2":
        pos_name = "good_enough"
        neg_name = "need_refine"
        a = [r for r in rows if _task2_decision(r) == pos_name]
        b = [r for r in rows if _task2_decision(r) == neg_name]
        a = _order_task2_pool_by_iou(a, task2_good_enough_sampling, rng)
        b = _order_task2_pool_by_iou(b, task2_need_refine_sampling, rng)
    else:
        key = "corruption_type"
        pos_name = "false_positive_addition"
        neg_name = "false_negative_deletion"
        a = [r for r in rows if r.get(key) == pos_name]
        b = [r for r in rows if r.get(key) == neg_name]
        rng.shuffle(a)
        rng.shuffle(b)

    half = sample_count // 2
    n_a = min(len(a), half)
    n_b = min(len(b), half)

    selected = a[:n_a] + b[:n_b]
    remain = sample_count - len(selected)
    if remain > 0:
        other = a[n_a:] + b[n_b:]
        rng.shuffle(other)
        selected.extend(other[:remain])
    rng.shuffle(selected)

    stat = {
        pos_name: (sum(1 for r in selected if _task2_decision(r) == pos_name) if task == "task2" else sum(1 for r in selected if r.get(key) == pos_name)),
        neg_name: (sum(1 for r in selected if _task2_decision(r) == neg_name) if task == "task2" else sum(1 for r in selected if r.get(key) == neg_name)),
        "total": len(selected),
    }
    return selected, stat


def add_title(img: Image.Image, title: str) -> Image.Image:
    base = img.convert("RGB")
    w, h = base.size
    top = Image.new("RGB", (w, h + 28), (255, 255, 255))
    top.paste(base, (0, 28))
    draw = ImageDraw.Draw(top)
    draw.text((8, 6), title, fill=(0, 0, 0))
    return top


def draw_point(img: Image.Image, point_xy: Sequence[int], text: str) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x, y = int(point_xy[0]), int(point_xy[1])
    r = 5
    draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=2)
    draw.rectangle([x + 6, y - 10, x + 140, y + 10], fill=(0, 0, 0))
    draw.text((x + 8, y - 8), text, fill=(255, 255, 255))
    return out


def merge_three(a: Image.Image, b: Image.Image, c: Image.Image) -> Image.Image:
    a = a.convert("RGB")
    b = b.convert("RGB").resize(a.size)
    c = c.convert("RGB").resize(a.size)
    W, H = a.size
    out = Image.new("RGB", (W * 3, H), (255, 255, 255))
    out.paste(a, (0, 0))
    out.paste(b, (W, 0))
    out.paste(c, (W * 2, 0))
    return out


def _load_font(size: int) -> ImageFont.ImageFont:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(p, size=max(12, int(size)))
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = str(text).replace("\n", " ").split()
    if not words:
        return [""]
    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        cand = f"{cur} {w}"
        if draw.textlength(cand, font=font) <= max_width:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def add_problem_text(board: Image.Image, problem_text: str, font_size: int = 34) -> Image.Image:
    base = board.convert("RGB")
    w, h = base.size
    pad_x, pad_y = 14, 10
    font = _load_font(font_size)
    probe = Image.new("RGB", (w, 10), (255, 255, 255))
    probe_draw = ImageDraw.Draw(probe)
    lines = _wrap_text(probe_draw, f"Problem: {problem_text}", font=font, max_width=w - 2 * pad_x)
    line_h = max(font_size + 6, 24)
    head_h = pad_y * 2 + line_h * len(lines)
    out = Image.new("RGB", (w, h + head_h), (255, 255, 255))
    out.paste(base, (0, head_h))
    draw = ImageDraw.Draw(out)
    y = pad_y
    for ln in lines:
        draw.text((pad_x, y), ln, fill=(0, 0, 0), font=font)
        y += line_h
    return out


def main() -> None:
    args = parse_args()
    dataset_path_arg = args.dataset_path or (DEFAULT_TASK2_PATH if args.task == "task2" else DEFAULT_TASK3_PATH)
    dataset_path = resolve_task_dataset_dir(Path(dataset_path_arg), task=args.task)
    source_dataset_path, source_split = infer_source_from_summary(
        dataset_path=dataset_path,
        source_dataset_path=args.source_dataset_path,
        source_split=args.source_split,
    )
    print(f"[INFO] task dataset: {dataset_path}", flush=True)

    rows = load_manifest(dataset_path=dataset_path, task=args.task)
    pre_rows, pre_stat = preprocess_rows(
        rows=rows,
        sample_ratio=float(args.sample_ratio),
        max_rows=int(args.max_rows),
        random_sample=bool(args.random_sample),
        seed=int(args.seed),
    )
    if args.balance_sample:
        selected, sampled_stat = pick_balanced(
            pre_rows,
            task=args.task,
            sample_count=args.sample_count,
            seed=args.seed,
            task2_good_enough_sampling=str(args.task2_good_enough_sampling),
            task2_need_refine_sampling=str(args.task2_need_refine_sampling),
        )
    else:
        rng = random.Random(args.seed)
        selected = list(pre_rows)
        rng.shuffle(selected)
        selected = selected[: max(0, int(args.sample_count))]
        sampled_stat = {"total": len(selected)}
    if args.task == "task2":
        selected = sorted(selected, key=_task2_iou, reverse=True)

    print(
        f"[INFO] manifest rows={len(rows)}, pre_selected={len(pre_rows)}, render_selected={len(selected)}",
        flush=True,
    )
    needed_indices = [int(r["global_index"]) for r in selected]
    src_images = load_images_by_global_index(
        source_dataset_path=source_dataset_path,
        split=source_split,
        needed_indices=needed_indices,
        log_every=max(1, int(args.log_every)),
    )

    out_dir = args.output_dir / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.task == "task2":
        group_dirs = {
            "good_enough": out_dir / "good_enough",
            "need_refine": out_dir / "need_refine",
        }
    else:
        group_dirs = {
            "false_positive_addition": out_dir / "false_positive_addition",
            "false_negative_deletion": out_dir / "false_negative_deletion",
        }
    for d in group_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    saved = 0
    saved_by_group = {k: 0 for k in group_dirs.keys()}
    print("[INFO] rendering boards...", flush=True)

    for i, rec in enumerate(selected):
        gidx = int(rec["global_index"])
        src = src_images.get(gidx, None)
        if src is None:
            continue

        npz_rel = rec.get("mask_npz")
        ov_rel = rec.get("overlay_image")
        if not isinstance(npz_rel, str) or not isinstance(ov_rel, str):
            continue
        npz_path = dataset_path / npz_rel
        ov_path = dataset_path / ov_rel
        if not npz_path.exists() or not ov_path.exists():
            continue

        d = np.load(npz_path)
        gt_key = "gt_union_mask"
        if gt_key not in d:
            continue
        gt_mask = np.array(d[gt_key]).astype(bool)
        gt_overlay = make_mask_overlay(src, gt_mask, color=(0, 255, 0), alpha=args.overlay_alpha)

        generated = Image.open(ov_path).convert("RGB")
        if args.task == "task3":
            pt = rec.get("target_point_2d", None)
            label = rec.get("target_point_label", None)
            if isinstance(pt, list) and len(pt) == 2:
                generated = draw_point(generated, pt, text=f"center p={label}")

        p1 = add_title(src, "Original")
        p2 = add_title(gt_overlay, "GT Overlay")
        p3 = add_title(generated, "Generated Overlay")
        board = merge_three(p1, p2, p3)
        if args.show_problem_text:
            q = rec.get("problem", rec.get("text", ""))
            board = add_problem_text(board, str(q), font_size=int(args.problem_font_size))

        rid = str(rec.get("id", f"sample_{i}"))
        rid = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in rid).strip("_.") or f"sample_{i}"
        if args.task == "task2":
            decision = rec.get("decision", "")
            group = "good_enough" if decision == "not_need_refine" else "need_refine"
            iou_v = _task2_iou(rec)
            iou_str = f"{iou_v:.4f}".replace(".", "p")
            file_name = f"{i:04d}_{rid}_iou{iou_str}.png"
        else:
            group = rec.get("corruption_type", "")
            if group not in group_dirs:
                continue
            file_name = f"{i:04d}_{rid}.png"
        board.save(group_dirs[group] / file_name)
        saved += 1
        saved_by_group[group] += 1
        if args.log_every > 0 and (saved == 1 or saved % args.log_every == 0):
            print(f"[INFO] rendered {saved}/{len(selected)}", flush=True)

    summary = {
        "task": args.task,
        "dataset_path": str(dataset_path),
        "source_dataset_path": str(source_dataset_path),
        "source_split": source_split,
        "sample_ratio": float(args.sample_ratio),
        "max_rows": int(args.max_rows),
        "random_sample": bool(args.random_sample),
        "balance_sample": bool(args.balance_sample),
        "task2_good_enough_sampling": str(args.task2_good_enough_sampling),
        "task2_need_refine_sampling": str(args.task2_need_refine_sampling),
        "requested_sample_count": args.sample_count,
        "pre_sampling_result": pre_stat,
        "balanced_sampling_result": sampled_stat,
        "saved_images": saved,
        "saved_by_group": saved_by_group,
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] task={args.task}")
    print(f"[OK] saved={saved}")
    print(f"[OK] output={out_dir}")


if __name__ == "__main__":
    main()
