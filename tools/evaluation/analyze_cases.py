import argparse
import glob
import json
import os
import re
import shutil
from typing import Any, Dict, List

# python ./tools/evaluation/analyze_cases.py --output_dir ./outputs/reasonseg_eval_results/pretrained_models/reasoning-model/ReasonSegX_test --top_k 50

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Directory containing output_*.json")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K per category for positive/negative case export")
    return parser.parse_args()


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _calc_iou(item: Dict[str, Any]) -> float:
    inter = _safe_float(item.get("intersection", 0.0))
    union = _safe_float(item.get("union", 0.0))
    return (inter / union) if union > 0 else 1.0


def _find_vis_compare(output_dir: str, item: Dict[str, Any]) -> str:
    rel = item.get("vis_relpath", None)
    if isinstance(rel, str) and rel:
        compare_path = os.path.join(output_dir, rel, "compare.png")
        if os.path.exists(compare_path):
            return os.path.relpath(compare_path, output_dir)
    ann_id = str(item.get("ann_id", ""))
    if ann_id:
        cands = glob.glob(os.path.join(output_dir, "visualization", f"*{ann_id}*", "compare.png"))
        if cands:
            return os.path.relpath(sorted(cands)[0], output_dir)
    return ""


def _to_brief_row(item: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    row = dict(item)
    row["iou"] = _calc_iou(item)
    row["vis_compare"] = _find_vis_compare(output_dir, item)
    return row


def _write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sanitize_component(v: Any) -> str:
    s = str(v) if v is not None else "uncategorized"
    s = s.strip()
    if not s:
        s = "uncategorized"
    out = []
    for ch in s:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def _group_by_reasoning_type(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        rt = r.get("reasoning_type", None)
        key = str(rt) if rt not in (None, "") else "uncategorized"
        groups.setdefault(key, []).append(r)
    return groups


def _format_think_for_reading(think: Any) -> Any:
    if not isinstance(think, str):
        return think
    lines = _think_to_lines(think)
    return "\n".join(lines)


def _think_to_lines(think: Any) -> List[str]:
    if not isinstance(think, str):
        return []
    s = think.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []

    # Step 1: split by sentence-ending punctuation.
    sentence_like = [x.strip() for x in re.findall(r"[^。.!?！？]+[。.!?！？]?", s) if x.strip()]
    lines: List[str] = []

    # Step 2: if a sentence is too long, split by comma.
    for seg in sentence_like:
        if len(seg) <= 120:
            lines.append(seg)
            continue
        comma_parts = [p.strip() for p in re.findall(r"[^，,]+[，,]?", seg) if p.strip()]
        if len(comma_parts) > 1:
            lines.extend(comma_parts)
        else:
            lines.append(seg)

    return lines


def _prepare_export_row(row: Dict[str, Any], split_name: str, rank: int) -> Dict[str, Any]:
    rr = dict(row)
    rr["category"] = split_name
    rr["rank"] = rank
    think_readable = _format_think_for_reading(rr.get("think", ""))
    rr["think"] = think_readable
    rr["think_lines"] = _think_to_lines(think_readable)
    return rr


def _export_case_images(
    output_dir: str,
    export_dir: str,
    split_name: str,
    rows: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    os.makedirs(export_dir, exist_ok=True)
    selected = rows[:top_k]
    exported = []
    for i, row in enumerate(selected, start=1):
        ann_id = _sanitize_component(row.get("ann_id", "na"))
        image_id = _sanitize_component(row.get("image_id", "na"))
        iou = _safe_float(row.get("iou", 0.0), 0.0)
        stem = f"{i:03d}_iou{iou:.4f}_ann{ann_id}_img{image_id}"
        dst_json = os.path.join(export_dir, f"{stem}.json")
        export_row = _prepare_export_row(row, split_name=split_name, rank=i)
        with open(dst_json, "w", encoding="utf-8") as f:
            json.dump(export_row, f, ensure_ascii=False, indent=2)

        vis_rel = row.get("vis_compare", "")
        has_image = False
        export_img_rel = ""
        if isinstance(vis_rel, str) and vis_rel:
            src_img = os.path.join(output_dir, vis_rel)
            if os.path.exists(src_img):
                ext = os.path.splitext(src_img)[1] or ".png"
                dst_img = os.path.join(export_dir, f"{stem}{ext}")
                shutil.copy2(src_img, dst_img)
                has_image = True
                export_img_rel = os.path.relpath(dst_img, output_dir)

        rr = dict(export_row)
        rr["export_json"] = os.path.relpath(dst_json, output_dir)
        rr["has_compare_image"] = has_image
        if has_image:
            rr["export_compare_image"] = export_img_rel
        exported.append(rr)
    return exported


def main():
    args = parse_args()
    output_files = sorted(glob.glob(os.path.join(args.output_dir, "output_*.json")))
    if not output_files:
        raise FileNotFoundError(f"No output_*.json found in {args.output_dir}")

    items: List[Dict[str, Any]] = []
    for fp in output_files:
        with open(fp, "r", encoding="utf-8") as f:
            part = json.load(f)
            if isinstance(part, list):
                items.extend(part)

    rows = [_to_brief_row(x, args.output_dir) for x in items]
    if not rows:
        raise RuntimeError("No evaluation items loaded.")

    top_k = max(1, int(args.top_k))
    groups = _group_by_reasoning_type(rows)

    no_object_errors = []
    for r in rows:
        gt_area = int(_safe_float(r.get("gt_mask_area", -1), -1))
        pred_area = int(_safe_float(r.get("pred_mask_area", -1), -1))
        if gt_area > 0 and pred_area == 0:
            no_object_errors.append(r)

    out_dir = os.path.join(args.output_dir, "case_analysis")
    os.makedirs(out_dir, exist_ok=True)

    _write_jsonl(os.path.join(out_dir, "no_object_errors.jsonl"), no_object_errors)

    export_root = os.path.join(out_dir, "visual_cases")
    export_stats = {}
    for rt, rt_rows in sorted(groups.items(), key=lambda kv: kv[0]):
        safe_rt = _sanitize_component(rt)
        rt_rows_desc = sorted(rt_rows, key=lambda x: x["iou"], reverse=True)
        rt_rows_asc = sorted(rt_rows, key=lambda x: x["iou"])
        pos_dir = os.path.join(export_root, safe_rt, "positive_topk")
        neg_dir = os.path.join(export_root, safe_rt, "negative_topk")
        exported_pos = _export_case_images(args.output_dir, pos_dir, "positive", rt_rows_desc, top_k)
        exported_neg = _export_case_images(args.output_dir, neg_dir, "negative", rt_rows_asc, top_k)
        export_stats[rt] = {
            "num_samples": len(rt_rows),
            "exported_positive": len(exported_pos),
            "exported_negative": len(exported_neg),
            "dir": os.path.relpath(os.path.join(export_root, safe_rt), args.output_dir),
        }

    summary = {
        "output_dir": args.output_dir,
        "num_items": len(rows),
        "top_k": top_k,
        "avg_iou": sum(x["iou"] for x in rows) / max(1, len(rows)),
        "num_no_object_errors": len(no_object_errors),
        "num_reasoning_types": len(groups),
        "by_reasoning_type": export_stats,
        "artifacts": {
            "no_object_errors": "case_analysis/no_object_errors.jsonl",
            "visual_cases_root": "case_analysis/visual_cases",
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
