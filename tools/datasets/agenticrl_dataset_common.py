#!/usr/bin/env python3
"""Common utilities for AgenticRL task2/task3 dataset building scripts."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pyarrow.ipc as ipc
from PIL import Image


def resolve_split_dir(dataset_path: Path, split: str) -> Path:
    """Resolve HF save_to_disk split directory.

    Supports:
    1) DatasetDict layout: <dataset_path>/<split>/data-*.arrow
    2) Single Dataset layout: <dataset_path>/data-*.arrow
    """
    split_dir = dataset_path / split
    if split_dir.exists():
        return split_dir
    if (dataset_path / "state.json").exists():
        return dataset_path
    raise FileNotFoundError(f"Cannot find split dir: {split_dir} (or state.json under {dataset_path})")


def get_arrow_files(split_dir: Path) -> List[Path]:
    files = sorted(split_dir.glob("data-*.arrow"))
    if not files:
        raise FileNotFoundError(f"No data-*.arrow found in {split_dir}")
    return files


def parse_solution(solution_str: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(solution_str)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


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


def _normalize_mask_list(mask_value: Any) -> List[np.ndarray]:
    arr = np.array(mask_value)
    if arr.ndim == 2:
        m = _to_bool_2d(arr)
        return [m] if m is not None else []
    if arr.ndim == 3 and (arr.dtype == bool or np.issubdtype(arr.dtype, np.number)):
        return [np.array(arr[i] > 0, dtype=bool) for i in range(arr.shape[0])]

    if isinstance(mask_value, list):
        out = []
        for x in mask_value:
            m = _to_bool_2d(x)
            if m is not None:
                out.append(m)
        return out
    return []


def decode_image_from_arrow_struct(image_struct: Any) -> Image.Image:
    """Decode HF Image feature payload from Arrow row."""
    if not isinstance(image_struct, dict):
        raise ValueError(f"image struct must be dict, got {type(image_struct)}")
    img_bytes = image_struct.get("bytes", None)
    img_path = image_struct.get("path", None)
    if isinstance(img_bytes, (bytes, bytearray)) and len(img_bytes) > 0:
        with Image.open(io.BytesIO(img_bytes)) as im:
            return im.convert("RGB")
    if isinstance(img_path, str) and img_path:
        with Image.open(img_path) as im:
            return im.convert("RGB")
    raise ValueError("image struct has neither valid bytes nor path")


def iter_dataset_rows(split_dir: Path) -> Iterator[Dict[str, Any]]:
    """Yield raw rows with stable fields used by task2/task3 builders."""
    global_idx = 0
    for fp in get_arrow_files(split_dir):
        reader = ipc.open_stream(fp)
        for batch in reader:
            cols = {name: batch.column(name) for name in batch.schema.names}
            schema_names = set(batch.schema.names)
            id_key = "id" if "id" in schema_names else ("image_id" if "image_id" in schema_names else None)
            text_key = "problem" if "problem" in schema_names else ("text" if "text" in schema_names else None)
            solution_key = "solution" if "solution" in schema_names else None
            mask_key = "solution_mask" if "solution_mask" in schema_names else ("mask" if "mask" in schema_names else None)
            image_key = "image" if "image" in schema_names else None
            for i in range(batch.num_rows):
                if image_key is None:
                    raise KeyError(f"Dataset row missing required image field, schema={sorted(schema_names)}")

                raw_mask_value = cols[mask_key][i].as_py() if mask_key is not None else []
                normalized_masks = _normalize_mask_list(raw_mask_value)
                row = {
                    "global_index": global_idx,
                    "id": (str(cols[id_key][i].as_py()) if id_key is not None else f"row_{global_idx}"),
                    "problem": (cols[text_key][i].as_py() if text_key is not None else ""),
                    "solution": (cols[solution_key][i].as_py() if solution_key is not None else "[]"),
                    "solution_mask": normalized_masks,
                    "image_struct": cols[image_key][i].as_py(),
                    "img_height": int(cols["img_height"][i].as_py()) if "img_height" in cols else None,
                    "img_width": int(cols["img_width"][i].as_py()) if "img_width" in cols else None,
                    "reasoning_type": cols["reasoning_type"][i].as_py() if "reasoning_type" in cols else None,
                    "source_text_field": text_key,
                    "source_mask_field": mask_key,
                    "source_id_field": id_key,
                }
                yield row
                global_idx += 1


def binary_erosion_3x3(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Pure NumPy 3x3 binary erosion (8-neighborhood)."""
    out = np.array(mask, dtype=bool)
    if iterations <= 0:
        return out
    for _ in range(iterations):
        h, w = out.shape
        p = np.pad(out, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        neigh = [
            p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
        out = np.logical_and.reduce(neigh)
    return out


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.array(mask_a, dtype=bool)
    b = np.array(mask_b, dtype=bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def denoised_iou_with_mode(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    erosion_iter: int = 1,
    mode: str = "symmetric",
) -> float:
    m = str(mode).strip().lower()
    pred = np.array(pred_mask, dtype=bool)
    gt = np.array(gt_mask, dtype=bool)
    if m == "none" or erosion_iter <= 0:
        return iou(pred, gt)
    if m == "gt_only":
        return iou(pred, binary_erosion_3x3(gt, iterations=erosion_iter))
    if m == "pred_only":
        return iou(binary_erosion_3x3(pred, iterations=erosion_iter), gt)
    # default: symmetric
    return iou(
        binary_erosion_3x3(pred, iterations=erosion_iter),
        binary_erosion_3x3(gt, iterations=erosion_iter),
    )


def denoised_iou(mask_a: np.ndarray, mask_b: np.ndarray, erosion_iter: int = 1) -> float:
    return denoised_iou_with_mode(mask_a, mask_b, erosion_iter=erosion_iter, mode="symmetric")


def make_mask_overlay(image: Image.Image, mask: np.ndarray, color=(255, 255, 0), alpha: float = 0.35) -> Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.float32)
    m = np.array(mask, dtype=bool)
    out = base.copy()
    c = np.array(color, dtype=np.float32)
    out[m] = out[m] * (1.0 - alpha) + c * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def clip_box_xyxy(bbox: List[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = min(max(0, x1), width - 1)
    x2 = min(max(0, x2), width - 1)
    y1 = min(max(0, y1), height - 1)
    y2 = min(max(0, y2), height - 1)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def bbox_center(bbox_xyxy: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 + x2) // 2, (y1 + y2) // 2


def resize_mask_to_shape(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    m = np.array(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape={m.shape}")
    if m.shape == (target_h, target_w):
        return m
    m_img = Image.fromarray((m.astype(np.uint8) * 255), mode="L")
    m_img = m_img.resize((target_w, target_h), Image.Resampling.NEAREST)
    return np.array(m_img, dtype=np.uint8) > 0


def mask_to_bbox_xyxy(mask: np.ndarray) -> Optional[List[int]]:
    m = np.array(mask, dtype=bool)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())
    return [x1, y1, x2, y2]


def mask_center_xy(mask: np.ndarray) -> Optional[List[int]]:
    m = np.array(mask, dtype=bool)
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    cx = int(round(float(xs.mean())))
    cy = int(round(float(ys.mean())))
    return [cx, cy]


def mask_center_xy_on_true(mask: np.ndarray) -> Optional[List[int]]:
    """Center point snapped to nearest True pixel (stable supervision point)."""
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


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """1-pixel-ish boundary of a binary mask."""
    m = (np.array(mask, dtype=np.uint8) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(m, kernel, iterations=1)
    boundary = np.logical_and(m > 0, eroded == 0)
    return np.array(boundary, dtype=bool)


def build_canny_vacuum_ring(
    gt_mask: np.ndarray,
    ring_width: int = 10,
    canny_low: int = 100,
    canny_high: int = 200,
) -> np.ndarray:
    """Build ignore ring around GT boundary via Canny + dilation."""
    m = (np.array(gt_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if int(m.sum()) == 0 or ring_width <= 0:
        return np.zeros_like(m, dtype=bool)
    edges = cv2.Canny(m, threshold1=int(canny_low), threshold2=int(canny_high))
    if int(edges.sum()) == 0:
        return np.zeros_like(m, dtype=bool)
    k = max(1, int(ring_width))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    ring = cv2.dilate(edges, kernel, iterations=1) > 0
    return np.array(ring, dtype=bool)


def iou_excluding_ignore(pred_mask: np.ndarray, gt_mask: np.ndarray, ignore_mask: np.ndarray) -> float:
    """IoU on valid area where ignore_mask=False."""
    pred = np.array(pred_mask, dtype=bool)
    gt = np.array(gt_mask, dtype=bool)
    ign = np.array(ignore_mask, dtype=bool)
    valid = ~ign
    inter = np.logical_and(np.logical_and(pred, gt), valid).sum()
    union = np.logical_and(np.logical_or(pred, gt), valid).sum()
    if union == 0:
        return 1.0
    return float(inter / union)
