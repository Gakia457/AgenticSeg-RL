import json
import re
import numpy as np
import cv2
from math import isfinite

"""
Task 3: 自我纠正 (Self-Correction / Point Matching)
--------------------------------------------------
奖励构成 (Total Max: 8.0):
1. Format Reward (Fmt) [Max 2.0]:
   - <think>...</think><answer>...</answer> structure (1.0)
   - JSON answer with point_2d and point_label (1.0)
2. Accuracy Reward (Acc) [Max 6.0]:
   - 标签奖励 (L) [Max 2.0]: 类别 (EXTRA/MISSING, 即 FP/FN) 匹配。
   - 点位奖励 (P) [Max 4.0]: 点先匹配一个 GT changed-region polygon，然后只按
     点到 polygon 边界的有符号距离给分。点在 polygon 内或边界上拿满 4 分；
     点在 polygon 外时，离边界越近分越高，离得越远指数衰减。
   - GT 原始 point_2d 只用于保留数据接口，不参与当前点位奖励。
   - 数据里可以保留 polygon_centers / polygon_areas，但默认不参与 reward。
   - label 错时，点位奖励 P 会乘 LABEL_WRONG_GATE 降权。

简洁调试打印阅读指南:
[AgenticRL] Task 3: Score=[总分]/8, Fmt=[x.x]/2, Acc=[L:x.x/2, P:x.x/4] | PolyDist=[d], Hit=[0/1]

- Score = Fmt + L + P。
- Fmt: 格式分。看输出是否有合法 tag / JSON。
- Acc: 准确性分，由 L 和 P 组成。
- L: label 分。2 表示 EXTRA/MISSING 判断对了；0 表示判断错了。
- P: 点位分，满分 4。它看点的位置好不好，不是概率。
  - 点在 polygon 内或边界上：P=4。
  - 点在 polygon 外：按 PolyDist 指数衰减，离边界越近 P 越高。
  - L=0 时：P 会乘 0.7，所以 label 错但点位接近时也只能拿折扣分。
- PolyDist: 点到匹配 polygon 边界的有符号距离。>=0 表示在区域内或边界上；<0 表示在区域外。
- Hit: 严格命中。1 表示 label 对且点在 polygon 内；0 表示还没严格命中。

例子:
Acc=[L:2.0/2, P:4.0/4] | PolyDist=10.0, Hit=1
表示 label 对，点在 polygon 内，点位拿满分。

Acc=[L:2.0/2, P:2.2/4] | PolyDist=-29.0, Hit=0
表示 label 对，但点在 polygon 外，离边界不算太远，所以按边界距离拿到部分 P。

Acc=[L:0.0/2, P:2.8/4] | PolyDist=2.0, Hit=0
表示点在 polygon 内，但 label 错了，所以 P 从 4 分按 0.7 折扣成 2.8 分。
"""

LABEL_REWARD_MAX = 2.0
PROXIMITY_REWARD_MAX = 4.0
REGION_HIT_REWARD_MAX = 2.0
CENTER_REWARD_MAX = 2.0
OUTSIDE_PROXIMITY_MAX = 1.0
LABEL_WRONG_GATE = 0.7
POLYGON_DISTANCE_TAU = 50.0
CENTER_DISTANCE_TAU_MIN = 10.0
# Task3 当前先强压格式：True 表示没有完整 <think> 时，整个 format 分都不给。
TASK3_FORMAT_REQUIRES_THINK = False
# 默认关闭中心点奖励，对齐 wandb/run-20260519_021228-m6h4fbvr 的 polygon-distance reward。
TASK3_USE_POLYGON_CENTER_REWARD = False

def _polygon_proximity_reward(signed_distance):
    if signed_distance is None:
        return 0.0
    if signed_distance >= 0:
        return PROXIMITY_REWARD_MAX
    return float(PROXIMITY_REWARD_MAX * np.exp(signed_distance / POLYGON_DISTANCE_TAU))

def _outside_polygon_reward(signed_distance):
    if signed_distance is None or signed_distance >= 0:
        return 0.0
    return float(OUTSIDE_PROXIMITY_MAX * np.exp(signed_distance / POLYGON_DISTANCE_TAU))

def _center_distance_reward(center_distance, area=None):
    if center_distance is None:
        return 0.0
    try:
        area_f = float(area) if area is not None else 0.0
    except Exception:
        area_f = 0.0
    tau = max(CENTER_DISTANCE_TAU_MIN, float(np.sqrt(max(area_f, 1.0) / np.pi)))
    return float(CENTER_REWARD_MAX * np.exp(-float(center_distance) / tau))

def _infer_grid(values):
    if not values: return 1000
    vmax = 0.0
    for v in values:
        try:
            fv = float(v)
            if isfinite(fv): vmax = max(vmax, abs(fv))
        except: continue
    if vmax <= 1.2: return 1.0
    if vmax <= 999.5: return 999.0
    if vmax <= 1000.5: return 1000.0
    return None

def _rescale_points(points, from_grid, to_grid):
    if from_grid is None or to_grid is None or from_grid == to_grid:
        return np.array(points)
    return np.array(points, dtype=float) * (to_grid / from_grid)

def _normalize_polygons(polygons):
    if not isinstance(polygons, list) or not polygons:
        return []
    if polygons and isinstance(polygons[0], list) and polygons[0] and isinstance(polygons[0][0], (int, float)):
        polygons = [polygons]
    return polygons

def _signed_distance_to_any_polygon(point, polygons):
    polygons = _normalize_polygons(polygons)
    best = None
    for poly in polygons:
        if not isinstance(poly, list) or len(poly) < 3:
            continue
        contour = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
        signed_distance = float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))
        if best is None or signed_distance > best:
            best = signed_distance
    return best

def _match_polygon(point, polygons):
    polygons = _normalize_polygons(polygons)
    best = None
    best_index = None
    for idx, poly in enumerate(polygons):
        if not isinstance(poly, list) or len(poly) < 3:
            continue
        contour = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
        signed_distance = float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))
        if best is None or signed_distance > best:
            best = signed_distance
            best_index = idx
    return best, best_index

def _center_for_index(centers, idx):
    if idx is None or not isinstance(centers, list) or idx >= len(centers):
        return None
    center = centers[idx]
    if isinstance(center, list) and len(center) == 2:
        try:
            return [float(center[0]), float(center[1])]
        except Exception:
            return None
    return None

def _area_for_index(areas, idx):
    if idx is None or not isinstance(areas, list) or idx >= len(areas):
        return None
    return areas[idx]

def analyze_task3_rollout(predict_str, ground_truth, image, is_qwen3=False):
    format_reward = 0.0
    has_think_answer = bool(re.search(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", predict_str, re.DOTALL))
    if has_think_answer:
        format_reward += 1.0

    pred_points = []
    pred_labels = []
    
    # 仍解析答案用于 accuracy 和可视化；format 是否给 JSON 分由 TASK3_FORMAT_REQUIRES_THINK 控制。
    try:
        json_match = re.search(r"<answer>\s*(.*?)\s*</answer>", predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            if isinstance(data, list):
                for item in data:
                    point = item.get("point_2d")
                    try:
                        point = [float(point[0]), float(point[1])] if isinstance(point, list) and len(point) == 2 else None
                    except Exception:
                        point = None
                    has_point = point is not None and all(isfinite(v) for v in point)
                    has_label = "point_label" in item and item["point_label"] in [0, 1, "0", "1"]
                    if has_point and has_label:
                        pred_points.append(point)
                        pred_labels.append(int(item["point_label"]))
                if pred_points and (has_think_answer or not TASK3_FORMAT_REQUIRES_THINK):
                    format_reward += 1.0
    except Exception: pass

    # 1. 坐标对齐
    target_w = 840
    if image is not None:
        try: target_w = image.size[0]
        except: pass
    
    if is_qwen3 and pred_points:
        all_vals = [v for p in pred_points for v in p]
        model_grid = _infer_grid(all_vals)
        pred_points_arr = _rescale_points(pred_points, model_grid, target_w)
    else:
        pred_points_arr = np.array(pred_points) if pred_points else np.array([])

    # 2. 准确率奖励
    accuracy_reward = 0.0
    label_reward = 0.0
    proximity_reward = 0.0
    region_hit = 0
    best_signed_distance = None
    best_center_distance = None
    best_pred_point = None
    best_pred_label = None
    best_gt_item = None
    best_matched_center = None
    best_matched_polygon_index = None
    
    gt_data = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    
    if len(pred_points_arr) > 0 and gt_data:
        gt_labels = np.array([item["point_label"] for item in gt_data])
        gt_polygons = [item.get("polygon") for item in gt_data]
        gt_centers = [item.get("polygon_centers") for item in gt_data]
        gt_areas = [item.get("polygon_areas") for item in gt_data]

        best_total_pair = -1.0
        for i in range(len(pred_points_arr)):
            for j in range(len(gt_labels)):
                label_correct = pred_labels[i] == gt_labels[j]
                curr_label_r = LABEL_REWARD_MAX if label_correct else 0.0
                signed_distance, poly_index = _match_polygon(pred_points_arr[i], gt_polygons[j])
                center = _center_for_index(gt_centers[j], poly_index) if TASK3_USE_POLYGON_CENTER_REWARD else None
                area = _area_for_index(gt_areas[j], poly_index) if TASK3_USE_POLYGON_CENTER_REWARD else None
                center_distance = None
                if center is not None:
                    center_distance = float(np.linalg.norm(np.array(pred_points_arr[i], dtype=float) - np.array(center, dtype=float)))
                if center is not None and signed_distance is not None and signed_distance >= 0:
                    curr_proximity_r = REGION_HIT_REWARD_MAX + _center_distance_reward(center_distance, area)
                elif center is not None:
                    curr_proximity_r = _outside_polygon_reward(signed_distance)
                else:
                    curr_proximity_r = _polygon_proximity_reward(signed_distance)
                gate = 1.0 if label_correct else LABEL_WRONG_GATE
                curr_total = curr_label_r + gate * curr_proximity_r

                if curr_total > best_total_pair:
                    best_total_pair = curr_total
                    label_reward = curr_label_r
                    proximity_reward = gate * curr_proximity_r
                    best_signed_distance = signed_distance
                    best_center_distance = center_distance
                    region_hit = 1 if signed_distance is not None and signed_distance >= 0 and label_correct else 0
                    best_pred_point = [float(pred_points_arr[i][0]), float(pred_points_arr[i][1])]
                    best_pred_label = int(pred_labels[i])
                    best_gt_item = gt_data[j]
                    best_matched_center = center
                    best_matched_polygon_index = poly_index
        accuracy_reward = max(0.0, best_total_pair)

    score = float(format_reward + accuracy_reward)
    return {
        "score": score,
        "format_reward": float(format_reward),
        "accuracy_reward": float(accuracy_reward),
        "label_reward": float(label_reward),
        "proximity_reward": float(proximity_reward),
        "region_hit": int(region_hit),
        "signed_distance": None if best_signed_distance is None else float(best_signed_distance),
        "center_distance": None if best_center_distance is None else float(best_center_distance),
        "has_think_answer": has_think_answer,
        "pred_points": [[float(p[0]), float(p[1])] for p in pred_points_arr.tolist()] if len(pred_points_arr) > 0 else [],
        "pred_labels": [int(v) for v in pred_labels],
        "best_pred_point": best_pred_point,
        "best_pred_label": best_pred_label,
        "matched_center": best_matched_center,
        "matched_polygon_index": best_matched_polygon_index,
        "best_gt": best_gt_item,
        "ground_truth": gt_data,
    }


def compute_task3_reward(predict_str, ground_truth, image, is_qwen3=False):
    analysis = analyze_task3_rollout(predict_str, ground_truth, image, is_qwen3)
    score = analysis["score"]

    # 简洁调试打印
    if analysis["pred_points"] and analysis["ground_truth"]:
        poly_dist = "None" if analysis["signed_distance"] is None else f"{analysis['signed_distance']:.1f}"
        # 中心点版调试打印已关闭；保留旧行便于必要时回看改动:
        # center_dist = "None" if analysis["center_distance"] is None else f"{analysis['center_distance']:.1f}"
        # print(f"[AgenticRL] Task 3: Score={score:.1f}/8, Fmt={analysis['format_reward']:.1f}/2, Acc=[L:{analysis['label_reward']:.1f}/2, P:{analysis['proximity_reward']:.1f}/4] | PolyDist={poly_dist}, CenterDist={center_dist}, Hit={analysis['region_hit']}")
        print(f"[AgenticRL] Task 3: Score={score:.1f}/8, Fmt={analysis['format_reward']:.1f}/2, Acc=[L:{analysis['label_reward']:.1f}/2, P:{analysis['proximity_reward']:.1f}/4] | PolyDist={poly_dist}, Hit={analysis['region_hit']}")
    else:
        print(f"[AgenticRL] Task 3: Score={score:.1f}/8, No points predicted.")

    details = {"format": analysis["format_reward"], "accuracy": analysis["accuracy_reward"]}
    return score, details
