import json
import re

"""
Task 2: 掩码理解 (Mask Understanding)
--------------------------------------
奖励构成 (Total Max: 8.0):
1. Format Reward (Fmt) [Max 3.0]: <think> & <answer> tags.
2. Accuracy Reward (Acc) [Max 5.0]: 5.0 * (1.0 - abs(GT - Pred))^2

调试日志阅读指南:
[AgenticRL] Task 2: Score=[x.x]/8, Fmt=[x.x]/3, Acc=[x.x]/5 | Qual=[预测], GT=[真实], Err=[误差]
"""

def compute_task2_reward(predict_str, ground_truth):
    format_reward = 0.0
    if re.fullmatch(r"<think>.*?</think>\s*<answer>.*?</answer>", predict_str, re.DOTALL):
        format_reward += 3.0 

    accuracy_reward = 0.0
    pred_score = -1.0
    gt_label = 0.0
    error = 0.0
    
    try:
        json_match = re.search(r"<answer>\s*(.*?)\s*</answer>", predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            gt_data = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth

            if "label" in gt_data:
                if gt_data["label"] == "good_enough":
                    gt_label = 1.0
                elif gt_data["label"] == "need_refine":
                    gt_label = 0.0
                else:
                    gt_label = float(gt_data["label"])

            if "quality_score" in data:
                try:
                    pred_score = float(data["quality_score"])
                    pred_score = max(0.0, min(1.0, pred_score))
                    error = abs(gt_label - pred_score)
                    accuracy_reward = 5.0 * ((1.0 - error) ** 2)
                except (ValueError, TypeError):
                    accuracy_reward = 0.0

    except Exception:
        pass

    score = float(format_reward + accuracy_reward)
    details = {"format": format_reward, "accuracy": accuracy_reward}

    # 增强调试打印
    if pred_score >= 0:
        print(f"[AgenticRL] Task 2: Score={score:.1f}/8, Fmt={format_reward:.1f}/3, Acc={accuracy_reward:.1f}/5 | Qual={pred_score:.2f}, GT={gt_label:.1f}, Err={error:.2f}")
    else:
        print(f"[AgenticRL] Task 2: Score={score:.1f}/8, No quality_score found.")

    return score, details
