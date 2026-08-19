import json
import logging

logger = logging.getLogger(__name__)

def infer_task_type(ground_truth):
    """
    根据 Ground Truth (solution 字段) 的 JSON 结构动态推断任务类型。
    
    Task 1 (Original): 列表，项包含 'bbox_2d'
    Task 2 (AgenticRL): 字典，包含 'label'
    Task 3 (AgenticRL): 列表，项包含 'point_label'
    """
    if ground_truth is None:
        return "task1" # 可能是 Stage 2，或者异常，默认按原流程
    
    try:
        if isinstance(ground_truth, str):
            data = json.loads(ground_truth)
        else:
            data = ground_truth
            
        if isinstance(data, dict):
            if "label" in data:
                return "task2"
        elif isinstance(data, list):
            if len(data) == 0:
                return "task1" # 空列表默认为 task1 (或者根据需求调整)
            
            first_item = data[0]
            if isinstance(first_item, dict):
                if "point_label" in first_item:
                    return "task3"
                if "bbox_2d" in first_item:
                    return "task1"
    except Exception as e:
        logger.warning(f"Error inferring task type: {e}")
        
    return "task1"

def route_and_compute_reward(
    predict_str,
    ground_truth,
    ground_truth_masks,
    image,
    predictor,
    is_qwen3=False,
    config=None
):
    """
    路由分发函数。如果是 Agentic 任务，调用专用奖励函数；
    如果是原有任务，返回 is_agentic=False 让 sam_worker 走原流程。
    """
    task_type = infer_task_type(ground_truth)
    
    # 如果配置明确关闭了某个任务，可以回退（暂留接口）
    if config:
        if task_type == "task2" and not config.get("enable_agentic_task2", True):
            task_type = "task1"
        if task_type == "task3" and not config.get("enable_agentic_task3", True):
            task_type = "task1"

    if task_type == "task2":
        from verl.agentic.reward_task2 import compute_task2_reward
        score, details = compute_task2_reward(predict_str, ground_truth)
        return score, details, True
    
    if task_type == "task3":
        from verl.agentic.reward_task3 import compute_task3_reward
        score, details = compute_task3_reward(predict_str, ground_truth, image, is_qwen3)
        return score, details, True
        
    return None, None, False
