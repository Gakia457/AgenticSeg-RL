import json
import sys
import os

# 将项目根目录加入 path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from verl.agentic.router import infer_task_type, route_and_compute_reward

def test_inference():
    print("Testing Task Inference...")
    
    # Task 1
    gt1 = json.dumps([{"label": "obj", "bbox_2d": [0,0,10,10], "point_2d": [5,5]}])
    assert infer_task_type(gt1) == "task1"
    
    # Task 2
    gt2 = json.dumps({"label": "good_enough"})
    assert infer_task_type(gt2) == "task2"
    
    # Task 3
    gt3 = json.dumps([{"point_2d": [100, 200], "point_label": 1}])
    assert infer_task_type(gt3) == "task3"
    
    print("Task Inference: OK")

def test_rewards():
    print("Testing Task Rewards...")
    
    # Task 2
    pred2 = "<think> reasoning </think> <answer>{\"label\": \"good_enough\"}</answer>"
    gt2 = json.dumps({"label": "good_enough"})
    score2, details2, is_agentic2 = route_and_compute_reward(pred2, gt2, None, None, None)
    print(f"Task 2 Score: {score2}, Details: {details2}, Is Agentic: {is_agentic2}")
    assert is_agentic2 is True
    assert score2 == 8.0 # 3.0 format + 5.0 acc
    
    # Task 3 (Qwen2.5 mode for simplicity in test)
    pred3 = "<think> reasoning </think> <answer>[{\"point_2d\": [100, 200], \"point_label\": 1}]</answer>"
    gt3 = json.dumps([{"point_2d": [100, 200], "point_label": 1}])
    score3, details3, is_agentic3 = route_and_compute_reward(pred3, gt3, None, None, None, is_qwen3=False)
    print(f"Task 3 Score: {score3}, Details: {details3}, Is Agentic: {is_agentic3}")
    assert is_agentic3 is True
    assert score3 == 8.0 # 3.0 format + 5.0 acc
    
    print("Task Rewards: OK")

if __name__ == "__main__":
    try:
        test_inference()
        test_rewards()
        print("\nALL TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
