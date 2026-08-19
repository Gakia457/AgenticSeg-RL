import json
from contextlib import redirect_stdout
from io import StringIO

from PIL import Image

from verl.agentic.reward_task3 import analyze_task3_rollout, compute_task3_reward
from verl.agentic.rollout_viz import should_log_step, select_rollout_indices


def test_task3_analyzer_returns_visualization_details():
    image = Image.new("RGB", (840, 840), "white")
    prediction = (
        '<think>brief</think><answer>'
        '[{"point_2d": [100, 100], "point_label": 1}]'
        "</answer>"
    )
    ground_truth = json.dumps(
        [
            {
                "point_2d": [100, 100],
                "point_label": 1,
                "polygon": [[[90, 90], [110, 90], [110, 110], [90, 110]]],
                "polygon_centers": [[100, 100]],
            }
        ]
    )

    analysis = analyze_task3_rollout(prediction, ground_truth, image, is_qwen3=False)

    assert analysis["score"] == 8.0
    assert analysis["format_reward"] == 2.0
    assert analysis["accuracy_reward"] == 6.0
    assert analysis["label_reward"] == 2.0
    assert analysis["proximity_reward"] == 4.0
    assert analysis["region_hit"] == 1
    assert analysis["center_distance"] is None
    assert analysis["matched_center"] is None
    assert analysis["best_pred_point"] == [100.0, 100.0]
    assert analysis["best_pred_label"] == 1


def test_task3_analyzer_ignores_polygon_center_and_uses_signed_distance_reward():
    image = Image.new("RGB", (840, 840), "white")
    ground_truth = json.dumps(
        [
            {
                "point_2d": [50, 50],
                "point_label": 1,
                "polygon": [[[0, 0], [100, 0], [100, 100], [0, 100]]],
                "polygon_centers": [[50, 50]],
            }
        ]
    )

    center_prediction = (
        '<think>brief</think><answer>'
        '[{"point_2d": [50, 50], "point_label": 1}]'
        "</answer>"
    )
    edge_prediction = (
        '<think>brief</think><answer>'
        '[{"point_2d": [0, 50], "point_label": 1}]'
        "</answer>"
    )
    outside_prediction = (
        '<think>brief</think><answer>'
        '[{"point_2d": [-20, 50], "point_label": 1}]'
        "</answer>"
    )
    wrong_label_inside_prediction = (
        '<think>brief</think><answer>'
        '[{"point_2d": [50, 50], "point_label": 0}]'
        "</answer>"
    )

    center = analyze_task3_rollout(center_prediction, ground_truth, image, is_qwen3=False)
    edge = analyze_task3_rollout(edge_prediction, ground_truth, image, is_qwen3=False)
    outside = analyze_task3_rollout(outside_prediction, ground_truth, image, is_qwen3=False)
    wrong_label_inside = analyze_task3_rollout(wrong_label_inside_prediction, ground_truth, image, is_qwen3=False)

    assert center["proximity_reward"] == 4.0
    assert center["center_distance"] is None
    assert center["matched_center"] is None
    assert edge["region_hit"] == 1
    assert edge["center_distance"] is None
    assert edge["proximity_reward"] == 4.0
    assert outside["region_hit"] == 0
    assert round(outside["proximity_reward"], 3) == 2.681
    assert outside["proximity_reward"] < edge["proximity_reward"]
    assert wrong_label_inside["label_reward"] == 0.0
    assert wrong_label_inside["region_hit"] == 0
    assert round(wrong_label_inside["proximity_reward"], 1) == 2.8


def test_task3_analyzer_keeps_json_format_and_accuracy_when_think_gate_is_disabled():
    image = Image.new("RGB", (840, 840), "white")
    prediction = '<answer>[{"point_2d": [100, 100], "point_label": 1}]</answer>'
    ground_truth = json.dumps(
        [
            {
                "point_2d": [100, 100],
                "point_label": 1,
                "polygon": [[[90, 90], [110, 90], [110, 110], [90, 110]]],
                "polygon_centers": [[100, 100]],
            }
        ]
    )

    analysis = analyze_task3_rollout(prediction, ground_truth, image, is_qwen3=False)

    assert analysis["format_reward"] == 1.0
    assert analysis["accuracy_reward"] == 6.0
    assert analysis["label_reward"] == 2.0
    assert analysis["proximity_reward"] == 4.0
    assert analysis["region_hit"] == 1


def test_task3_reward_debug_print_matches_old_poly_dist_log_format():
    image = Image.new("RGB", (840, 840), "white")
    prediction = (
        '<think>brief</think><answer>'
        '[{"point_2d": [50, 50], "point_label": 1}]'
        "</answer>"
    )
    ground_truth = json.dumps(
        [
            {
                "point_2d": [50, 50],
                "point_label": 1,
                "polygon": [[[0, 0], [100, 0], [100, 100], [0, 100]]],
                "polygon_centers": [[50, 50]],
            }
        ]
    )

    stdout = StringIO()
    with redirect_stdout(stdout):
        compute_task3_reward(prediction, ground_truth, image, is_qwen3=False)

    log_line = stdout.getvalue()
    assert "PolyDist=" in log_line
    assert "Hit=1" in log_line
    assert "CenterDist" not in log_line


def test_rollout_sampler_logs_every_ten_steps_and_picks_one_question_eight_rollouts():
    assert not should_log_step(9, 10)
    assert should_log_step(10, 10)

    selected = select_rollout_indices(
        total_rollouts=64,
        rollout_n=16,
        max_samples=1,
        max_rollouts_per_sample=8,
        global_step=10,
        seed=42,
    )

    assert len(selected) == 8
    assert len({item.sample_index for item in selected}) == 1
    assert len({item.rollout_index for item in selected}) == 8
    assert all(item.flat_index == item.sample_index * 16 + item.rollout_index for item in selected)


if __name__ == "__main__":
    test_task3_analyzer_returns_visualization_details()
    test_task3_analyzer_ignores_polygon_center_and_uses_signed_distance_reward()
    test_task3_analyzer_keeps_json_format_and_accuracy_when_think_gate_is_disabled()
    test_task3_reward_debug_print_matches_old_poly_dist_log_format()
    test_rollout_sampler_logs_every_ten_steps_and_picks_one_question_eight_rollouts()
