#!/usr/bin/env python3
"""Tests for Task3 rollout log analyzer CLI helpers."""

from pathlib import Path
from tempfile import TemporaryDirectory

from tools.rollout_analysis.analyze_task3_rollout_log import (
    analyze_log,
    default_output_dir_for_log,
    extract_run_name_from_log,
    filter_rows_by_step,
    resolve_output_dir,
)


def test_filter_rows_by_step_keeps_steps_from_start_through_limit() -> None:
    rows = [{"derived_step": 1}, {"derived_step": 2}, {"derived_step": 3}]
    step_metrics = [{"step": 1}, {"step": 2}, {"step": 3}]

    kept_rows, kept_step_metrics = filter_rows_by_step(rows, step_metrics, 2)

    assert [row["derived_step"] for row in kept_rows] == [1, 2]
    assert [row["step"] for row in kept_step_metrics] == [1, 2]


def test_filter_rows_by_step_keeps_all_when_limit_exceeds_log() -> None:
    rows = [{"derived_step": 1}, {"derived_step": 2}]
    step_metrics = [{"step": 1}, {"step": 2}]

    kept_rows, kept_step_metrics = filter_rows_by_step(rows, step_metrics, 100)

    assert kept_rows == rows
    assert kept_step_metrics == step_metrics


def test_resolve_output_dir_adds_step_suffix_only_when_requested() -> None:
    base_dir = Path("vis/task3_log_analysis_output_05212312")

    assert resolve_output_dir(base_dir, None) == base_dir
    assert resolve_output_dir(base_dir, 100) == Path("vis/task3_log_analysis_output_05212312_step100")


def test_default_output_dir_uses_run_name_under_log_dir_and_step_suffix() -> None:
    with TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "output_test.txt"
        log_path.write_text("+ RUN_NAME=AgenticSegRL_Qwen3VL8B_Task3/20260522_120000\n", encoding="utf-8")

        assert extract_run_name_from_log(log_path) == "AgenticSegRL_Qwen3VL8B_Task3/20260522_120000"
        assert default_output_dir_for_log(log_path, Path("log"), 100) == Path(
            "log/AgenticSegRL_Qwen3VL8B_Task3/20260522_120000_step100"
        )


def test_default_output_dir_falls_back_to_log_stem_when_run_name_missing() -> None:
    with TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "output_test.txt"
        log_path.write_text("no run name here\n", encoding="utf-8")

        assert extract_run_name_from_log(log_path) is None
        assert default_output_dir_for_log(log_path, Path("log"), None) == Path("log/output_test")


def test_analyze_log_step_limit_filters_reward_and_step_lines() -> None:
    reward_line = (
        "(WorkerDict pid=123) [AgenticRL] Task 3: "
        "Score=1.0/8, Fmt=1.0/2, Acc=[L:0.0/2, P:0.0/4] "
        "| PolyDist=-29.0, CenterDist=81.3, Hit=0\n"
    )
    step_lines = [
        "(main_task pid=999) step 1: - critic/score/mean:1.0\n",
        "(main_task pid=999) step 2: - critic/score/mean:2.0\n",
    ]
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        log_path = tmp_dir / "task3.log"
        out_dir = tmp_dir / "analysis"
        log_path.write_text(step_lines[0] + reward_line * 64 + step_lines[1] + reward_line * 16, encoding="utf-8")

        result = analyze_log(log_path, out_dir, step_limit=1)

        assert result["raw_parsed_rollouts"] == 80
        assert result["parsed_rollouts"] == 64
        assert result["total_rollouts"] == 64
        assert result["total_samples"] == 4
        assert [row["step"] for row in result["step_metrics"]] == [1]


if __name__ == "__main__":
    test_filter_rows_by_step_keeps_steps_from_start_through_limit()
    test_filter_rows_by_step_keeps_all_when_limit_exceeds_log()
    test_resolve_output_dir_adds_step_suffix_only_when_requested()
    test_default_output_dir_uses_run_name_under_log_dir_and_step_suffix()
    test_default_output_dir_falls_back_to_log_stem_when_run_name_missing()
    test_analyze_log_step_limit_filters_reward_and_step_lines()
