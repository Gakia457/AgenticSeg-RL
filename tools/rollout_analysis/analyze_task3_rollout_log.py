#!/usr/bin/env python3
"""Task3 终端 reward 日志分析脚本。

这个脚本用于分析一次 Task3 训练保存下来的终端输出 txt。日志路径通过命令行
`--input-log` 传入，不再绑定某个固定的 output_*.txt。

脚本不会读 wandb 文件，也不会重新算 reward，只会从终端日志里抽取形如下面的
Task3 reward 打印：

    [AgenticRL] Task 3: Score=.../8, Fmt=.../2,
    Acc=[L:.../2, P:.../4] | PolyDist=..., CenterDist=..., Hit=...

核心分析口径：

    连续 16 条 Task3 reward 行 = 同一个 sample 的 16 次 rollout

原因是这次训练配置为：

    data.rollout_batch_size=4
    worker.rollout.n=16

也就是一个训练 step 有 4 个 sample，每个 sample 有 16 条 rollout。
脚本不以 step 作为主分析单位；step 只通过 sample_id 派生出来，用于辅助对账。

会产出三种统计视角：

    1. all-rollout view
       每条 rollout 都参与统计，用来看整体输出分布、format reward、label 是否正确、
       proximity reward、Hit=1 比例、No points predicted 比例等。

    2. best-of-sample view
       每个 sample 只取 16 条 rollout 中 score 最高的一条，用来看“这个问题是否至少
       有一次 rollout 给出了可提供优势值的较好答案”。

    3. sample-any view
       每个 sample 只判断 16 次 rollout 里是否至少有一次满足条件。这个视角适合看
       “这个 sample 有没有出现过可学习信号”，但 score 阈值类信息主要用 best 分布表达，
       不再额外堆 score_ge_* rate。

命令行参数说明：

    --input-log
        输入日志 txt。下一次换训练输出文件时，只需要改这个参数。

    --output-root
        默认输出根目录。可不传，默认是 `./log`。脚本会从输入 txt 里读取
        `RUN_NAME` / `trainer.experiment_name`，生成 `./log/<run_name>`。

    --output-dir
        手动指定完整分析产物目录。传了这个参数时，不再按 run_name 自动生成目录。

    --step
        只统计从开头到指定 step 的日志。例如 `--step 100` 表示只分析 step 1 到
        step 100。若指定值超过日志实际 step 数，就等价于分析整个日志。设置该参数后，
        最终输出目录会自动追加 `_stepN` 后缀，例如 `_step100`。

    ROLLOUTS_PER_SAMPLE
        每个 sample 的 rollout 数。当前训练 worker.rollout.n=16，所以这里是 16。
        如果之后 rollout.n 改了，这里也要同步改。

    SAMPLES_PER_STEP
        每个 step 的 sample 数。当前 data.rollout_batch_size=4，所以这里是 4。
        这个值只影响 derived_step 和 sample_in_step，不影响 16-rollout sample 主分析。

常用命令：

    运行分析：
        cd .
        MPLCONFIGDIR=/tmp/matplotlib-task3-log python3 tools/rollout_analysis/analyze_task3_rollout_log.py \
            --input-log ./output_05220121.txt

    只统计前 100 个 step：
        cd .
        MPLCONFIGDIR=/tmp/matplotlib-task3-log python3 tools/rollout_analysis/analyze_task3_rollout_log.py \
            --input-log ./output_05220121.txt \
            --step 100

    改输出根目录：
        cd .
        MPLCONFIGDIR=/tmp/matplotlib-task3-log python3 tools/rollout_analysis/analyze_task3_rollout_log.py \
            --input-log ./output_05221138.txt \
            --output-root ./log
            
        # 目前使用
        MPLCONFIGDIR=/tmp/matplotlib-task3-log python3 tools/rollout_analysis/analyze_task3_rollout_log.py \
            --input-log ./output_05222254.txt \
            --output-root ./log \
            --step 800

    只检查语法：
        cd .
        python3 -m py_compile tools/rollout_analysis/analyze_task3_rollout_log.py

主要输出文件：

    task3_analysis_report.zh.md / task3_analysis_report.en.md
        自动生成的中文和英文报告。只保留这两份 report。

    all_rollouts/task3_rollout_rows.csv
        每条 Task3 reward 日志一行，包含 score、fmt、label_reward、
        proximity_reward、poly_dist、center_dist、hit、pid、sample_id、rollout_in_sample 等字段。

    task3_sample_summary.csv
        每 16 条 rollout 聚合成一个 sample 后的统计。

    best_per_sample/task3_best_per_sample.csv
        每个 sample 只保留 score 最高的一条 rollout。

    task3_step_summary_lines.csv
        解析 trainer 的 "step N:" summary，主要用于和 reward 聚合结果对账。

    all_rollouts/task3_metric_summary.csv / best_per_sample/task3_metric_summary.csv
        all-rollout 和 best-of-sample 两个视角下，各 reward 小项的统计量。

    all_rollouts/task3_rate_summary.csv / best_per_sample/task3_rate_summary.csv
        all-rollout 和 best-of-sample 两个视角下的 rate 指标。

    task3_sample_any_rate_summary.csv
        sample-any 视角下的 rate 指标，回答 16 次 rollout 里是否至少一次满足条件。

    all_rollouts/task3_poly_label_split.csv / best_per_sample/task3_poly_label_split.csv
        按 label_reward 是否拿到拆开统计 poly_near 和 poly_inside_or_touch。

    all_rollouts/task3_reward_component_bins.csv / best_per_sample/task3_reward_component_bins.csv
        score、format、accuracy、label、proximity、poly_dist、center_dist 等指标的分桶分布。

    all_rollouts/*.png
        all-rollout 视角下的分布图和 cumulative ratio 图。

    best_per_sample/*.png
        best-of-sample 视角下的分布图和 cumulative ratio 图，这是当前最重点看的目录。

    comparison/*.png
        all-rollout 和 best-of-sample 两个视角叠在一起的分布/CDF 对比图。

注意事项：

    1. 如果输入日志还在被 tee 持续写入，脚本结果只代表运行那一刻的快照。
       训练结束后需要重新跑一次脚本。

    2. 当前终端 reward 行没有打印模型原始 point_label，只打印了 label_reward。
       因此脚本能统计 label 是否正确，但不能直接统计模型偏向输出 label 0 还是 label 1。

    3. "No points predicted." 这类日志会被解析为 0 分 rollout。

    4. 如果日志尾部因为训练仍在写入而出现不满 16 条的半个 sample，脚本会丢弃这部分
       尾部 rollout，只分析完整 sample，并在 report 中记录丢弃数量。
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any


ROLLOUTS_PER_SAMPLE = 16
SAMPLES_PER_STEP = 4
POLY_NEAR_THRESHOLD = -50.0
LEGACY_ASSET_NAMES = [
    "score_distribution.png",
    "accuracy_distribution.png",
    "format_distribution.png",
    "label_reward_distribution.png",
    "proximity_reward_distribution.png",
    "poly_dist_distribution.png",
    "center_dist_distribution.png",
    "sample_behavior_rates.png",
    "sample_score_curve.png",
    "sample_mean_vs_best_score_curve.png",
    "sample_component_rate_curves.png",
    "sample_best_component_curves.png",
    "all_vs_best_score_distribution.png",
    "all_vs_best_accuracy_distribution.png",
    "rate_summary_bar.png",
    "task3_analysis_report.md",
    "task3_rollout_rows.csv",
    "task3_best_per_sample.csv",
    "task3_metric_summary_all_rollouts.csv",
    "task3_metric_summary_best_per_sample.csv",
    "task3_rate_summary.csv",
    "task3_reward_component_bins_all_rollouts.csv",
    "task3_reward_component_bins_best_per_sample.csv",
    "task3_poly_label_split_all_rollouts.csv",
    "task3_poly_label_split_best_per_sample.csv",
]

METRIC_KEYS = [
    "score",
    "fmt",
    "accuracy_reward",
    "label_reward",
    "proximity_reward",
    "poly_dist",
    "center_dist",
]

BIN_SPECS = {
    "score": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "fmt": [0, 0.5, 1.5, 2.5],
    "accuracy_reward": [0, 1, 2, 3, 4, 5, 6, 7],
    "label_reward": [0, 0.5, 1.5, 2.5],
    "proximity_reward": [0, 1, 2, 3, 4, 5],
    "poly_dist": [-math.inf, -200, -100, -50, 0, 50, math.inf],
    "center_dist": [0, 10, 25, 50, 100, 200, math.inf],
}

ANSI_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
TASK3_RE = re.compile(
    r"\(WorkerDict pid=(?P<pid>\d+)\).*?\[AgenticRL\] Task 3: "
    r"Score=(?P<score>-?\d+(?:\.\d+)?)/8, "
    r"Fmt=(?P<fmt>-?\d+(?:\.\d+)?)/2, "
    r"Acc=\[L:(?P<label>-?\d+(?:\.\d+)?)/2, P:(?P<prox>-?\d+(?:\.\d+)?)/4\] "
    r"\| PolyDist=(?P<poly>-?\d+(?:\.\d+)?|None)"
    r"(?:, CenterDist=(?P<center>-?\d+(?:\.\d+)?|None))?, Hit=(?P<hit>[01])"
)
NO_POINTS_RE = re.compile(
    r"\(WorkerDict pid=(?P<pid>\d+)\).*?\[AgenticRL\] Task 3: "
    r"Score=(?P<score>-?\d+(?:\.\d+)?)/8, No points predicted\."
)
STEP_RE = re.compile(r"\(main_task pid=\d+\) step (?P<step>\d+): (?P<body>.*)")
METRIC_RE = re.compile(r"- (?P<name>[\w/]+):(?P<value>-?\d+(?:\.\d+)?)")
RUN_NAME_PATTERNS = [
    re.compile(r"(?:^|\s)\+?\s*RUN_NAME=(?P<name>\S+)"),
    re.compile(r"trainer\.experiment_name=(?P<name>\S+)"),
    re.compile(r'"experiment_name"\s*:\s*"(?P<name>[^"]+)"'),
]


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("--step 必须是正整数")
    return value


def resolve_output_dir(base_dir: Path, step_limit: int | None) -> Path:
    if step_limit is None:
        return base_dir
    return base_dir.with_name(f"{base_dir.name}_step{step_limit}")


def extract_run_name_from_log(log_path: Path, max_lines: int = 5000) -> str | None:
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_index, line in enumerate(f):
            if line_index >= max_lines:
                break
            clean = strip_ansi(line)
            for pattern in RUN_NAME_PATTERNS:
                match = pattern.search(clean)
                if match is not None:
                    return match.group("name").strip().strip("'\"")
    return None


def default_output_dir_for_log(log_path: Path, output_root: Path = Path("log"), step_limit: int | None = None) -> Path:
    run_name = extract_run_name_from_log(log_path) or log_path.stem
    return resolve_output_dir(output_root / Path(run_name), step_limit)


def filter_rows_by_step(
    rows: list[dict[str, Any]],
    step_metrics: list[dict[str, Any]],
    step_limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if step_limit is None:
        return rows, step_metrics
    filtered_rows = [row for row in rows if row["derived_step"] <= step_limit]
    filtered_step_metrics = [step for step in step_metrics if step["step"] <= step_limit]
    return filtered_rows, filtered_step_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Task3 terminal reward logs.")
    parser.add_argument("--input-log", type=Path, required=True, help="Task3 training terminal log txt path.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("log"),
        help="Output root used when --output-dir is not set. Defaults to ./log.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Explicit final output directory. When omitted, uses <output-root>/<run-name-from-log>.",
    )
    parser.add_argument(
        "--step",
        type=positive_int,
        default=None,
        help="Analyze logs from the beginning through this derived training step.",
    )
    return parser.parse_args()


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_task3_line(line: str, line_no: int, rollout_index: int) -> dict[str, Any] | None:
    clean = strip_ansi(line)
    match = TASK3_RE.search(clean)
    no_points = False
    if match is None:
        match = NO_POINTS_RE.search(clean)
        no_points = match is not None
    if match is None:
        return None

    sample_id = rollout_index // ROLLOUTS_PER_SAMPLE + 1
    rollout_in_sample = rollout_index % ROLLOUTS_PER_SAMPLE
    derived_step = (sample_id - 1) // SAMPLES_PER_STEP + 1
    sample_in_step = (sample_id - 1) % SAMPLES_PER_STEP

    if no_points:
        score = float(match.group("score"))
        fmt = label = prox = poly = 0.0
        center_dist = math.nan
        hit = 0
    else:
        score = float(match.group("score"))
        fmt = float(match.group("fmt"))
        label = float(match.group("label"))
        prox = float(match.group("prox"))
        poly_text = match.group("poly")
        poly = math.nan if poly_text == "None" else float(poly_text)
        center_text = match.group("center")
        center_dist = math.nan if center_text in (None, "None") else float(center_text)
        hit = int(match.group("hit"))

    return {
        "line_no": line_no,
        "pid": int(match.group("pid")),
        "rollout_global_index": rollout_index,
        "sample_id": sample_id,
        "rollout_in_sample": rollout_in_sample,
        "derived_step": derived_step,
        "sample_in_step": sample_in_step,
        "score": score,
        "fmt": fmt,
        "label_reward": label,
        "proximity_reward": prox,
        "accuracy_reward": label + prox,
        "poly_dist": poly,
        "center_dist": center_dist,
        "hit": hit,
        "label_correct": label > 0.0,
        "format_positive": fmt > 0.0,
        "proximity_positive": prox > 0.0,
        "proximity_full": prox >= 4.0,
        "accuracy_positive": (label + prox) > 0.0,
        "accuracy_high": (label + prox) >= 4.0,
        "poly_inside_or_touch": (not math.isnan(poly)) and poly >= 0.0,
        "poly_near": (not math.isnan(poly)) and poly >= POLY_NEAR_THRESHOLD,
        "no_points_predicted": no_points,
    }


def parse_step_line(line: str, line_no: int) -> dict[str, Any] | None:
    clean = strip_ansi(line)
    match = STEP_RE.search(clean)
    if match is None:
        return None
    metrics = {m.group("name"): float(m.group("value")) for m in METRIC_RE.finditer(match.group("body"))}
    metrics["step"] = int(match.group("step"))
    metrics["line_no"] = line_no
    return metrics


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row[key]
        if isinstance(value, float) and math.isnan(value):
            continue
        values.append(float(value))
    return values


def summarize_values(values: list[float], prefix: str = "") -> dict[str, float]:
    name = f"{prefix}_" if prefix else ""
    if not values:
        return {
            f"{name}count": 0,
            f"{name}mean": math.nan,
            f"{name}std": math.nan,
            f"{name}min": math.nan,
            f"{name}q25": math.nan,
            f"{name}median": math.nan,
            f"{name}q75": math.nan,
            f"{name}max": math.nan,
        }
    return {
        f"{name}count": len(values),
        f"{name}mean": mean(values),
        f"{name}std": pstdev(values) if len(values) > 1 else 0.0,
        f"{name}min": min(values),
        f"{name}q25": percentile(values, 0.25),
        f"{name}median": median(values),
        f"{name}q75": percentile(values, 0.75),
        f"{name}max": max(values),
    }


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
    return ordered[index]


def rate(rows: list[dict[str, Any]], predicate) -> float:
    if not rows:
        return math.nan
    return sum(1 for row in rows if predicate(row)) / len(rows)


def count_if(rows: list[dict[str, Any]], predicate) -> int:
    return sum(1 for row in rows if predicate(row))


def metric_summary_rows(view: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows = []
    for key in METRIC_KEYS:
        item: dict[str, Any] = {"view": view, "metric": key}
        item.update(summarize_values(numeric_values(rows, key)))
        summary_rows.append(item)
    return summary_rows


def rate_summary_rows(view: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("format_positive_rate", "fmt > 0", lambda row: row["fmt"] > 0.0),
        ("label_correct_rate", "label_reward > 0", lambda row: row["label_reward"] > 0.0),
        ("proximity_positive_rate", "proximity_reward > 0", lambda row: row["proximity_reward"] > 0.0),
        ("proximity_full_rate", "proximity_reward >= 4", lambda row: row["proximity_reward"] >= 4.0),
        ("accuracy_positive_rate", "accuracy_reward > 0", lambda row: row["accuracy_reward"] > 0.0),
        ("accuracy_high_rate", "accuracy_reward >= 4", lambda row: row["accuracy_reward"] >= 4.0),
        ("hit_rate", "Hit == 1", lambda row: row["hit"] == 1),
        ("no_point_rate", "No points predicted", lambda row: row["no_points_predicted"]),
        ("poly_inside_or_touch_rate", "poly_dist >= 0", lambda row: not math.isnan(row["poly_dist"]) and row["poly_dist"] >= 0.0),
        ("poly_near_rate", f"poly_dist >= {POLY_NEAR_THRESHOLD}", lambda row: not math.isnan(row["poly_dist"]) and row["poly_dist"] >= POLY_NEAR_THRESHOLD),
    ]
    return [
        {
            "view": view,
            "rate_name": name,
            "definition": definition,
            "numerator": count_if(rows, predicate),
            "denominator": len(rows),
            "rate": rate(rows, predicate),
        }
        for name, definition, predicate in specs
    ]


def sample_any_rate_rows(sample_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = [
        ("samples_with_any_fmt_positive", "at least one rollout has fmt > 0", lambda row: row["any_format_positive"]),
        ("samples_with_any_label_correct", "at least one rollout has label_reward > 0", lambda row: row["any_label_correct"]),
        ("samples_with_any_proximity_positive", "at least one rollout has proximity_reward > 0", lambda row: row["any_proximity_positive"]),
        ("samples_with_any_proximity_full", "at least one rollout has proximity_reward >= 4", lambda row: row["any_proximity_full"]),
        ("samples_with_any_accuracy_positive", "at least one rollout has accuracy_reward > 0", lambda row: row["any_accuracy_positive"]),
        ("samples_with_any_accuracy_high", "at least one rollout has accuracy_reward >= 4", lambda row: row["any_accuracy_high"]),
        ("samples_with_any_hit", "at least one rollout has Hit == 1", lambda row: row["any_hit"]),
    ]
    return [
        {
            "view": "sample_any",
            "rate_name": name,
            "definition": definition,
            "numerator": count_if(sample_summaries, predicate),
            "denominator": len(sample_summaries),
            "rate": rate(sample_summaries, predicate),
        }
        for name, definition, predicate in specs
    ]


def bin_label(left: float, right: float) -> str:
    if left == -math.inf:
        return f"<{right:g}"
    if right == math.inf:
        return f">={left:g}"
    return f"[{left:g},{right:g})"


def bin_distribution_rows(view: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    total_by_metric = {key: len(numeric_values(rows, key)) for key in METRIC_KEYS}
    for key in METRIC_KEYS:
        values = numeric_values(rows, key)
        edges = BIN_SPECS[key]
        for left, right in zip(edges[:-1], edges[1:]):
            count = 0
            for value in values:
                if left <= value < right or (right == edges[-1] and value == right):
                    count += 1
            total = total_by_metric[key]
            output.append(
                {
                    "view": view,
                    "metric": key,
                    "bin": bin_label(left, right),
                    "count": count,
                    "total": total,
                    "ratio": count / total if total else math.nan,
                }
            )
    return output


def poly_label_split_rows(view: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    groups = [
        ("label_reward_positive", lambda row: row["label_reward"] > 0.0),
        ("label_reward_zero", lambda row: row["label_reward"] <= 0.0),
    ]
    for group_name, predicate in groups:
        group_rows = [row for row in rows if predicate(row)]
        output.append(
            {
                "view": view,
                "group": group_name,
                "count": len(group_rows),
                "poly_near_count": count_if(group_rows, lambda row: not math.isnan(row["poly_dist"]) and row["poly_dist"] >= POLY_NEAR_THRESHOLD),
                "poly_near_rate": rate(group_rows, lambda row: not math.isnan(row["poly_dist"]) and row["poly_dist"] >= POLY_NEAR_THRESHOLD),
                "poly_inside_or_touch_count": count_if(group_rows, lambda row: not math.isnan(row["poly_dist"]) and row["poly_dist"] >= 0.0),
                "poly_inside_or_touch_rate": rate(group_rows, lambda row: not math.isnan(row["poly_dist"]) and row["poly_dist"] >= 0.0),
                "poly_dist_mean": mean(numeric_values(group_rows, "poly_dist")) if group_rows else math.nan,
                "poly_dist_median": median(numeric_values(group_rows, "poly_dist")) if group_rows else math.nan,
            }
        )
    return output


def summarize_sample(sample_id: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = numeric_values(rows, "score")
    fmt_values = numeric_values(rows, "fmt")
    label_values = numeric_values(rows, "label_reward")
    prox_values = numeric_values(rows, "proximity_reward")
    acc_values = numeric_values(rows, "accuracy_reward")
    poly_values = numeric_values(rows, "poly_dist")
    center_values = numeric_values(rows, "center_dist")
    best = best_row(rows)

    summary: dict[str, Any] = {
        "sample_id": sample_id,
        "derived_step": rows[0]["derived_step"],
        "sample_in_step": rows[0]["sample_in_step"],
        "rollout_count": len(rows),
        "pid_set": " ".join(str(pid) for pid in sorted({row["pid"] for row in rows})),
        "hit_count": sum(int(row["hit"]) for row in rows),
        "hit_rate": mean([float(row["hit"]) for row in rows]),
        "label_correct_count": sum(int(row["label_correct"]) for row in rows),
        "label_correct_rate": mean([float(row["label_correct"]) for row in rows]),
        "format_positive_count": sum(int(row["format_positive"]) for row in rows),
        "format_positive_rate": mean([float(row["format_positive"]) for row in rows]),
        "proximity_positive_count": sum(int(row["proximity_positive"]) for row in rows),
        "proximity_positive_rate": mean([float(row["proximity_positive"]) for row in rows]),
        "proximity_full_count": sum(int(row["proximity_full"]) for row in rows),
        "proximity_full_rate": mean([float(row["proximity_full"]) for row in rows]),
        "accuracy_positive_count": sum(int(row["accuracy_positive"]) for row in rows),
        "accuracy_positive_rate": mean([float(row["accuracy_positive"]) for row in rows]),
        "accuracy_high_count": sum(int(row["accuracy_high"]) for row in rows),
        "accuracy_high_rate": mean([float(row["accuracy_high"]) for row in rows]),
        "poly_inside_or_touch_count": sum(int(row["poly_inside_or_touch"]) for row in rows),
        "poly_inside_or_touch_rate": mean([float(row["poly_inside_or_touch"]) for row in rows]),
        "poly_near_count": sum(int(row["poly_near"]) for row in rows),
        "poly_near_rate": mean([float(row["poly_near"]) for row in rows]),
        "no_point_count": sum(int(row["no_points_predicted"]) for row in rows),
        "no_point_rate": mean([float(row["no_points_predicted"]) for row in rows]),
        "any_hit": any(row["hit"] == 1 for row in rows),
        "any_label_correct": any(row["label_correct"] for row in rows),
        "any_format_positive": any(row["format_positive"] for row in rows),
        "any_proximity_positive": any(row["proximity_positive"] for row in rows),
        "any_proximity_full": any(row["proximity_full"] for row in rows),
        "any_accuracy_positive": any(row["accuracy_positive"] for row in rows),
        "any_accuracy_high": any(row["accuracy_high"] for row in rows),
        "any_poly_inside_or_touch": any(row["poly_inside_or_touch"] for row in rows),
        "any_poly_near": any(row["poly_near"] for row in rows),
        "score_q25": percentile(scores, 0.25),
        "score_q75": percentile(scores, 0.75),
        "best_rollout_in_sample": best["rollout_in_sample"],
        "best_score": best["score"],
        "best_fmt": best["fmt"],
        "best_accuracy_reward": best["accuracy_reward"],
        "best_label_reward": best["label_reward"],
        "best_proximity_reward": best["proximity_reward"],
        "best_hit": best["hit"],
        "best_poly_dist": best["poly_dist"],
        "best_center_dist": best["center_dist"],
        "best_no_points_predicted": best["no_points_predicted"],
    }
    for key, values in [
        ("score", scores),
        ("fmt", fmt_values),
        ("label_reward", label_values),
        ("proximity_reward", prox_values),
        ("accuracy_reward", acc_values),
        ("poly_dist", poly_values),
        ("center_dist", center_values),
    ]:
        summary.update(summarize_values(values, key))
    return summary


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["score"],
            row["hit"],
            row["label_reward"],
            row["proximity_reward"],
            -row["rollout_in_sample"],
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dirs(out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in LEGACY_ASSET_NAMES:
        legacy_path = out_dir / name
        if legacy_path.exists():
            legacy_path.unlink()
    all_dir = out_dir / "all_rollouts"
    best_dir = out_dir / "best_per_sample"
    comparison_dir = out_dir / "comparison"
    for subdir in [all_dir, best_dir, comparison_dir]:
        if subdir.exists():
            shutil.rmtree(subdir)
        subdir.mkdir(parents=True, exist_ok=True)
    return all_dir, best_dir, comparison_dir


def write_plots(out_dir: Path, all_dir: Path, best_dir: Path, comparison_dir: Path, rows: list[dict[str, Any]], best_rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local environment
        return [f"matplotlib unavailable, skipped plots: {exc}"]

    notes: list[str] = []

    def save_hist(target_dir: Path, source_rows: list[dict[str, Any]], metric: str, title: str, filename: str, bins: int = 24) -> None:
        values = numeric_values(source_rows, metric)
        if not values:
            return
        plt.figure(figsize=(10, 4))
        plt.hist(values, bins=bins, alpha=0.82)
        plt.xlabel(metric)
        plt.ylabel("count")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(target_dir / filename, dpi=160)
        plt.close()
        notes.append(str((target_dir / filename).relative_to(out_dir)))

    def save_cdf(target_dir: Path, source_rows: list[dict[str, Any]], metric: str, title: str, filename: str) -> None:
        values = sorted(numeric_values(source_rows, metric))
        if not values:
            return
        y = [(idx + 1) / len(values) for idx in range(len(values))]
        plt.figure(figsize=(9, 4))
        plt.plot(values, y, linewidth=1.8)
        plt.xlabel(metric)
        plt.ylabel("cumulative ratio")
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(target_dir / filename, dpi=160)
        plt.close()
        notes.append(str((target_dir / filename).relative_to(out_dir)))

    def save_poly_label_split(target_dir: Path, source_rows: list[dict[str, Any]], title: str, filename: str) -> None:
        yes = [row["poly_dist"] for row in source_rows if row["label_reward"] > 0.0 and not math.isnan(row["poly_dist"])]
        no = [row["poly_dist"] for row in source_rows if row["label_reward"] <= 0.0 and not math.isnan(row["poly_dist"])]
        if not yes and not no:
            return
        plt.figure(figsize=(10, 4))
        if yes:
            plt.hist(yes, bins=30, alpha=0.65, label="label reward > 0")
        if no:
            plt.hist(no, bins=30, alpha=0.65, label="label reward = 0")
        plt.axvline(POLY_NEAR_THRESHOLD, color="black", linestyle="--", linewidth=1, label="poly near threshold")
        plt.axvline(0, color="black", linestyle="-", linewidth=1, label="inside boundary")
        plt.xlabel("poly_dist")
        plt.ylabel("count")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(target_dir / filename, dpi=160)
        plt.close()
        notes.append(str((target_dir / filename).relative_to(out_dir)))

    def save_overlay_hist(metric: str, title: str, filename: str) -> None:
        all_values = numeric_values(rows, metric)
        best_values = numeric_values(best_rows, metric)
        if not all_values or not best_values:
            return
        plt.figure(figsize=(10, 4))
        plt.hist(all_values, bins=24, alpha=0.7, label="all rollouts")
        plt.hist(best_values, bins=24, alpha=0.55, label="best per sample")
        plt.xlabel(metric)
        plt.ylabel("count")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(comparison_dir / filename, dpi=160)
        plt.close()
        notes.append(str((comparison_dir / filename).relative_to(out_dir)))

    def save_overlay_cdf(metric: str, title: str, filename: str) -> None:
        all_values = sorted(numeric_values(rows, metric))
        best_values = sorted(numeric_values(best_rows, metric))
        if not all_values or not best_values:
            return
        all_y = [(idx + 1) / len(all_values) for idx in range(len(all_values))]
        best_y = [(idx + 1) / len(best_values) for idx in range(len(best_values))]
        plt.figure(figsize=(9, 4))
        plt.plot(all_values, all_y, linewidth=1.6, label="all rollouts")
        plt.plot(best_values, best_y, linewidth=1.8, label="best per sample")
        plt.xlabel(metric)
        plt.ylabel("cumulative ratio")
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(comparison_dir / filename, dpi=160)
        plt.close()
        notes.append(str((comparison_dir / filename).relative_to(out_dir)))

    for target_dir, source_rows, view_name in [
        (all_dir, rows, "all rollouts"),
        (best_dir, best_rows, "best per sample"),
    ]:
        save_hist(target_dir, source_rows, "score", f"Task3 score distribution, {view_name}", "score_distribution.png")
        save_hist(target_dir, source_rows, "accuracy_reward", f"Task3 accuracy reward distribution, {view_name}", "accuracy_reward_distribution.png")
        save_hist(target_dir, source_rows, "fmt", f"Task3 format reward distribution, {view_name}", "format_distribution.png", bins=6)
        save_hist(target_dir, source_rows, "label_reward", f"Task3 label reward distribution, {view_name}", "label_reward_distribution.png", bins=6)
        save_hist(target_dir, source_rows, "proximity_reward", f"Task3 proximity reward distribution, {view_name}", "proximity_reward_distribution.png")
        save_hist(target_dir, source_rows, "poly_dist", f"Task3 polygon signed distance distribution, {view_name}", "poly_dist_distribution.png")
        save_hist(target_dir, source_rows, "center_dist", f"Task3 center distance distribution, {view_name}", "center_dist_distribution.png")
        for metric in METRIC_KEYS:
            save_cdf(target_dir, source_rows, metric, f"Task3 {metric} cumulative distribution, {view_name}", f"{metric}_cdf.png")
        save_poly_label_split(target_dir, source_rows, f"Task3 poly distance split by label reward, {view_name}", "poly_dist_by_label_reward.png")

    for metric in METRIC_KEYS:
        save_overlay_hist(metric, f"Task3 {metric}: all rollouts vs best per sample", f"{metric}_distribution_comparison.png")
        save_overlay_cdf(metric, f"Task3 {metric} CDF: all rollouts vs best per sample", f"{metric}_cdf_comparison.png")

    rate_rows = rate_summary_rows("all_rollouts", rows)
    rate_rows += rate_summary_rows("best_per_sample", best_rows)
    selected = [
        row
        for row in rate_rows
        if row["rate_name"]
        in {
            "label_correct_rate",
            "proximity_positive_rate",
            "proximity_full_rate",
            "accuracy_high_rate",
            "hit_rate",
            "poly_near_rate",
            "poly_inside_or_touch_rate",
        }
    ]
    if selected:
        labels = [f"{row['view']}\n{row['rate_name'].replace('_rate', '')}" for row in selected]
        values = [row["rate"] for row in selected]
        plt.figure(figsize=(max(12, len(labels) * 0.55), 5))
        plt.bar(range(len(labels)), values)
        plt.xticks(range(len(labels)), labels, rotation=55, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("rate")
        plt.title("Task3 key rate summary")
        plt.tight_layout()
        plt.savefig(comparison_dir / "rate_summary_bar.png", dpi=160)
        plt.close()
        notes.append(str((comparison_dir / "rate_summary_bar.png").relative_to(out_dir)))

    return notes


def write_report(
    out_dir: Path,
    log_path: Path,
    parsed_rollout_count: int,
    raw_parsed_rollout_count: int,
    step_limit: int | None,
    dropped_tail_rollouts: int,
    rows: list[dict[str, Any]],
    sample_summaries: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    step_metrics: list[dict[str, Any]],
    all_metric_summary: list[dict[str, Any]],
    best_metric_summary: list[dict[str, Any]],
    rate_summary: list[dict[str, Any]],
    all_bins: list[dict[str, Any]],
    best_bins: list[dict[str, Any]],
    all_poly_label_split: list[dict[str, Any]],
    best_poly_label_split: list[dict[str, Any]],
    plot_notes: list[str],
) -> None:
    pid_counter = Counter(row["pid"] for row in rows)
    incomplete_samples = [row for row in sample_summaries if row["rollout_count"] != ROLLOUTS_PER_SAMPLE]

    def format_metric_table(title: str, metric_rows: list[dict[str, Any]], zh: bool = False) -> list[str]:
        headers = ["指标", "均值", "中位数", "标准差", "最小值", "25分位", "75分位", "最大值"] if zh else ["metric", "mean", "median", "std", "min", "q25", "q75", "max"]
        table = [
            f"## {title}",
            "",
            "| " + " | ".join(headers) + " |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in metric_rows:
            table.append(
                "| {metric} | {mean:.4f} | {median:.4f} | {std:.4f} | {min:.4f} | {q25:.4f} | {q75:.4f} | {max:.4f} |".format(
                    **row
                )
            )
        return table + [""]

    def format_rate_table(title: str, view: str, zh: bool = False) -> list[str]:
        selected = [row for row in rate_summary if row["view"] == view]
        headers = ["rate", "数值", "分子 / 分母", "定义"] if zh else ["rate", "value", "numerator / denominator", "definition"]
        table = [
            f"## {title}",
            "",
            "| " + " | ".join(headers) + " |",
            "|---|---:|---:|---|",
        ]
        for row in selected:
            definition = zh_rate_definition(row["rate_name"]) if zh else row["definition"]
            table.append(
                f"| {row['rate_name']} | {row['rate']:.4f} | {row['numerator']} / {row['denominator']} | {definition} |"
            )
        return table + [""]

    def zh_rate_definition(rate_name: str) -> str:
        mapping = {
            "format_positive_rate": "格式奖励大于 0 的比例",
            "label_correct_rate": "label_reward 大于 0 的比例",
            "proximity_positive_rate": "proximity_reward 大于 0 的比例",
            "proximity_full_rate": "proximity_reward 达到 4 分的比例",
            "accuracy_positive_rate": "accuracy_reward 大于 0 的比例",
            "accuracy_high_rate": "accuracy_reward 不低于 4 分的比例",
            "hit_rate": "严格命中 Hit=1 的比例",
            "no_point_rate": "没有解析出预测点的比例",
            "poly_inside_or_touch_rate": "poly_dist 不小于 0 的比例，即点在 polygon 内或边界上",
            "poly_near_rate": f"poly_dist 不小于 {POLY_NEAR_THRESHOLD} 的比例，即点接近 polygon",
            "samples_with_any_fmt_positive": "每个 sample 的 16 次 rollout 中至少一次格式奖励大于 0 的比例",
            "samples_with_any_label_correct": "每个 sample 的 16 次 rollout 中至少一次 label_reward 大于 0 的比例",
            "samples_with_any_proximity_positive": "每个 sample 的 16 次 rollout 中至少一次 proximity_reward 大于 0 的比例",
            "samples_with_any_proximity_full": "每个 sample 的 16 次 rollout 中至少一次 proximity_reward 达到 4 分的比例",
            "samples_with_any_accuracy_positive": "每个 sample 的 16 次 rollout 中至少一次 accuracy_reward 大于 0 的比例",
            "samples_with_any_accuracy_high": "每个 sample 的 16 次 rollout 中至少一次 accuracy_reward 不低于 4 分的比例",
            "samples_with_any_hit": "每个 sample 的 16 次 rollout 中至少一次严格命中的比例",
        }
        return mapping.get(rate_name, rate_name)

    def format_poly_split_table(title: str, split_rows: list[dict[str, Any]], zh: bool = False) -> list[str]:
        headers = ["分组", "数量", "poly_near 比例", "poly_inside_or_touch 比例", "poly_dist 中位数"] if zh else ["group", "count", "poly_near_rate", "poly_inside_or_touch_rate", "poly_dist_median"]
        table = [
            f"## {title}",
            "",
            "| " + " | ".join(headers) + " |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in split_rows:
            table.append(
                f"| {row['group']} | {row['count']} | {row['poly_near_rate']:.4f} | {row['poly_inside_or_touch_rate']:.4f} | {row['poly_dist_median']:.4f} |"
            )
        return table + [""]

    zh_head = [
        "",
        f"- 输入日志：`{log_path}`",
        f"- step 截断参数：`{step_limit if step_limit is not None else '未设置'}`",
        f"- step 截断前解析到的 rollout 总数：`{raw_parsed_rollout_count}`",
        f"- 解析到的 rollout 总数：`{parsed_rollout_count}`",
        f"- 参与分析的完整 sample rollout 数：`{len(rows)}`",
        f"- 丢弃的尾部不完整 rollout 数：`{dropped_tail_rollouts}`",
        f"- 按 16 条 rollout 分组得到的 sample 数：`{len(sample_summaries)}`",
        f"- 预期每个 sample 的 rollout 数：`{ROLLOUTS_PER_SAMPLE}`",
        f"- 训练器 step 汇总行数：`{len(step_metrics)}`",
        f"- 根据 reward 行推导出的 step 数：`{math.ceil(len(sample_summaries) / SAMPLES_PER_STEP) if sample_summaries else 0}`",
        f"- 各 WorkerDict pid 的 reward 行数：`{dict(sorted(pid_counter.items()))}`",
        "",
    ]
    en_head = [
        "",
        f"- input_log: `{log_path}`",
        f"- step_limit: `{step_limit if step_limit is not None else 'not_set'}`",
        f"- raw_parsed_rollouts_before_step_filter: `{raw_parsed_rollout_count}`",
        f"- parsed_rollouts: `{parsed_rollout_count}`",
        f"- analyzed_complete_sample_rollouts: `{len(rows)}`",
        f"- dropped_incomplete_tail_rollouts: `{dropped_tail_rollouts}`",
        f"- total_samples_by_16_rollouts: `{len(sample_summaries)}`",
        f"- expected_rollouts_per_sample: `{ROLLOUTS_PER_SAMPLE}`",
        f"- step_summary_lines_found: `{len(step_metrics)}`",
        f"- derived_steps_from_reward_lines: `{math.ceil(len(sample_summaries) / SAMPLES_PER_STEP) if sample_summaries else 0}`",
        f"- pid_reward_line_counts: `{dict(sorted(pid_counter.items()))}`",
        "",
    ]

    zh_lines = ["# Task3 rollout 日志分析报告"]
    zh_lines.extend(zh_head)
    zh_lines.extend(format_metric_table("All-rollout 指标统计", all_metric_summary, zh=True))
    zh_lines.extend(format_metric_table("Best-of-sample 指标统计", best_metric_summary, zh=True))
    zh_lines.extend(format_rate_table("All-rollout rate 指标", "all_rollouts", zh=True))
    zh_lines.extend(format_rate_table("Best-of-sample rate 指标", "best_per_sample", zh=True))
    zh_lines.extend(format_rate_table("Sample-any rate 指标", "sample_any", zh=True))
    zh_lines.extend(format_poly_split_table("All-rollout: poly 指标按 label_reward 分组", all_poly_label_split, zh=True))
    zh_lines.extend(format_poly_split_table("Best-of-sample: poly 指标按 label_reward 分组", best_poly_label_split, zh=True))
    zh_lines.extend(
        [
        "## 完整性检查",
        "",
        f"- 不满 16 条 rollout 的 sample 数：`{len(incomplete_samples)}`",
        f"- 参与分析的 reward 行数除以 16 的余数：`{len(rows) % ROLLOUTS_PER_SAMPLE}`",
        f"- 按当前 sample 数推导出的 reward 行数：`{len(sample_summaries) * ROLLOUTS_PER_SAMPLE}`",
        f"- all-rollout 分桶统计行数：`{len(all_bins)}`",
        f"- best-of-sample 分桶统计行数：`{len(best_bins)}`",
        "",
        "## 输出资产",
        "",
        "- `all_rollouts/task3_rollout_rows.csv`：逐 rollout 明细，每条 Task3 reward 日志一行。",
        "- `all_rollouts/task3_metric_summary.csv`：all-rollout 视角下各 reward 小项的统计量。",
        "- `all_rollouts/task3_rate_summary.csv`：all-rollout 视角下的 rate 指标。",
        "- `all_rollouts/task3_reward_component_bins.csv`：all-rollout 视角下的分桶分布。",
        "- `all_rollouts/task3_poly_label_split.csv`：all-rollout 视角下按 label_reward 是否拿到拆开的 poly 指标。",
        "- `best_per_sample/task3_best_per_sample.csv`：每个 sample 只保留 score 最高的 rollout。",
        "- `best_per_sample/task3_metric_summary.csv`：best-of-sample 视角下各 reward 小项的统计量。",
        "- `best_per_sample/task3_rate_summary.csv`：best-of-sample 视角下的 rate 指标。",
        "- `best_per_sample/task3_reward_component_bins.csv`：best-of-sample 视角下的分桶分布。",
        "- `best_per_sample/task3_poly_label_split.csv`：best-of-sample 视角下按 label_reward 是否拿到拆开的 poly 指标。",
        "- `comparison/*.png`：all-rollout 与 best-of-sample 的叠加对比图。",
        "- `task3_sample_summary.csv`：每 16 条 rollout 聚合成一个 sample 的统计表。",
        "- `task3_step_summary_lines.csv`：从终端日志解析出的训练器 step 汇总。",
        "- 图表文件：`" + "`, `".join(plot_notes) + "`",
        "",
        "## 限制说明",
        "",
        "- 终端 reward 行没有打印模型预测的原始 `point_label`，只打印了 `label_reward`。因此本脚本能统计 label 是否正确，但不能直接统计模型偏向输出 label 0 还是 label 1。",
        f"- sample 分组假设 reward manager 会连续打印同一个 sample 的 16 条 rollout。本次快照在 step 截断后解析到 `{parsed_rollout_count}` 条 reward 行，其中 `{len(rows)}` 条进入完整 sample 分析，尾部 `{dropped_tail_rollouts}` 条不满 16 的 rollout 被丢弃。",
        "- 如果输入日志仍在被 `tee` 持续写入，脚本结果只代表运行时的快照。训练结束后需要重新运行脚本刷新资产。",
        ]
    )

    en_lines = ["# Task3 rollout log analysis report"]
    en_lines.extend(en_head)
    en_lines.extend(format_metric_table("All-rollout metric summary", all_metric_summary))
    en_lines.extend(format_metric_table("Best-of-sample metric summary", best_metric_summary))
    en_lines.extend(format_rate_table("All-rollout rates", "all_rollouts"))
    en_lines.extend(format_rate_table("Best-of-sample rates", "best_per_sample"))
    en_lines.extend(format_rate_table("Sample-any rates", "sample_any"))
    en_lines.extend(format_poly_split_table("All-rollout: poly metrics split by label_reward", all_poly_label_split))
    en_lines.extend(format_poly_split_table("Best-of-sample: poly metrics split by label_reward", best_poly_label_split))
    en_lines.extend(
        [
            "## Integrity checks",
            "",
            f"- incomplete_16_rollout_samples: `{len(incomplete_samples)}`",
            f"- analyzed_rollout_lines_mod_16: `{len(rows) % ROLLOUTS_PER_SAMPLE}`",
            f"- reward_lines_expected_if_4_samples_per_step: `{len(sample_summaries) * ROLLOUTS_PER_SAMPLE}`",
            f"- all_rollout_distribution_rows: `{len(all_bins)}`",
            f"- best_distribution_rows: `{len(best_bins)}`",
            "",
            "## Assets",
            "",
            "- `task3_rollout_rows.csv`: every parsed rollout line.",
            "- `task3_sample_summary.csv`: one row per 16-rollout sample.",
            "- `task3_best_per_sample.csv`: one best rollout per sample.",
            "- `task3_step_summary_lines.csv`: trainer step summary metrics parsed from terminal output.",
            "- `task3_metric_summary_all_rollouts.csv`: metric statistics for every rollout.",
            "- `task3_metric_summary_best_per_sample.csv`: metric statistics after keeping only the best rollout from each sample.",
            "- `task3_rate_summary.csv`: rate metrics for all-rollout, best-of-sample, and sample-any views.",
            "- `task3_reward_component_bins_all_rollouts.csv`: binned distributions for all rollouts.",
            "- `task3_reward_component_bins_best_per_sample.csv`: binned distributions for best rollouts.",
            "- `task3_poly_label_split_all_rollouts.csv`: poly near/inside rates split by whether label reward was obtained.",
            "- `task3_poly_label_split_best_per_sample.csv`: same split for best rollouts.",
            "- `all_rollouts/*.png`: all-rollout distribution plots.",
            "- `best_per_sample/*.png`: best-of-sample distribution plots.",
            "- plots: `" + "`, `".join(plot_notes) + "`",
            "",
            "## Limits",
            "",
            "- Terminal reward lines do not include the raw predicted point_label, only label_reward. This script can measure label correctness, but not the raw label-0/label-1 preference.",
            f"- Sample grouping assumes 16 consecutive rollouts belong to one sample. In this snapshot, `{parsed_rollout_count}` reward lines remained after the step filter; `{len(rows)}` complete-sample lines were analyzed and `{dropped_tail_rollouts}` incomplete trailing lines were dropped.",
            "- If the input log is still being written by `tee`, rerun this script after training stops.",
        ]
    )

    zh_text = "\n".join(zh_lines) + "\n"
    (out_dir / "task3_analysis_report.zh.md").write_text(zh_text, encoding="utf-8")
    (out_dir / "task3_analysis_report.en.md").write_text("\n".join(en_lines) + "\n", encoding="utf-8")


def analyze_log(log_path: Path, out_dir: Path, step_limit: int | None = None) -> dict[str, Any]:
    all_dir, best_dir, comparison_dir = prepare_output_dirs(out_dir)

    rows: list[dict[str, Any]] = []
    step_metrics: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            row = parse_task3_line(line, line_no, len(rows))
            if row is not None:
                rows.append(row)
                continue
            step = parse_step_line(line, line_no)
            if step is not None:
                step_metrics.append(step)

    raw_parsed_rollout_count = len(rows)
    rows, step_metrics = filter_rows_by_step(rows, step_metrics, step_limit)
    parsed_rollout_count = len(rows)
    dropped_tail_rollouts = parsed_rollout_count % ROLLOUTS_PER_SAMPLE
    if dropped_tail_rollouts:
        rows = rows[:-dropped_tail_rollouts]

    samples: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        samples[row["sample_id"]].append(row)
    sample_summaries = [summarize_sample(sample_id, sample_rows) for sample_id, sample_rows in sorted(samples.items())]
    best_rows = [best_row(sample_rows) for _, sample_rows in sorted(samples.items())]
    all_metric_summary = metric_summary_rows("all_rollouts", rows)
    best_metric_summary = metric_summary_rows("best_per_sample", best_rows)
    rate_summary = rate_summary_rows("all_rollouts", rows)
    rate_summary += rate_summary_rows("best_per_sample", best_rows)
    rate_summary += sample_any_rate_rows(sample_summaries)
    all_bins = bin_distribution_rows("all_rollouts", rows)
    best_bins = bin_distribution_rows("best_per_sample", best_rows)
    all_poly_label_split = poly_label_split_rows("all_rollouts", rows)
    best_poly_label_split = poly_label_split_rows("best_per_sample", best_rows)
    sample_any_rates = [row for row in rate_summary if row["view"] == "sample_any"]
    all_rates = [row for row in rate_summary if row["view"] == "all_rollouts"]
    best_rates = [row for row in rate_summary if row["view"] == "best_per_sample"]

    write_csv(out_dir / "task3_sample_summary.csv", sample_summaries)
    write_csv(out_dir / "task3_step_summary_lines.csv", step_metrics)
    write_csv(out_dir / "task3_sample_any_rate_summary.csv", sample_any_rates)
    write_csv(all_dir / "task3_rollout_rows.csv", rows)
    write_csv(all_dir / "task3_metric_summary.csv", all_metric_summary)
    write_csv(all_dir / "task3_rate_summary.csv", all_rates)
    write_csv(all_dir / "task3_reward_component_bins.csv", all_bins)
    write_csv(all_dir / "task3_poly_label_split.csv", all_poly_label_split)
    write_csv(best_dir / "task3_best_per_sample.csv", best_rows)
    write_csv(best_dir / "task3_metric_summary.csv", best_metric_summary)
    write_csv(best_dir / "task3_rate_summary.csv", best_rates)
    write_csv(best_dir / "task3_reward_component_bins.csv", best_bins)
    write_csv(best_dir / "task3_poly_label_split.csv", best_poly_label_split)
    plot_notes = write_plots(out_dir, all_dir, best_dir, comparison_dir, rows, best_rows)
    write_report(
        out_dir,
        log_path,
        parsed_rollout_count,
        raw_parsed_rollout_count,
        step_limit,
        dropped_tail_rollouts,
        rows,
        sample_summaries,
        best_rows,
        step_metrics,
        all_metric_summary,
        best_metric_summary,
        rate_summary,
        all_bins,
        best_bins,
        all_poly_label_split,
        best_poly_label_split,
        plot_notes,
    )

    return {
        "rows": rows,
        "sample_summaries": sample_summaries,
        "best_rows": best_rows,
        "step_metrics": step_metrics,
        "all_metric_summary": all_metric_summary,
        "best_metric_summary": best_metric_summary,
        "rate_summary": rate_summary,
        "all_poly_label_split": all_poly_label_split,
        "best_poly_label_split": best_poly_label_split,
        "total_rollouts": len(rows),
        "raw_parsed_rollouts": raw_parsed_rollout_count,
        "parsed_rollouts": parsed_rollout_count,
        "step_limit": step_limit,
        "dropped_tail_rollouts": dropped_tail_rollouts,
        "total_samples": len(sample_summaries),
        "output_dir": out_dir,
    }


def main() -> None:
    args = parse_args()
    output_dir = (
        resolve_output_dir(args.output_dir, args.step)
        if args.output_dir is not None
        else default_output_dir_for_log(args.input_log, args.output_root, args.step)
    )
    result = analyze_log(args.input_log, output_dir, args.step)
    print(f"input_log: {args.input_log}")
    print(f"step_limit: {args.step if args.step is not None else 'not set'}")
    print(f"raw parsed rollouts: {result['raw_parsed_rollouts']}")
    print(f"parsed rollouts: {result['parsed_rollouts']}")
    print(f"analyzed complete-sample rollouts: {result['total_rollouts']}")
    print(f"dropped incomplete tail rollouts: {result['dropped_tail_rollouts']}")
    print(f"parsed samples: {result['total_samples']}")
    print(f"output_dir: {result['output_dir']}")


if __name__ == "__main__":
    main()
