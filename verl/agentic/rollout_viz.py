import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw

from verl.agentic.router import infer_task_type
from verl.agentic.reward_task3 import analyze_task3_rollout


@dataclass(frozen=True)
class SelectedRollout:
    sample_index: int
    rollout_index: int
    flat_index: int


def should_log_step(global_step: int, log_every_steps: int) -> bool:
    return log_every_steps > 0 and global_step > 0 and global_step % log_every_steps == 0


def select_rollout_indices(
    total_rollouts: int,
    rollout_n: int,
    max_samples: int,
    max_rollouts_per_sample: int,
    global_step: int,
    seed: int = 0,
) -> List[SelectedRollout]:
    if total_rollouts <= 0 or rollout_n <= 0:
        return []

    num_samples = total_rollouts // rollout_n
    if num_samples <= 0:
        return []

    rng = random.Random(seed + global_step)
    sample_count = min(max(1, max_samples), num_samples)
    rollout_count = min(max(1, max_rollouts_per_sample), rollout_n)

    sample_indices = rng.sample(range(num_samples), sample_count)
    selected: List[SelectedRollout] = []
    for sample_index in sample_indices:
        rollout_indices = rng.sample(range(rollout_n), rollout_count)
        for rollout_index in sorted(rollout_indices):
            selected.append(
                SelectedRollout(
                    sample_index=sample_index,
                    rollout_index=rollout_index,
                    flat_index=sample_index * rollout_n + rollout_index,
                )
            )
    return selected


class AgenticRolloutVisualizer:
    def __init__(self, config, tokenizer, is_qwen3: bool, run_dir: str):
        self.config = config
        self.tokenizer = tokenizer
        self.is_qwen3 = is_qwen3
        self.run_dir = run_dir
        self.output_dir = self._resolve_output_dir()

    def _resolve_output_dir(self) -> str:
        output_dir = self.config.output_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(self.run_dir, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def log_step(self, batch, logger, global_step: int, rollout_n: int) -> None:
        if not self.config.enabled:
            return
        if not should_log_step(global_step, self.config.log_every_steps):
            return

        selected = select_rollout_indices(
            total_rollouts=len(batch),
            rollout_n=rollout_n,
            max_samples=self.config.max_samples,
            max_rollouts_per_sample=self.config.max_rollouts_per_sample,
            global_step=global_step,
            seed=self.config.seed,
        )
        if not selected:
            return

        step_dir = os.path.join(self.output_dir, f"step_{global_step:06d}")
        if self.config.save_local:
            os.makedirs(step_dir, exist_ok=True)

        rows = []
        for item in selected:
            record = self._build_record(batch[item.flat_index], item, global_step, step_dir)
            if record is not None:
                rows.append(record)
        if not rows:
            return

        if self.config.save_local:
            jsonl_path = os.path.join(step_dir, "rollouts.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for record in rows:
                    f.write(json.dumps(self._json_safe_record(record), ensure_ascii=False) + "\n")

        if self.config.log_to_wandb:
            self._log_wandb(rows, logger, global_step)

    def _build_record(self, data_item, selected: SelectedRollout, global_step: int, step_dir: str) -> Optional[Dict[str, Any]]:
        prompt_str, response_str = self._decode_prompt_response(data_item)
        ground_truth = data_item.non_tensor_batch.get("solution")
        image = data_item.non_tensor_batch.get("image")
        problem = data_item.non_tensor_batch.get("problem", "")
        task_type = infer_task_type(ground_truth)
        if self.config.task not in ("auto", task_type):
            return None

        record: Dict[str, Any] = {
            "step": global_step,
            "task_type": task_type,
            "sample_index": selected.sample_index,
            "rollout_index": selected.rollout_index,
            "flat_index": selected.flat_index,
            "problem": problem,
            "prompt": prompt_str,
            "response": response_str,
            "ground_truth": ground_truth,
        }

        if task_type == "task3":
            analysis = analyze_task3_rollout(response_str, ground_truth, image, self.is_qwen3)
            record.update(analysis)
            if self.config.save_local and image is not None:
                image_path = os.path.join(
                    step_dir,
                    f"sample_{selected.sample_index:02d}_rollout_{selected.rollout_index:02d}.png",
                )
                self._draw_task3_image(image, analysis).save(image_path)
                record["image_path"] = image_path
        else:
            record["reserved_interface"] = True
            if self.config.save_local and image is not None:
                image_path = os.path.join(
                    step_dir,
                    f"{task_type}_sample_{selected.sample_index:02d}_rollout_{selected.rollout_index:02d}.png",
                )
                image.save(image_path)
                record["image_path"] = image_path

        return record

    def _decode_prompt_response(self, data_item) -> tuple[str, str]:
        prompt_ids = data_item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        return prompt_str, response_str

    def _draw_task3_image(self, image: Image.Image, analysis: Dict[str, Any]) -> Image.Image:
        canvas = image.convert("RGBA")
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        gt_items = analysis.get("ground_truth") or []
        for gt in gt_items:
            for poly in _normalize_polygons(gt.get("polygon")):
                draw.polygon([tuple(p) for p in poly], outline=(255, 80, 80, 255), fill=(255, 80, 80, 45))
            for center in _normalize_points(gt.get("polygon_centers")):
                _draw_cross(draw, center, (120, 0, 220, 255), size=6)
            gt_point = gt.get("point_2d")
            if isinstance(gt_point, list) and len(gt_point) == 2:
                _draw_cross(draw, gt_point, (0, 200, 0, 255), size=7)

        for point, label in zip(analysis.get("pred_points", []), analysis.get("pred_labels", [])):
            color = (40, 120, 255, 255) if label == 1 else (255, 190, 0, 255)
            _draw_circle(draw, point, color, radius=6)

        return Image.alpha_composite(canvas, overlay).convert("RGB")

    def _log_wandb(self, rows: List[Dict[str, Any]], logger, global_step: int) -> None:
        try:
            import wandb  # type: ignore
        except Exception:
            return

        columns = [
            "step",
            "task_type",
            "sample_index",
            "rollout_index",
            "problem",
            "visualization",
            "response",
            "score",
            "format_reward",
            "accuracy_reward",
            "label_reward",
            "proximity_reward",
            "region_hit",
            "signed_distance",
            "center_distance",
            "best_pred_point",
            "best_pred_label",
            "matched_center",
            "matched_polygon_index",
            "ground_truth",
        ]
        table = wandb.Table(columns=columns)
        for record in rows:
            image_obj = None
            if record.get("image_path"):
                image_obj = wandb.Image(record["image_path"])
            table.add_data(
                record.get("step"),
                record.get("task_type"),
                record.get("sample_index"),
                record.get("rollout_index"),
                record.get("problem"),
                image_obj,
                record.get("response"),
                record.get("score"),
                record.get("format_reward"),
                record.get("accuracy_reward"),
                record.get("label_reward"),
                record.get("proximity_reward"),
                record.get("region_hit"),
                record.get("signed_distance"),
                record.get("center_distance"),
                record.get("best_pred_point"),
                record.get("best_pred_label"),
                record.get("matched_center"),
                record.get("matched_polygon_index"),
                record.get("ground_truth"),
            )
        logger.log({"agentic/task_rollouts": table}, step=global_step, backend=["wandb"])

    def _json_safe_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        safe = dict(record)
        safe.pop("prompt", None)
        return safe


def _normalize_polygons(polygons) -> List[List[List[float]]]:
    if not isinstance(polygons, list) or not polygons:
        return []
    if isinstance(polygons[0], list) and polygons[0] and isinstance(polygons[0][0], (int, float)):
        polygons = [polygons]

    normalized = []
    for poly in polygons:
        if isinstance(poly, list) and len(poly) >= 3:
            points = []
            for point in poly:
                if isinstance(point, list) and len(point) == 2:
                    points.append([float(point[0]), float(point[1])])
            if len(points) >= 3:
                normalized.append(points)
    return normalized


def _normalize_points(points) -> List[List[float]]:
    if not isinstance(points, list):
        return []
    normalized = []
    for point in points:
        if isinstance(point, list) and len(point) == 2:
            normalized.append([float(point[0]), float(point[1])])
    return normalized


def _draw_circle(draw: ImageDraw.ImageDraw, point, color, radius: int) -> None:
    x, y = float(point[0]), float(point[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)


def _draw_cross(draw: ImageDraw.ImageDraw, point, color, size: int) -> None:
    x, y = float(point[0]), float(point[1])
    draw.line((x - size, y, x + size, y), fill=color, width=3)
    draw.line((x, y - size, x, y + size), fill=color, width=3)
