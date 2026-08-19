# Agentic Task Layer

This package contains the task-specific layer used by AgenticSeg-RL.

## Routing

`router.py` inspects each sample and selects one of three reward paths:

- Task 1 falls through to the SAM-backed localization reward.
- Task 2 uses `reward_task2.py` for mask-quality classification.
- Task 3 uses `reward_task3.py` for error-label and correction-point scoring.

Unknown or malformed task metadata fails explicitly instead of silently selecting an unrelated reward.

## Task 2 Reward

Task 2 expects a structured decision indicating whether a candidate mask is acceptable or requires refinement. The scorer validates the response format and compares the predicted decision with dataset metadata.

## Task 3 Reward

Task 3 expects an error label and a correction point. The scorer combines format validity, label correctness, geometric proximity, and changed-region overlap. Geometry is evaluated against normalized image coordinates stored by the dataset builder.

## Rollout Visualization

`rollout_viz.py` renders selected samples and rollout predictions during training. Visualization can be saved locally, sent to W&B, or disabled without changing reward computation.

Relevant trainer configuration keys live under `trainer.agentic_viz` in `verl/trainer/config.py`. Example launchers are available in `scripts/train/`.

## Tests

```bash
pytest -q verl/agentic/test_integration.py verl/agentic/test_rollout_viz.py
```
