# SAM Checkpoint

Place the SAM 2.1 checkpoint used by the reward and evaluation workers in this directory. The default configuration expects a compatible Hiera checkpoint, for example:

```text
sam2/checkpoints/sam2.1_hiera_large.pt
```

Checkpoint files are excluded from Git. Download them from the official Segment Anything 2 release and verify that the selected YAML configuration under `sam2/configs/` matches the checkpoint architecture.
