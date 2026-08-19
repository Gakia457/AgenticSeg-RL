# Data Directory

This directory is reserved for local datasets and generated training assets. Dataset files are intentionally excluded from Git because of their size and redistribution requirements.

Expected layout:

```text
data/
|-- base_segmentation_train/
|-- ReasonSegX_train/
|-- ReasonSegX_val/
|-- ReasonSegX_test/
|-- ReasonSeg_refine/
|-- MMR/
|-- MUSE/
|-- agenticrl/
|   |-- task2_mask_understanding/
|   `-- task3_self_correction/
`-- hf_agentic/
    `-- base_dataset/
        |-- task2/
        `-- task3/
```

Use the utilities in `tools/benchmark_preparation/` to prepare supported benchmarks. Use `tools/datasets/` to generate Task 2 and Task 3 assets and convert them into Hugging Face dataset format.

Do not commit Arrow shards, images, masks, manifests containing private paths, or licensed datasets to this repository.
