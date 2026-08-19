# Pretrained Models

Place locally downloaded vision-language models in this directory. Model weights are excluded from Git.

Recommended layout:

```text
pretrained_models/
|-- Qwen3-VL-8B-Instruct/
|-- Qwen3-VL-4B-Instruct/
`-- reasoning-model/
```

Download each model from its official provider and ensure the directory contains the Hugging Face configuration, tokenizer or processor files, and weight shards. Update `MODEL_PATH` in the selected training launcher when using a different location.
