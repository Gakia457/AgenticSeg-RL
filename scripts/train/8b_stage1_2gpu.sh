export CUDA_VISIBLE_DEVICES=1,4
# export HF_ENDPOINT=https://hf-mirror.com
# [Optional] To use Weights & Biases logging, run `wandb login` first.
# To disable wandb logging, uncomment the line below:
# export WANDB_MODE=disabled

set -x

MODEL_PATH=pretrained_models/Qwen3-VL-8B-Instruct
RUN_NAME=AgenticSegRL_Qwen3VL8B/$(date +%Y%m%d_%H%M%S)

python3 -m verl.trainer.main \
    algorithm.adv_estimator=grpo \
    worker.actor.loss_mode=grpo \
    config=configs/agentic_seg_rl.yaml \
    data.train_files=data/base_segmentation_train \
    data.seed=42 \
    data.val_files=None \
    data.rollout_batch_size=2 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.lora_type=lora \
    worker.actor.model.lora_rank=64 \
    worker.actor.model.lora_alpha=64 \
    worker.actor.model.target_modules=all-linear \
    worker.actor.model.exclude_modules='.*visual.*' \
    worker.actor.model.enable_gradient_checkpointing=true \
    worker.actor.use_kl_loss=false \
    worker.actor.kl_loss_coef=0 \
    worker.actor.entropy_coeff=0 \
    worker.actor.optim.lr=1.0e-5 \
    worker.actor.optim.weight_decay=0.001 \
    worker.actor.global_batch_size=2 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.offload.param_offload=false \
    worker.actor.offload.optimizer_offload=false \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.rollout.enable_chunked_prefill=true \
    worker.rollout.enforce_eager=false \
    worker.rollout.free_cache_engine=false \
    worker.rollout.n=16 \
    worker.rollout.layered_summon=true \
    worker.rollout.load_format=safetensors \
    worker.reward.compute_score=agentic_seg \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.total_episodes=1 \
    trainer.save_checkpoint_path=checkpoints/${RUN_NAME}


    # 开启summon 很关键
    # worker.rollout.layered_summon=true \
    # worker.rollout.load_format=safetensors \

    # 有summon之后rollout.n可以安心开到8 甚至是16
    # worker.rollout.n=8 \
    # 这些限制也可以用默认值了 不用调这么小
    # data.max_response_length=512 \
    # data.max_prompt_length=1024 \
    # 这些都可以关
    # export PYTORCH_ALLOC_CONF=expandable_segments:True
    # worker.actor.offload.param_offload=true \
    # worker.actor.offload.optimizer_offload=true \
