export CUDA_VISIBLE_DEVICES=1,4
# export HF_ENDPOINT=https://hf-mirror.com
# [Optional] To use Weights & Biases logging, run `wandb login` first.
# To disable wandb logging, uncomment the line below:
# export WANDB_MODE=disabled

set -x

MODEL_PATH=pretrained_models/Qwen3-VL-8B-Instruct
RUN_NAME=AgenticSegRL_Qwen3VL8B/S2_$(date +%Y%m%d_%H%M%S)

# Set the path to your Stage 1 LoRA checkpoint
# e.g., checkpoints/AgenticSegRL_Qwen3VL8B_.../global_step_XXX/actor/lora_adapter
STAGE1_LORA_CHECKPOINT=${STAGE1_LORA_CHECKPOINT:-checkpoints/stage1/actor/lora_adapter}

python3 -m verl.trainer.main \
    algorithm.adv_estimator=grpo \
    worker.actor.loss_mode=grpo \
    config=configs/agentic_seg_rl.yaml \
    data.train_files=data/ReasonSegX_train \
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
    worker.actor.lora_checkpoint_path=${STAGE1_LORA_CHECKPOINT} \
    worker.actor.use_kl_loss=false \
    worker.actor.kl_loss_coef=0 \
    worker.actor.entropy_coeff=0 \
    worker.actor.optim.lr=5.0e-6 \
    worker.actor.optim.weight_decay=0.001 \
    worker.actor.global_batch_size=2 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.offload.param_offload=false \
    worker.rollout.use_self_correction=false \
    worker.rollout.tensor_parallel_size=2 \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.rollout.enable_chunked_prefill=false \
    worker.rollout.enforce_eager=false \
    worker.rollout.free_cache_engine=false \
    worker.rollout.n=64 \
    worker.rollout.m=16 \
    worker.rollout.layered_summon=true \
    worker.rollout.load_format=safetensors \
    worker.reward.compute_score=agentic_seg_s2 \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.total_episodes=10 \
    trainer.save_checkpoint_path=checkpoints/${RUN_NAME}
