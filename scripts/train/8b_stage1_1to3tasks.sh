# bash ./scripts/train/8b_stage1_1to3tasks.sh 2>&1 | tee output_05190144.txt

# PYTHONUNBUFFERED=1 RAY_DEDUP_LOGS=0 stdbuf -oL -eL \
# bash ./scripts/train/8b_stage1_1to3tasks.sh \
# 2>&1 | tee >(sed -r 's/\x1B\[[0-9;]*[mK]//g' > output_05222254.txt)

# Override TMPDIR externally when Ray needs a larger temporary volume.

# debug 有多卡通信问题
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
# export TORCH_NCCL_DUMP_ON_TIMEOUT=1
# export TORCH_NCCL_DESYNC_DEBUG=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 这次跑的是，之前单独训task1后的权重，然后现在串行训练task3

# 正式执行
export CUDA_VISIBLE_DEVICES=0,3,5,7  # 要用同类型显卡
export HF_ENDPOINT=https://hf-mirror.com
# [Optional] To use Weights & Biases logging, run `wandb login` first.
# To disable wandb logging, uncomment the line below:
# export WANDB_MODE=disabled

set -x

MODEL_PATH=pretrained_models/Qwen3-VL-8B-Instruct
# 混合任务运行名称
RUN_NAME=AgenticSegRL_Qwen3VL8B_1to3/$(date +%Y%m%d_%H%M%S)

# 核心：当前任务数据集
TRAIN_FILES=data/hf_agentic/base_dataset/task3
# TRAIN_FILES=data/hf_agentic/base_dataset/task2


# 上一阶段 Task 1 训练完的最佳权重路径 (指向 lora_adapter 目录)
CKPT_PATH=${CKPT_PATH:-checkpoints/stage1/actor/lora_adapter}

python3 -m verl.trainer.main \
    algorithm.adv_estimator=grpo \
    worker.actor.loss_mode=grpo \
    config=configs/agentic_seg_rl.yaml \
    data.train_files=${TRAIN_FILES} \
    data.seed=42 \
    data.val_files=None \
    data.rollout_batch_size=4 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.lora_checkpoint_path=${CKPT_PATH} \
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
    worker.actor.global_batch_size=4 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.offload.param_offload=false \
    worker.actor.offload.optimizer_offload=false \
    worker.rollout.tensor_parallel_size=4 \
    worker.rollout.gpu_memory_utilization=0.3 \
    worker.rollout.enable_chunked_prefill=true \
    worker.rollout.enforce_eager=false \
    worker.rollout.free_cache_engine=false \
    worker.rollout.n=16 \
    worker.rollout.layered_summon=false \
    worker.reward.compute_score=agentic_seg \
    trainer.agentic_viz.enabled=true \
    trainer.agentic_viz.task=task3 \
    trainer.agentic_viz.log_every_steps=10 \
    trainer.agentic_viz.max_samples=1 \
    trainer.agentic_viz.max_rollouts_per_sample=8 \
    trainer.agentic_viz.seed=42 \
    trainer.agentic_viz.save_local=true \
    trainer.agentic_viz.log_to_wandb=true \
    trainer.agentic_viz.output_dir=agentic_rollouts \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.total_episodes=1 \
    trainer.save_checkpoint_path=checkpoints/${RUN_NAME}

    # 实在没显存才开 不然速度慢
    # worker.rollout.layered_summon=true \
    # worker.rollout.load_format=safetensors \
    
    # 建议开 2 这才是性价比
    # worker.rollout.tensor_parallel_size=2 \
