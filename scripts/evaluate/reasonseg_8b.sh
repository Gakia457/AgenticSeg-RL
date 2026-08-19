#!/bin/bash

# export HF_ENDPOINT=https://hf-mirror.com

BASE_MODEL_PATH=${BASE_MODEL_PATH:-pretrained_models/Qwen3-VL-8B-Instruct}
REASONING_MODEL_PATH=${REASONING_MODEL_PATH:-pretrained_models/reasoning-model}

# ---------------------------------------------------------------------------------------------- #
# 自动从模型路径解析短输出目录：
# 目标格式：outputs/<模型名>/<月日_时分>/<step>/<测试集名>/
# 例如：outputs/AgenticSegRL_Qwen3VL8B/0428_1831/400/ReasonSegX_val/
REASONING_PATH_NORM="${REASONING_MODEL_PATH%/}"
OUTPUT_SUBDIR=""
if [[ "$REASONING_PATH_NORM" =~ /([^/]+)/((S[0-9]+_)?([0-9]{8}_[0-9]{6}))(/global_step_([0-9]+))?(/|$) ]]; then
    MODEL_NAME_SHORT="${BASH_REMATCH[1]}"
    STAGE_PREFIX_RAW="${BASH_REMATCH[3]}"   # e.g. "S2_" or ""
    RUN_TIME_RAW="${BASH_REMATCH[4]}"       # e.g. "20260429_195037"
    STEP_SHORT="${BASH_REMATCH[6]}"         # e.g. "20" (may be empty)
    [ -z "$STEP_SHORT" ] && STEP_SHORT="na"
    RUN_TIME_SHORT="${RUN_TIME_RAW:4:4}_${RUN_TIME_RAW:9:4}"  # MMDD_HHMM
    # 统一处理 S1/S2：有 S2_ 前缀时保留；无前缀时就是普通时间目录
    OUTPUT_SUBDIR="${MODEL_NAME_SHORT}/${STAGE_PREFIX_RAW}${RUN_TIME_SHORT}/${STEP_SHORT}"
else
    echo "[WARN] 无法从 REASONING_MODEL_PATH 解析短目录：$REASONING_MODEL_PATH"
    echo "[WARN] 已回退为直接拼接路径。"
    OUTPUT_SUBDIR="${REASONING_PATH_NORM}"
fi
MODEL_DIR="$OUTPUT_SUBDIR"
# 这一大串 本质上是 代替原本的下面这一行
# MODEL_DIR=$(echo $REASONING_MODEL_PATH | sed -E 's/.*pretrained_models\/(.*)\/actor\/.*/\1/')
# ---------------------------------------------------------------------------------------------- #


# TEST_DATA_PATH="data/ReasonSegX_val"
TEST_DATA_PATH=${TEST_DATA_PATH:-data/ReasonSegX_test}

TEST_NAME=$(echo $TEST_DATA_PATH | sed -E 's/.*\/([^\/]+)$/\1/')
OUTPUT_PATH="./outputs/${MODEL_DIR}/${TEST_NAME}"

# MODE="single"  # 单卡推理（只起1个分片进程，idx 固定为 0）
MODE="multi" # 多卡推理（每卡1个分片进程，idx 自动按 0..N-1 分配）

# 物理卡号修改
SINGLE_GPU_ID=3
# MULTI_GPU_IDS=(4 5 6 7)
MULTI_GPU_IDS=(4 6)

# 常改参数
# VISUALIZATION=true
VISUALIZATION=false
USE_MAJORITY_VOTING=false


# 分割线
##########################################################################################

if [ "$MODE" = "single" ]; then
    GPU_IDS=($SINGLE_GPU_ID)
    NUM_PARTS=1
elif [ "$MODE" = "multi" ]; then
    GPU_IDS=("${MULTI_GPU_IDS[@]}")
    NUM_PARTS=${#GPU_IDS[@]}
else
    echo "Invalid MODE: $MODE (must be 'single' or 'multi')"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_PATH"

echo "REASONING_MODEL_PATH: $REASONING_MODEL_PATH"
echo "OUTPUT_PATH: '$OUTPUT_PATH'"
echo "MODE: $MODE"
echo "GPU_IDS: ${GPU_IDS[*]}"
echo "NUM_PARTS: $NUM_PARTS"

# Run n processes in parallel
for idx in "${!GPU_IDS[@]}"; do
    gpu_id=${GPU_IDS[$idx]}
    export CUDA_VISIBLE_DEVICES=$gpu_id
    python tools/evaluation/evaluate_reasoning_segmentation.py \
        --reasoning_model_path $REASONING_MODEL_PATH \
        --vl_model_version qwen3 \
        --qwen3_base_path $BASE_MODEL_PATH \
        --output_path $OUTPUT_PATH \
        --test_data_path $TEST_DATA_PATH \
        --idx $idx \
        --num_parts $NUM_PARTS \
        --use_lora true \
        --dump_analysis_fields true \
        --visualization $VISUALIZATION \
        --use_majority_voting $USE_MAJORITY_VOTING \
        --num_samples 32 \
        --sampling_temperature 1.0 \
        --batch_size 32 &
done

# Wait for all processes to complete
wait

echo "MODEL_DIR: '$MODEL_DIR'"
echo "Test data path: '$TEST_DATA_PATH'"

python tools/evaluation/calculate_iou.py --output_dir $OUTPUT_PATH || exit 1
# python tools/evaluation/calculate_sweep.py --output_dir $OUTPUT_PATH
# visualization=true 时，才运行我的自动分析脚本
[ "$VISUALIZATION" = "true" ] && python tools/evaluation/analyze_cases.py --output_dir $OUTPUT_PATH --top_k 50
