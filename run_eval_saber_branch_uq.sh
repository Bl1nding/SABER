#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$PROJECT_ROOT"

# ===============================
# 1. Hardware & Model
# ===============================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"

MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-Distill-Qwen-7B}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
BASE_SAVE_DIR="${BASE_SAVE_DIR:-${PROJECT_ROOT}/results/${MODEL_NAME}}"

# ===============================
# 2. Dataset
# ===============================
DATASETS=(
    "gsm8k"
    "math_500"
    "aime_24"
    "aime_25"
    "amc23"
    "olympiadbench"
    "gpqa"
)

# ===============================
# 3. Hyper-parameters
# ===============================

# Branch uncertainty difference thresholds
BRANCH_UQ_DIFF_THRESHOLDS=(0.1)
# ===============================
# 4. Running Config
# ===============================

MAX_EXAMPLES=-1
MIN_STEP=0

MAX_TOKENS=16384
MAX_MODEL_LEN=32768

CONCURRENCY=32

# ===============================
# Main Loop
# ===============================

for DATA in "${DATASETS[@]}"; do

    echo "========================================"
    echo "Dataset : $DATA"
    echo "========================================"

    if [[ "$DATA" == "aime_24" || "$DATA" == "aime_25" || "$DATA" == "aime2425" || "$DATA" == "amc23" ]]; then
        CUR_RUNS=4
    else
        CUR_RUNS=1
    fi

    echo "Runs = $CUR_RUNS"

    for BRANCH_UQ_DIFF_TAU in "${BRANCH_UQ_DIFF_THRESHOLDS[@]}"; do

            echo "----------------------------------------"
            echo "Branch-UQ Diff Threshold : $BRANCH_UQ_DIFF_TAU"
            echo "----------------------------------------"

            for ((RUN_ID=1; RUN_ID<=CUR_RUNS; RUN_ID++)); do

                SAVE_DIR="${BASE_SAVE_DIR}/${DATA}/branch_uq_diff_${BRANCH_UQ_DIFF_TAU}/run_${RUN_ID}"

                mkdir -p "$SAVE_DIR"

                "$PYTHON_BIN" -m src.inference_vllm_branch_uq \
                    --model_path "$MODEL_PATH" \
                    --model_name "$MODEL_NAME" \
                    --data_name "$DATA" \
                    --save_path "$SAVE_DIR" \
                    --min_step_tokens $MIN_STEP \
                    --max_tokens $MAX_TOKENS \
                    --max_model_len $MAX_MODEL_LEN \
                    --max_example $MAX_EXAMPLES \
                    --concurrency $CONCURRENCY \
                    --temperature 0.6 \
                    --top_p 0.95 \
                    --probe_n 4 \
                    --branch_uq_diff_threshold $BRANCH_UQ_DIFF_TAU \
                    --runid $RUN_ID

                echo "Finished | Dataset=$DATA | Branch-UQ Diff=$BRANCH_UQ_DIFF_TAU | Run=$RUN_ID"

            done

    done

    echo "========================================"
    echo "$DATA Finished."
    echo "========================================"

done

echo "All experiments finished."
