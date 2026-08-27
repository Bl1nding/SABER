#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
cd "$PROJECT_ROOT"

# Hardware and model
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-Distill-Qwen-7B}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
BASE_SAVE_DIR="${BASE_SAVE_DIR:-${PROJECT_ROOT}/results/${MODEL_NAME}}"
DATASETS=(
    "gsm8k"
    "math_500"
    "aime_24"
    "aime_25"
    "amc23"
    "olympiadbench"
    "gpqa"
)

ALPHAS=(0.3)
TAUS=(0.9)

# Runtime settings
MAX_EXAMPLES=-1
MIN_STEP=0
MAX_TOKENS=16384
MAX_MODEL_LEN=32768
CONCURRENCY=32
# ===============================
# Main loop
# ===============================
for DATA in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Evaluating dataset: $DATA"
    echo "========================================"

    if [[ "$DATA" == "aime_24" || "$DATA" == "aime_25" || "$DATA" == "aime2425" || "$DATA" == "amc23" ]]; then
        CUR_RUNS=4
    else
        CUR_RUNS=1
    fi

    echo "Runs = $CUR_RUNS"

    for ALPHA in "${ALPHAS[@]}"; do
        for TAU in "${TAUS[@]}"; do

            echo "------ alpha=$ALPHA | tau=$TAU ------"

            for ((RUN_ID=1; RUN_ID<=CUR_RUNS; RUN_ID++)); do
                echo "Run $RUN_ID / $CUR_RUNS"


                SAVE_DIR="${BASE_SAVE_DIR}/${DATA}/alpha_${ALPHA}_tau_${TAU}/run_${RUN_ID}"
                mkdir -p "$SAVE_DIR"

                "$PYTHON_BIN" -m src.inference_vllm_saber \
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
                    --rss_threshold $TAU \
                    --probe_n 4 \
                    --alpha $ALPHA \
                    --runid $RUN_ID

                echo "Finished: $DATA | alpha=$ALPHA | tau=$TAU | run=$RUN_ID"
            done

        done
    done

    echo "Completed all runs for $DATA."
done
