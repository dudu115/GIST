#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source /path/to/conda/etc/profile.d/conda.sh
PYTHON_BIN="$(which python)"

setting=''
export CUDA_VISIBLE_DEVICES=
export HF_HOME=/path/to/huggingface/cache
export OMP_NUM_THREADS=8
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Clearing GPU cache..."
"${PYTHON_BIN}" -c "import torch; torch.cuda.empty_cache()"

OUTPUT_DIR="/path/to/output/$setting"

echo "Deep cleaning disk space..."
rm -rf $OUTPUT_DIR 2>/dev/null
rm -rf /path/to/other/output/* 2>/dev/null
if [ "$HF_HOME" != "~/.cache/huggingface" ]; then
    echo "Preserving new cache directory: $HF_HOME"
    rm -rf ~/.cache/huggingface/hub/models--* 2>/dev/null
fi

find /tmp -name "*.tmp" -delete 2>/dev/null || true
find "${PROJECT_ROOT}" -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null || true

mkdir -p $OUTPUT_DIR
mkdir -p $HF_HOME/hub

LOCAL_MODEL_PATH="/path/to/local/models/Qwen/Qwen3-VL-4B-Instruct"
if [ -d "$LOCAL_MODEL_PATH" ]; then
    echo "Using locally downloaded model from $LOCAL_MODEL_PATH"
    MODEL_PATH="$LOCAL_MODEL_PATH"
else
    echo "Using model from Hugging Face Hub: Qwen/Qwen3-VL-4B-Instruct"
    MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
fi

df -h | grep /path/to/tmp || true

echo "Available space before training:"
df -h /path/to/tmp | awk 'NR==2 {print $4}'

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m accelerate.commands.launch \
    --config_file "${PROJECT_ROOT}/accelerate_configs/deepspeed_zero3.yaml" \
    --num_processes 2 \
    --main_process_port 20092 \
    "${PROJECT_ROOT}/grpo-gr/GRPO_GR.py" \
    --train_data_path "/path/to/data/chartqa_train.jsonl,/path/to/data/gqa_train.jsonl,/path/to/data/mathvision_train.jsonl,/path/to/data/mathvista_train.jsonl,/path/to/data/mme_train.jsonl,/path/to/data/tallyqa_train.jsonl,/path/to/data/vsr_train.jsonl" \
    --train_image_folder_path "/path/to/data/chartqa,/path/to/data/gqa,/path/to/data/mathvision,/path/to/data/mathvista,/path/to/data/mme,/path/to/data/tallyqa,/path/to/data/vsr" \
    --eval_data_path "/path/to/data/chartqa_val.jsonl,/path/to/data/gqa_val.jsonl,/path/to/data/mathvision_val.jsonl,/path/to/data/mathvista_val.jsonl,/path/to/data/mme_val.jsonl,/path/to/data/tallyqa_val.jsonl,/path/to/data/vsr_val.jsonl" \
    --eval_image_folder_path "/path/to/data/chartqa,/path/to/data/gqa,/path/to/data/mathvision,/path/to/data/mathvista,/path/to/data/mme,/path/to/data/tallyqa,/path/to/data/vsr" \
    --setting $setting \
    --max_turns 2 \
    --output_dir $OUTPUT_DIR \
    --project_root_path "${PROJECT_ROOT}" \
    --python_path_for_dino $(which python) \
    --dataset_name rr \
    --learning_rate 3e-6 \
    --num_generations 6 \
    --phase2_num_generations 3 \
    --eval_num_generations 1 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --per_device_eval_batch_size 9 \
    --generation_batch_size 24 \
    --temperature 0.7 \
    --torch_empty_cache_steps 5 \
    --gradient_checkpointing True \
    --eval_strategy steps \
    --eval_steps 2 \
    --save_steps 10 \
    --save_total_limit 1 \
    --num_train_epochs 100 \
    --max_completion_length 1024 \
    --max_prompt_length 4096 \
    --bf16 True \
    --bf16_full_eval True \
    --run_name  \
    --logging_steps 1 \
    --beta 0.01 \
    --phase2_no_image_steps 0 \
    --log_completions True \
    --model_name_or_path "$MODEL_PATH" \


if [ ! -d "$LOCAL_MODEL_PATH" ]; then
    echo "\nTo speed up future model loading, you can download the model to local directory with:"
    echo "git lfs install"
    echo "git clone https://huggingface.co/$MODEL_PATH $LOCAL_MODEL_PATH"
fi
