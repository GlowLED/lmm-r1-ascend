#!/bin/bash
# =================== Single-NPU Development Script ===================
# For quick iteration on a dev machine with only 1 Ascend NPU.
# NOT for production training — batch sizes are minimal.
# =====================================================================

# =================== Ascend CANN Environment ===================
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
NNAL_LIB="/usr/local/Ascend/nnal/latest/lib64"
if [ -d "$NNAL_LIB" ] && [[ ":$LD_LIBRARY_PATH:" != *":$NNAL_LIB:"* ]]; then
    export LD_LIBRARY_PATH="$NNAL_LIB:${LD_LIBRARY_PATH}"
fi
# ===============================================================

# =================== User Configuration ===================
export WORKSPACE_DIR="$(pwd)"
export DATASET_PATH="${WORKSPACE_DIR}/data/deepscaler/deepscaler_message.jsonl"
export PRETRAIN_MODEL_PATH="${WORKSPACE_DIR}/models/Qwen2.5-VL-3B-Instruct"
export SAVE_PATH="${WORKSPACE_DIR}/checkpoints"
export MODEL_NAME="lmm-r1-fre-text-dev"
export WANDB_DIR="${WORKSPACE_DIR}"
# =========================================================

SCRIPT_PID=$$
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export LOG_DIR="${SAVE_PATH}/${MODEL_NAME}/logs"
export CUR_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"

# Stop any existing processes
pkill -f math_verifier 2>/dev/null || true
pkill -f openrlhf 2>/dev/null || true
sleep 1
ray stop 2>/dev/null || true

mkdir -p "${SAVE_PATH}/${MODEL_NAME}"
mkdir -p "${LOG_DIR}"
mkdir -p "${CUR_LOG_DIR}"

echo "================================================================"
echo "LMM-R1 FRE-Text Training (Single NPU Dev)"
echo "================================================================"
echo "Model: ${PRETRAIN_MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Logs: ${CUR_LOG_DIR}"
echo "================================================================"

# Start ray with 1 GPU
echo "Starting ray..."
ray start --head --node-ip-address 0.0.0.0 --num-gpus 1 --temp-dir ~/.cache/ray
sleep 5

echo "Starting remote reward model server..."
python -m openrlhf.models.remote_rm.math_verifier \
    --input_key message \
    --prompt-template chatml 2>&1 | tee -a "${CUR_LOG_DIR}/remote_rm.log" &
REMOTE_RM_PID=$!
sleep 5

echo "Starting training..."
export RAY_ADDRESS="0.0.0.0:6379"
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

python -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 1 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 1 \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.5 \
   --vllm_sync_backend gloo \
   --enable_prefix_caching \
   --pretrain ${PRETRAIN_MODEL_PATH} \
   --save_path ${SAVE_PATH}/${MODEL_NAME} \
   --micro_train_batch_size 1 \
   --train_batch_size 8 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 8 \
   --temperature 1.0 \
   --n_samples_per_prompt 2 \
   --max_epochs 1 \
   --num_episodes 1 \
   --prompt_max_len 2048 \
   --max_samples 100 \
   --generate_max_len 2048 \
   --advantage_estimator reinforce_baseline \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 4e-7 \
   --init_kl_coef 0.001 \
   --prompt_data ${DATASET_PATH} \
   --input_key message \
   --label_key "answer" \
   --normalize_reward \
   --lambd 1 \
   --gamma 1 \
   --gradient_checkpointing \
   --save_steps 20 \
   --ckpt_path ${SAVE_PATH}/${MODEL_NAME}/ckpt \
   --save_hf_ckpt \
   --use_tensorboard ${LOG_DIR} > >(tee -a "${CUR_LOG_DIR}/train.log") 2>&1 &

TRAIN_PID=$!

echo "Remote RM PID: $REMOTE_RM_PID" > "${CUR_LOG_DIR}/process_pids.txt"
echo "Train PID: $TRAIN_PID" >> "${CUR_LOG_DIR}/process_pids.txt"

echo "Training running. Check: tail -f ${CUR_LOG_DIR}/train.log"
