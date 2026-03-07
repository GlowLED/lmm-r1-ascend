#!/bin/bash
# =================== Single-NPU Development Script ===================
# Based on train_fre_text_modelmate.sh, adapted for 1-NPU dev machine.
# NOT for production training — batch sizes are minimal.
# =====================================================================

# =================== 强制环境重置 (防平台劫持) ===================
export PATH="/usr/local/bin:$PATH"
unset PYTHONPATH
unset LD_PRELOAD

# 强制加载华为昇腾 CANN 的环境变量 (Ray 依赖底层的 acl 库)
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
# NNAL (Neural Network Acceleration Library) provides libatb.so
NNAL_LIB="/usr/local/Ascend/nnal/latest/lib64"
if [ -d "$NNAL_LIB" ] && [[ ":$LD_LIBRARY_PATH:" != *":$NNAL_LIB:"* ]]; then
    export LD_LIBRARY_PATH="$NNAL_LIB:${LD_LIBRARY_PATH}"
fi

PYTHON_EXEC="/usr/local/bin/python3.11"
RAY_EXEC="/usr/local/bin/ray"
# =================================================================

# =================== User Configuration ===================
export WORKSPACE_DIR="$(pwd)"
export DATASET_PATH="${WORKSPACE_DIR}/data/deepscaler/deepscaler_message.jsonl"
export PRETRAIN_MODEL_PATH="${WORKSPACE_DIR}/models/Qwen2.5-VL-3B-Instruct"
export SAVE_PATH="${WORKSPACE_DIR}/checkpoints"

# Model configuration
export MODEL_NAME="lmm-r1-fre-text-dev"

# Wandb configuration (optional)
export WANDB_DIR="${WORKSPACE_DIR}"
# ==========================================================

# Get script PID and setup directories
SCRIPT_PID=$$
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export LOG_DIR="${SAVE_PATH}/${MODEL_NAME}/logs"
export CUR_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"

# Stop any existing processes
pkill -f math_verifier 2>/dev/null || true
pkill -f openrlhf 2>/dev/null || true
sleep 1
${RAY_EXEC} stop 2>/dev/null || true

# Create necessary directories
mkdir -p "${SAVE_PATH}/${MODEL_NAME}"
mkdir -p "${LOG_DIR}"
mkdir -p "${CUR_LOG_DIR}"

# Print help information
echo "================================================================"
echo "LMM-R1 FRE-Text Training (Single NPU Dev)"
echo "================================================================"
echo "Model: ${PRETRAIN_MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Logs: ${CUR_LOG_DIR}"
echo "================================================================"

# Start ray with 1 GPU
echo "Starting ray..."
${RAY_EXEC} start --head --node-ip-address 0.0.0.0 --num-gpus 1 --temp-dir ~/.cache/ray

# 等待 Ray GCS 就绪
echo "Waiting for Ray to be ready..."
sleep 5

# Start remote reward model server
echo "Starting remote reward model server..."
${PYTHON_EXEC} -m openrlhf.models.remote_rm.math_verifier \
    --input_key message \
    --prompt-template chatml 2>&1 | tee -a "${CUR_LOG_DIR}/remote_rm.log" &
REMOTE_RM_PID=$!

# 等待 reward model 启动
sleep 5

# Start training
echo "Starting training..."
export RAY_ADDRESS="0.0.0.0:6379"
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

${PYTHON_EXEC} -m openrlhf.cli.train_ppo_ray \
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

# Record process IDs
echo "Remote RM PID: $REMOTE_RM_PID" > "${CUR_LOG_DIR}/process_pids.txt"
echo "Train PID: $TRAIN_PID" >> "${CUR_LOG_DIR}/process_pids.txt"

echo "Training is running in the background. Check logs at ${CUR_LOG_DIR}/train.log"
echo "To attach to the training process: wait $TRAIN_PID"
