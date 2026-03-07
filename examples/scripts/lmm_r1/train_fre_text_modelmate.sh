#!/bin/bash

# =================== 强制环境重置 (防平台劫持) ===================
export PATH="/usr/local/bin:$PATH"
unset PYTHONPATH
unset LD_PRELOAD

# 强制加载华为昇腾 CANN 的环境变量 (Ray 依赖底层的 acl 库)
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
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
export MODEL_NAME="lmm-r1-fre-text"              

# Wandb configuration (optional)
export WANDB_DIR="${WORKSPACE_DIR}"                
export WANDB_API_KEY="YOUR_WANDB_API_KEY"          
# ==========================================================

# Get script PID and setup directories
SCRIPT_PID=$$
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export LOG_DIR="${SAVE_PATH}/${MODEL_NAME}/logs"
export CUR_LOG_DIR="${LOG_DIR}/${TIMESTAMP}"

# Stop any existing ray processes
${RAY_EXEC} stop

# Create necessary directories
mkdir -p "${SAVE_PATH}/${MODEL_NAME}"
mkdir -p "${LOG_DIR}"
mkdir -p "${CUR_LOG_DIR}"

# Print help information
echo "================================================================"
echo "LMM-R1 FRE-Text Training"
echo "================================================================"
echo "Model name: ${MODEL_NAME}"
echo "Dataset: ${DATASET_PATH}"
echo "Pretrained model: ${PRETRAIN_MODEL_PATH}"
echo "Logs will be saved to: ${CUR_LOG_DIR}"
echo
echo "To monitor logs:"
echo "  tail -f ${CUR_LOG_DIR}/train.log"
echo
echo "================================================================"

# Start ray
echo "Starting ray..."
${RAY_EXEC} start --head --node-ip-address 0.0.0.0 --num-gpus 8 --temp-dir ~/.cache/ray

# 等待 Ray 的 Dashboard 服务在 8265 端口完全启动！
echo "Waiting for Ray dashboard to initialize (10 seconds)..."
sleep 10

# Start remote reward model server
echo "Starting remote reward model server..."
${PYTHON_EXEC} -m openrlhf.models.remote_rm.math_verifier \
    --input_key message \
    --prompt-template chatml 2>&1 | tee -a "${CUR_LOG_DIR}/remote_rm.log" &
REMOTE_RM_PID=$!

# Start training
echo "Starting training..."
${RAY_EXEC} job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="{\"working_dir\": \"${WORKSPACE_DIR}\",\"env_vars\":{\"VLLM_USE_V1\":\"1\",\"VLLM_ENABLE_V1_MULTIPROCESSING\":\"0\"}}" \
   -- ${PYTHON_EXEC} -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --remote_rm_url http://127.0.0.1:5000/get_reward \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --colocate_all_models \
   --vllm_enable_sleep \
   --vllm_gpu_memory_utilization 0.5 \
   --vllm_sync_backend gloo \
   --enable_prefix_caching \
   --pretrain ${PRETRAIN_MODEL_PATH} \
   --save_path ${SAVE_PATH}/${MODEL_NAME} \
   --micro_train_batch_size 2 \
   --train_batch_size 256 \
   --micro_rollout_batch_size 2 \
   --rollout_batch_size 256 \
   --temperature 1.0 \
   --n_samples_per_prompt 16 \
   --max_epochs 1 \
   --num_episodes 2 \
   --prompt_max_len 4096 \
   --max_samples 100000 \
   --generate_max_len 4096 \
   --advantage_estimator reinforce_baseline \
   --zero_stage 3 \
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
   --load_checkpoint \
   --use_wandb ${WANDB_API_KEY} \
   --wandb_run_name ${MODEL_NAME} \
   --wandb_group "lmm-r1-training" \
   --use_tensorboard ${LOG_DIR} > >(tee -a "${CUR_LOG_DIR}/train.log") 2>&1 &

TRAIN_PID=$!

# Record process IDs
echo "Remote RM PID: $REMOTE_RM_PID" > "${CUR_LOG_DIR}/process_pids.txt"
echo "Train PID: $TRAIN_PID" >> "${CUR_LOG_DIR}/process_pids.txt"

# Wait for training to complete
echo "Training is running in the background. Check logs at ${CUR_LOG_DIR}/train.log"
echo "To attach to the training process: wait $TRAIN_PID"

# Cleanup instructions
echo "When finished, clean up with:"
echo "pkill -f openrlhf"
echo "${RAY_EXEC} stop"
echo "All logs are available in ${CUR_LOG_DIR}"