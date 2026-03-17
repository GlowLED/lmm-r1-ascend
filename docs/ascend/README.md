# Ascend NPU 移植指南

本目录记录了 OpenRLHF (lmm-r1-ascend) 框架向华为 Ascend NPU 移植的技术方案与注意事项。

## 当前开发进度

### 已完成

- [x] **flash_attn 兼容层** — 消除 flash_attn 硬依赖，自动 fallback 到纯 PyTorch SDPA
- [x] **device_utils 抽象层** — `torch.cuda.*` → 设备无关 API（NPU/CUDA 自动切换）
- [x] **通信后端适配** — NCCL → HCCL/gloo 自动选择
- [x] **Ray NPU 可见性** — `ASCEND_RT_VISIBLE_DEVICES` + `VLLM_ENABLE_V1_MULTIPROCESSING=0`
- [x] **ATB/NNAL 全面 fallback** — 无 NNAL 环境下 stub 所有 ATB 算子：
  - `_npu_flash_attention_unpad` → SDPA fallback
  - `_npu_matmul_add_fp32` → `torch.addmm` fallback
  - `_npu_reshape_and_cache` → 直接张量索引写入
  - `_OpNamespace.__getattr__` 补丁（仅拦截 `atb` 命名空间）
  - `NPUWorker._warm_up_atb` → no-op
- [x] **Pillow 兼容** — `Image.ExifTags` + `ExifTags.Base` shim（适配 Pillow < 8.2）
- [x] **Ray 序列化修复** — `import torch` 移入 `__init__` 避免 `_Ops` pickle 失败
- [x] **datasets/pandas 兼容** — 原生 json 加载 .jsonl 绕过 `ujson_loads` bug
- [x] **数据格式处理** — 自动包装裸 JSON 数组为 `{"message": [...]}` 字典
- [x] **Messages 格式序列化** — 非字符串 prompt 用 `json.dumps` 序列化
- [x] **Qwen2.5-VL 模型适配**：
  - `embed_tokens` 自适应查找（`self.model` vs `self.model.language_model`）
  - `get_rope_index` 缺失方法 fallback
  - 假视觉 forward 包裹 `torch.no_grad()` + `.detach()`（避免 Conv3d backward SIGSEGV）
- [x] **ZeRO-3 + adam_offload** — 单 NPU 下 CPU 卸载优化器状态
- [x] **权重广播修复（broadcast_to_vllm）**：
  - `device="cuda"` → 设备自适应（`get_current_device_string()`）
  - gloo 后端下 NPU 张量 → CPU 中转 → broadcast → 回到 NPU
- [x] **batch_inference ATB stub** — `batch_inference.py` 的 `generate_vllm` 路径加入 ATB fallback
- [x] **单 NPU PPO 训练完整流程验证**（Qwen2.5-VL-3B-Instruct, 16 步, reward/loss 指标正常）

### 进行中 / 待验证

- [ ] 多 NPU（8 卡）大规模训练测试
- [ ] `broadcast_to_vllm` 权重同步端到端稳定性（已修复代码，待部署验证）
- [ ] vLLM batch_inference 完整 ATB fallback 覆盖（视觉编码器路径可能还需补丁）
- [ ] 多模态（图像/视频）输入推理验证

### 已知不支持

- Ring Attention（`ring_attn_size > 1`）— 依赖 CUDA 专用 ring_flash_attn 包
- CUDA IPC 权重同步 — NPU 不支持，使用 gloo broadcast 替代
- Triton cross_entropy 加速 — 自动 fallback 到 PyTorch logsumexp

## 目录

| 文档 | 说明 |
|------|------|
| [flash_attn_compat.md](flash_attn_compat.md) | flash_attn 兼容层：调研结论、解决方案设计、替代函数实现细节 |
| [device_compat.md](device_compat.md) | CUDA → NPU 设备兼容层：device_utils 抽象、通信后端适配、Ray NPU 可见性 |
| [known_issues.md](known_issues.md) | 已知问题、限制与排查指南 |

## 背景

原始框架深度依赖 `flash_attn` 库（基于 NVIDIA CUDA 的高性能注意力实现）。在 Ascend NPU 环境下，`flash_attn` 无法安装。本次移植的核心目标是：

1. **消除 flash_attn 硬依赖** — 确保框架在无 flash_attn 的环境下可以正常 import 和运行
2. **保持 CUDA 环境兼容** — 当 flash_attn 可用时，自动使用原生实现以获得最佳性能
3. **支持 SDPA 注意力后端** — 在 Ascend 上通过 `torch_npu` 的 SDPA 支持获得加速
4. **设备 API 抽象** — 将所有 `torch.cuda.*` 调用替换为设备无关的抽象层，支持 NPU 和 CUDA 自动切换
5. **通信后端适配** — 自动选择 HCCL（NPU）或 NCCL（CUDA）后端，保证分布式训练正常运行
6. **ATB 算子兼容** — 在无 NNAL 的 Ascend 环境下，为所有 ATB 算子提供纯 PyTorch fallback

## 实际验证环境

| 组件 | 版本 |
|------|------|
| 服务器 | notebook-lmmr1tra-a66eacb1, 1× Ascend NPU (~64GB HBM), aarch64 |
| CANN | 8.5.0 |
| torch | 2.9.0+cpu |
| torch_npu | 2.9.0 |
| vLLM | v0.14.1 + vllm_ascend 插件 |
| deepspeed | 0.18.7（用户 site-packages） |
| transformers | 5.x |
| NNAL/ATB | **未安装**（所有 ATB 算子使用 fallback） |
| 模型 | Qwen2.5-VL-3B-Instruct |

## 开发工作流

```
笔记本 (无 flash_attn, 无 torch_npu)
  │  开发 & 测试 import
  │
  ├─► git push
  │
服务器 (有 torch_npu, 无 flash_attn, 无 NNAL)
  │  git pull
  │  运行训练 / 推理
```

在笔记本上开发时，所有代码路径应能正常 import（不报 `ModuleNotFoundError`）。  
在服务器上运行时，通过 `torch_npu` 提供 NPU 计算支持，通过 SDPA 注意力后端进行加速。

## 快速开始

### 1. 安装依赖

```bash
# Ascend 服务器上
pip install -e .
# flash_attn 不需要安装（也无法在 Ascend 上安装）
```

### 2. 验证兼容层

```bash
python -c "from openrlhf.utils.flash_attn_compat import FLASH_ATTN_AVAILABLE; print(f'flash_attn available: {FLASH_ATTN_AVAILABLE}')"
# 预期输出: flash_attn available: False
```

### 3. 单 NPU PPO 训练（正确性验证）

```bash
cd /home/work/user-job-dir/app/model/lmm-r1-ascend
bash examples/scripts/lmm_r1/train_fre_text_1npu.sh
```

此脚本使用最小配置（100 条数据, batch_size=8, 1 epoch）验证整个 PPO pipeline 端到端可用。

### 4. 推理测试（训练后模型）

```bash
# 交互式对话
/usr/local/bin/python3.11 -m openrlhf.cli.interactive_chat \
    --pretrain /absolute/path/to/checkpoints/lmm-r1-fre-text-dev \
    --bf16 --apply_chat_template --max_len 2048

# 批量推理（vLLM 路径，不依赖 MPI）
VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_ASCEND_ENABLE_NZ=0 DISABLE_ATB_EXTENSION=1 \
/usr/local/bin/python3.11 -m openrlhf.cli.batch_inference \
    --eval_task generate_vllm \
    --pretrain /absolute/path/to/checkpoints/lmm-r1-fre-text-dev \
    --dataset /absolute/path/to/data.jsonl \
    --input_key message --max_samples 20 --tp_size 1 \
    --max_new_tokens 2048 \
    --output_path /absolute/path/to/eval_output.jsonl
```

> **注意**：模型路径必须使用**绝对路径**，否则 HuggingFace 会将 `./` 开头的路径误判为 repo_id。

## 修改的文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `openrlhf/utils/flash_attn_compat.py` | **新增** | flash_attn 兼容层，提供纯 PyTorch 回退实现 |
| `openrlhf/utils/device_utils.py` | **新增** | 设备抽象层，提供 7 个设备无关函数 + `get_current_device_string()` |
| `openrlhf/models/ring_attn_utils.py` | 修改 | 顶层 import 改为从兼容层导入；`torch.cuda.current_device()` → `current_device()` |
| `openrlhf/utils/deepspeed/deepspeed.py` | 修改 | 条件导入 + HCCL 通信后端支持 + device_utils 替换 |
| `openrlhf/utils/deepspeed/deepspeed_utils.py` | 修改 | `empty_cache`、`synchronize` 替换为 device_utils |
| `openrlhf/utils/distributed_util.py` | 修改 | 同步函数使用 device_utils |
| `openrlhf/utils/utils.py` | 修改 | 原生 json 加载 .jsonl + 自动包装裸 JSON 数组 |
| `openrlhf/cli/train_sft.py` | 修改 | 解除 packing_samples 对 flash_attn 的强制绑定 |
| `openrlhf/cli/train_dpo.py` | 修改 | 同上 |
| `openrlhf/cli/train_rm.py` | 修改 | 同上 |
| `openrlhf/cli/train_ppo_ray.py` | 修改 | `--vllm_sync_backend` 默认值从 nccl 改为 gloo |
| `openrlhf/cli/batch_inference.py` | 修改 | device_utils 替换 + ATB fallback stub |
| `openrlhf/cli/interactive_chat.py` | 修改 | `current_device` 替换 |
| `openrlhf/trainer/sft_trainer.py` | 修改 | device_utils 替换 |
| `openrlhf/trainer/dpo_trainer.py` | 修改 | 同上 |
| `openrlhf/trainer/kto_trainer.py` | 修改 | 同上 |
| `openrlhf/trainer/rm_trainer.py` | 修改 | 同上 |
| `openrlhf/trainer/kd_trainer.py` | 修改 | 同上 |
| `openrlhf/trainer/prm_trainer.py` | 修改 | 同上 |
| `openrlhf/trainer/ray/launcher.py` | 修改 | device_utils 替换 + `ASCEND_RT_VISIBLE_DEVICES` 设置 |
| `openrlhf/trainer/ray/ppo_actor.py` | 修改 | device_utils 替换 + gloo CPU 中转 broadcast |
| `openrlhf/trainer/ray/ppo_critic.py` | 修改 | device_utils 替换 |
| `openrlhf/trainer/ray/vllm_engine.py` | 修改 | NPU 初始化 + ATB 全面 fallback + Pillow shim + repo_id 修复 |
| `openrlhf/trainer/ray/vllm_worker_wrap.py` | 修改 | `device="cuda"` → 设备自适应 + gloo CPU broadcast |
| `openrlhf/trainer/ray/utils.py` | 修改 | `get_physical_gpu_id()` NPU 适配 |
| `openrlhf/trainer/ppo_utils/replay_buffer.py` | 修改 | 设备字符串构造替换 |
| `openrlhf/datasets/prompts_dataset.py` | 修改 | 非字符串 prompt json.dumps 序列化 |
| `openrlhf/models/lmm_kits/qwen2_5_vl/patch.py` | 修改 | embed_tokens 自适应 + get_rope_index fallback + Conv3d no_grad |
| `openrlhf/models/lmm_kits/phi4mm/src/speech_conformer_encoder.py` | 修改 | `.cuda()` → `.to(device)` |
| `openrlhf/models/actor.py` | 修改 | 默认注意力后端改为 `sdpa` |
| `openrlhf/models/model.py` | 修改 | 同上 |
| `requirements.txt` | 修改 | `flash-attn` 改为可选依赖 |
| `examples/scripts/lmm_r1/train_fre_text_1npu.sh` | **新增** | 单 NPU 正确性验证脚本 |

## 架构概览

```
用户代码 (actor.py / model.py / CLI)
    │
    ├─► ring_attn_utils.py
    │       │
    │       └─► flash_attn_compat.py  ← 兼容层
    │               │
    │               ├─► flash_attn (CUDA, 如果可用)
    │               └─► 纯 PyTorch fallback (Ascend / 无 flash_attn)
    │
    ├─► device_utils.py  ← 设备抽象
    │       │
    │       ├─► torch.npu.* (Ascend NPU)
    │       └─► torch.cuda.* (NVIDIA GPU)
    │
    ├─► vllm_engine.py  ← ATB fallback (训练时)
    │       │
    │       ├─► ATB 算子 fallback table
    │       ├─► Pillow ExifTags shim
    │       └─► _OpNamespace.__getattr__ 补丁
    │
    └─► deepspeed.py
            │
            ├─► substitute_ring_flash_attn()  ← 安全保护
            └─► NCCL / HCCL 自动选择
```
