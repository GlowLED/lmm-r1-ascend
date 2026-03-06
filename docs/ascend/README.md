# Ascend NPU 移植指南

本目录记录了 OpenRLHF (lmm-r1-ascend) 框架向华为 Ascend NPU 移植的技术方案与注意事项。

## 目录

| 文档 | 说明 |
|------|------|
| [flash_attn_compat.md](flash_attn_compat.md) | flash_attn 兼容层：调研结论、解决方案设计、替代函数实现细节 |
| [known_issues.md](known_issues.md) | 已知问题、限制与排查指南 |

## 背景

原始框架深度依赖 `flash_attn` 库（基于 NVIDIA CUDA 的高性能注意力实现）。在 Ascend NPU 环境下，`flash_attn` 无法安装。本次移植的核心目标是：

1. **消除 flash_attn 硬依赖** — 确保框架在无 flash_attn 的环境下可以正常 import 和运行
2. **保持 CUDA 环境兼容** — 当 flash_attn 可用时，自动使用原生实现以获得最佳性能
3. **支持 SDPA 注意力后端** — 在 Ascend 上通过 `torch_npu` 的 SDPA 支持获得加速

## 开发工作流

```
笔记本 (无 flash_attn, 无 torch_npu)
  │  开发 & 测试 import
  │
  ├─► git push
  │
服务器 (有 torch_npu, 无 flash_attn)
  │  git pull
  │  运行训练
```

在笔记本上开发时，所有代码路径应能正常 import（不报 `ModuleNotFoundError`）。  
在服务器上运行时，通过 `torch_npu` 提供 NPU 计算支持，通过 SDPA 注意力后端进行加速。

## 环境要求

### Ascend 服务器

| 组件 | 版本要求 |
|------|----------|
| CANN (Ascend toolkit) | >= 8.0 |
| torch | >= 2.1 (需带 SDPA 支持) |
| torch_npu | 与 torch 版本匹配 |
| transformers | == 4.51.3 (与 requirements.txt 一致) |
| deepspeed | == 0.16.7 |

### 开发笔记本

无特殊要求，只需 Python 3.8+ 和 `pip install -e .` 能成功即可。  
`flash_attn` **不是必须**的 — 不安装不会影响 import 或基础功能。

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

### 3. 启动 SFT 训练（示例）

```bash
# Ascend 上不需要 --flash_attn 参数
# 使用 --packing_samples 时会自动使用 SDPA 后端
deepspeed --num_gpus 8 \
    openrlhf/cli/train_sft.py \
    --pretrain <model_path> \
    --dataset <data_path> \
    --packing_samples \
    --bf16 \
    --micro_train_batch_size 4
```

### 4. 启动 PPO 训练（示例）

```bash
# 训练命令参考 examples/scripts/ 下的脚本
# 移除 --flash_attn 参数即可在 Ascend 上运行
```

## 修改的文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `openrlhf/utils/flash_attn_compat.py` | **新增** | flash_attn 兼容层，提供纯 PyTorch 回退实现 |
| `openrlhf/models/ring_attn_utils.py` | 修改 | 顶层 import 改为从兼容层导入 |
| `openrlhf/utils/deepspeed/deepspeed.py` | 修改 | 条件导入 + HCCL 通信后端支持 |
| `openrlhf/cli/train_sft.py` | 修改 | 解除 packing_samples 对 flash_attn 的强制绑定 |
| `openrlhf/cli/train_dpo.py` | 修改 | 同上 |
| `openrlhf/cli/train_rm.py` | 修改 | 同上 |
| `openrlhf/cli/train_ppo_ray.py` | 修改 | 同上 |
| `openrlhf/models/actor.py` | 修改 | 默认注意力后端改为 `sdpa` |
| `openrlhf/models/model.py` | 修改 | 同上 |
| `requirements.txt` | 修改 | `flash-attn` 改为可选依赖 |

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
    └─► deepspeed.py
            │
            ├─► substitute_ring_flash_attn()  ← 安全保护
            └─► NCCL / HCCL 自动选择
```
