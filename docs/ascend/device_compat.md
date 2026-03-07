# CUDA → NPU 设备兼容层技术文档

## 1. 背景与动机

原始 OpenRLHF 代码库中大量使用 `torch.cuda.*` API 进行设备管理：

- `torch.cuda.current_device()` — 获取当前设备索引
- `torch.cuda.set_device()` — 设置当前设备
- `torch.cuda.device_count()` — 获取可用设备数
- `torch.cuda.empty_cache()` — 释放显存缓存
- `torch.cuda.synchronize()` — 同步等待所有 kernel 完成
- `"nccl"` — 分布式通信后端硬编码

在 Ascend NPU 环境下，这些 API 不可用或行为不正确。虽然 `torch_npu` 在某些场景下提供 CUDA 兼容层，但在以下情况仍然会失败：

1. **设备初始化阶段**：Ray worker 启动时，`torch.cuda.*` 无法感知 NPU 设备
2. **通信后端**：NPU 使用 HCCL（Huawei Collective Communication Library）而非 NCCL
3. **设备可见性**：Ray 不会自动设置 `ASCEND_RT_VISIBLE_DEVICES`，导致 NPU 进程无法正确识别设备
4. **CUDA IPC**：NPU 不支持 CUDA 进程间通信（IPC），需要使用广播替代

## 2. 解决方案：device_utils 抽象层

### 2.1 架构设计

```
openrlhf/utils/device_utils.py
├── _npu_available()      — 检测 torch.npu 是否可用
├── _cuda_available()     — 检测 torch.cuda 是否可用
├── current_device()      — 自适应获取当前设备索引
├── set_device(device)    — 自适应设置当前设备
├── device_count()        — 自适应获取设备数量
├── empty_cache()         — 自适应释放缓存
├── synchronize()         — 自适应同步等待
└── get_default_backend() — 返回 "hccl" 或 "nccl"
```

### 2.2 检测逻辑

```python
def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()
```

**优先级**：NPU 优先。当 `torch_npu` 已安装且 NPU 设备可用时，所有函数均路由到 `torch.npu.*`；否则 fallback 到 `torch.cuda.*`。

### 2.3 使用方式

```python
# 修改前
import torch
torch.cuda.set_device(local_rank)
device = torch.cuda.current_device()
torch.cuda.empty_cache()
torch.cuda.synchronize()

# 修改后
from openrlhf.utils.device_utils import set_device, current_device, empty_cache, synchronize
set_device(local_rank)
device = current_device()
empty_cache()
synchronize()
```

## 3. 修改文件清单

### 3.1 新增文件

| 文件 | 说明 |
|------|------|
| `openrlhf/utils/device_utils.py` | 设备抽象层，6 个设备无关函数 |

### 3.2 训练器 — 替换 `torch.cuda.current_device()`

| 文件 | 修改内容 |
|------|----------|
| `openrlhf/trainer/sft_trainer.py` | `torch.cuda.current_device()` → `current_device()` |
| `openrlhf/trainer/dpo_trainer.py` | 同上 |
| `openrlhf/trainer/kto_trainer.py` | 同上 |
| `openrlhf/trainer/rm_trainer.py` | 同上 |
| `openrlhf/trainer/kd_trainer.py` | 同上 |
| `openrlhf/trainer/prm_trainer.py` | 同上 |

这些文件中 `current_device()` 用于确定当前进程的设备索引，将数据或模型放置到对应设备上。

### 3.3 Ray 分布式组件

| 文件 | 修改内容 |
|------|----------|
| `openrlhf/trainer/ray/launcher.py` | 替换 `current_device()`、`empty_cache()`、`synchronize()`；在 `__init__` 中添加 `ASCEND_RT_VISIBLE_DEVICES` 环境变量设置 |
| `openrlhf/trainer/ray/ppo_actor.py` | 替换所有 `torch.cuda.*` 调用；NCCL 默认值改为 `get_default_backend()` |
| `openrlhf/trainer/ray/ppo_critic.py` | 替换所有 `torch.cuda.*` 调用 |
| `openrlhf/trainer/ray/vllm_engine.py` | 添加 `ASCEND_RT_VISIBLE_DEVICES` 设置（从 `ray.get_gpu_ids()` 获取） |
| `openrlhf/trainer/ray/vllm_worker_wrap.py` | `torch.cuda.synchronize()` 分 NPU/CUDA 两条路径 |
| `openrlhf/trainer/ray/utils.py` | `get_physical_gpu_id()` 优先尝试 `torch_npu`（NPU 不支持 UUID 方式的 IPC） |

### 3.4 DeepSpeed 与分布式工具

| 文件 | 修改内容 |
|------|----------|
| `openrlhf/utils/deepspeed/deepspeed.py` | 替换 `set_device`、`synchronize`、`current_device`、`empty_cache`；通信后端自动检测 HCCL/NCCL |
| `openrlhf/utils/deepspeed/deepspeed_utils.py` | 替换 `empty_cache`、`synchronize` |
| `openrlhf/utils/distributed_util.py` | `torch_dist_barrier_and_cuda_sync()` 内部使用 device_utils 的 `synchronize` |

### 3.5 CLI 入口

| 文件 | 修改内容 |
|------|----------|
| `openrlhf/cli/batch_inference.py` | 替换 `current_device`、`device_count` |
| `openrlhf/cli/interactive_chat.py` | 替换 `current_device` |
| `openrlhf/cli/train_ppo_ray.py` | `--vllm_sync_backend` 默认值从 `"nccl"` 改为 `"gloo"` |

### 3.6 模型与数据处理

| 文件 | 修改内容 |
|------|----------|
| `openrlhf/models/ring_attn_utils.py` | 替换 `current_device` |
| `openrlhf/trainer/ppo_utils/replay_buffer.py` | `f"cuda:{torch.cuda.current_device()}"` → `current_device()` |
| `openrlhf/models/lmm_kits/phi4mm/src/speech_conformer_encoder.py` | `.cuda()` → `.to(xs_pad.device)`；`is_cuda` → `device.type != 'cpu'` |

## 4. 通信后端适配

### 4.1 分布式后端选择逻辑

| 场景 | CUDA (NVIDIA) | NPU (Ascend) |
|------|---------------|--------------|
| DeepSpeed 内部通信 | DeepSpeed 自行管理 | DeepSpeed 自行管理 |
| Ring Attention process group | `nccl` | `hccl` |
| PPO Actor ↔ vLLM 权重同步 | `nccl` (或用户指定) | `get_default_backend()` → `hccl` |
| vLLM 同步后端 (`--vllm_sync_backend`) | `nccl` | `gloo`（默认值已改） |

### 4.2 CUDA IPC 的处理

CUDA IPC（`torch.cuda.ipc_handle()`）允许同一节点上的 GPU 进程高效共享 tensor。NPU 不支持此机制。

**处理方式**：`ppo_actor.py` 中，当 `backend == "nccl"` 且 `colocate_all_models` 时才启用 CUDA IPC。在 NPU 上 backend 为 `"hccl"` 或 `"gloo"`，条件不满足，自动走广播路径。

```python
# ppo_actor.py
backend = getattr(self.strategy.args, "vllm_sync_backend", get_default_backend())
self.use_cuda_ipc = False
if backend == "nccl" and self.strategy.args.colocate_all_models:
    self.use_cuda_ipc = True  # 仅 CUDA + NCCL 时启用
```

## 5. Ray 与 NPU 设备可见性

### 5.1 问题

Ray 将 Ascend NPU 作为 "GPU" 资源管理（通过 `ray.get_gpu_ids()` 获取分配结果），但 Ray 不会自动设置 `ASCEND_RT_VISIBLE_DEVICES` 环境变量。这导致子进程（如 vLLM EngineCore）看不到正确的 NPU 设备，触发 `aclInit error 107001`。

### 5.2 解决方案

在 Ray actor 初始化时手动设置环境变量：

```python
# launcher.py — DistributedTorchRayActor.__init__
try:
    import torch_npu
    gpu_ids = ray.get_gpu_ids()
    if gpu_ids:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(int(gpu_ids[0]))
except ImportError:
    pass

# vllm_engine.py — LLMRayActor.__init__
try:
    import torch_npu
    gpu_ids = ray.get_gpu_ids()
    if gpu_ids:
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(int(gpu_ids[0]))
except ImportError:
    pass
```

### 5.3 `get_physical_gpu_id()` 的适配

原函数通过 `torch.cuda.get_device_properties(device).uuid` 获取 GPU 物理 ID，用于 CUDA IPC 的 handle 分发。NPU 没有 UUID 概念，也不使用 IPC。

```python
def get_physical_gpu_id():
    try:
        import torch_npu
        device = torch.npu.current_device()
        return str(device)  # NPU: 返回设备索引
    except ImportError:
        pass
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)  # CUDA: 返回 UUID
```

## 6. Tensor 设备迁移（.cuda() → .to(device)）

### 6.1 speech_conformer_encoder.py

原代码使用 `is_cuda` 判断 + `.cuda()` 迁移：

```python
# 修改前
if xs_pad.is_cuda:
    enc_streaming_mask = enc_streaming_mask.cuda()
    xs_pad = xs_pad.cuda()

# 修改后
if not xs_pad.device.type == 'cpu':
    enc_streaming_mask = enc_streaming_mask.to(xs_pad.device)
    xs_pad = xs_pad.to(xs_pad.device)
```

**原因**：`is_cuda` 在 NPU tensor 上为 `False`，导致 mask 留在 CPU 而 xs_pad 在 NPU，后续运算报设备不匹配错误。改为 `device.type != 'cpu'` 后，任何加速器设备（cuda、npu、xpu 等）均能正确处理。

### 6.2 replay_buffer.py

```python
# 修改前
device = f"cuda:{torch.cuda.current_device()}"

# 修改后
from openrlhf.utils.device_utils import current_device
device = current_device()
```

`current_device()` 返回的是设备索引（int），可直接用于 DeepSpeed / transformers 传参。

## 7. 验证方法

### 7.1 静态检查

确认无遗漏的 `torch.cuda.*` 调用：

```bash
# 应仅返回 device_utils.py 自身和已有 NPU 分支的代码
grep -rn 'torch\.cuda\.' openrlhf/ --include='*.py' | grep -v '__pycache__'

# 应无结果
grep -rn '\.cuda()' openrlhf/ --include='*.py' | grep -v '__pycache__' | grep -v '#'
```

### 7.2 导入测试

```bash
# 在无 torch_npu 的环境中（fallback 到 CUDA 路径）
python -c "from openrlhf.utils.device_utils import current_device, get_default_backend; print(get_default_backend())"
# 预期: nccl

# 在 Ascend NPU 环境中
python -c "from openrlhf.utils.device_utils import current_device, get_default_backend; print(get_default_backend())"
# 预期: hccl
```

### 7.3 运行时测试

```bash
# 单机 SFT 训练
deepspeed --num_gpus 8 openrlhf/cli/train_sft.py \
    --pretrain <model_path> --dataset <data_path> --bf16

# Ray PPO 训练
python openrlhf/cli/train_ppo_ray.py \
    --pretrain <model_path> --vllm_sync_backend gloo ...
```

## 8. 注意事项

1. **vllm_worker_wrap.py 中保留了 `device="cuda"` 字面量**：vllm_ascend 插件内部会将 `"cuda"` 映射到 NPU，不需要修改
2. **`--vllm_sync_backend` 默认值改为 `gloo`**：HCCL 在 vLLM 跨进程同步场景下可能不稳定，`gloo` 更通用
3. **CUDA IPC 仅在 NCCL + colocate 模式下启用**：NPU 环境下自动禁用，走广播路径，功能等价但可能略慢
4. **device_utils.py 不处理 `torch.device` 对象构造**：如需构造设备对象，使用 `torch.device("npu", current_device())` 或让框架自行处理
