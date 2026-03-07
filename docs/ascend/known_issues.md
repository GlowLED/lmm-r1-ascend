# Ascend 移植：已知问题与排查指南

## 1. 已知限制

### 1.1 ring_attn_size > 1 不可用

**原因**：`ring_flash_attn` 第三方包内部直接调用 flash_attn CUDA kernel，无法在 Ascend 上运行。

**影响**：无法使用序列并行（ring attention）功能。`ring_attn_size` 必须设为 1。

**错误信息**：
```
RuntimeError: ring_flash_attn package is required for ring attention (ring_attn_size > 1) 
but was not found. Please install it or set ring_attn_size=1.
```

**解决方案**：在训练脚本中确保 `--ring_attn_size 1`（默认值即为 1）。

**未来计划**：等待 `ring_flash_attn` 支持 Ascend NPU，或开发基于 torch_npu 的等价实现。

### 1.2 Triton cross_entropy 加速不可用

**原因**：`flash_attn.ops.triton.cross_entropy` 依赖 Triton（NVIDIA GPU 专用编译器）。

**影响**：`log_probs_from_logits` 函数会自动 fallback 到纯 PyTorch 实现（logsumexp），计算结果一致，但 fp32 下可能略慢。

**是否需要处理**：不需要，代码已有完善的 `try/except ImportError` 保护。

### 1.3 视觉模型 FlashAttention2 注意力类不可用

**影响的模型**：
- Phi3V (`modeling_phi3_v.py`)
- Phi4MM (`modeling_phi4mm.py`, `vision_siglip_navit.py`)

**表现**：这些模型的 `FlashAttention2` 变体注意力类无法实例化。

**解决方案**：使用 SDPA 或 Eager 注意力实现。不传 `--flash_attn` 参数时，默认使用 SDPA（已修改）。相关 flash_attn 导入已被 `try/except` 或 `is_flash_attn_2_available()` 保护，不会报错。

## 2. 常见错误排查

### 2.1 `ModuleNotFoundError: No module named 'flash_attn'`

**排查步骤**：

1. 确认没有在命令行传递 `--flash_attn` 参数
2. 检查是否有旧版代码未更新：
   ```bash
   grep -rn "from flash_attn" openrlhf/ --include="*.py" | grep -v "try:" | grep -v "compat"
   ```
3. 如果是 `ring_attn_utils.py` 报错，说明兼容层未正确引入，检查该文件的 import 是否指向 `openrlhf.utils.flash_attn_compat`
4. 如果是视觉模型文件报错，检查是否使用了 `attn_implementation="flash_attention_2"`

### 2.2 `torch.cuda.current_device()` 在 Ascend 上的行为

**背景**：原代码中大量使用 `torch.cuda.current_device()` 等 CUDA API 进行设备管理。

**解决方案**：已通过 `openrlhf/utils/device_utils.py` 抽象层统一替换，自动检测 NPU/CUDA 并路由到对应 API。详见 [device_compat.md](device_compat.md)。

**当前状态**：所有 `torch.cuda.*` 调用已替换为 device_utils 中的设备无关函数。如添加新代码，请使用：

```python
from openrlhf.utils.device_utils import current_device, set_device, device_count, empty_cache, synchronize, get_default_backend
```

### 2.3 NCCL vs HCCL 通信后端

**改动说明**：`deepspeed.py` 中的 `setup_ring_attn` 已自动检测设备类型：
- Ascend NPU (`torch.npu.is_available()` 为 True)：使用 `hccl` 后端
- CUDA GPU：使用 `nccl` 后端

**注意**：`deepspeed.init_distributed()` 的通信后端由 DeepSpeed 自行管理，此处仅影响 ring attention 的 process group 创建。DeepSpeed 本身对 Ascend 的支持请参考 DeepSpeed 官方文档。

### 2.4 `packing_samples` + SDPA 性能注意事项

**背景**：`packing_samples` 功能将多个短序列打包成一个长序列以减少 padding 浪费。该功能依赖 `unpad_input` / `pad_input` 工具函数。

**性能影响**：
- CUDA 上使用 flash_attn 原生函数：最优性能（CUDA kernel）
- Ascend 上使用 fallback 函数：纯 PyTorch 实现，operations 涉及 `torch.nonzero`、advanced indexing、scatter，性能可能不如 CUDA kernel 但功能完全正确
- 整体影响：`unpad_input` / `pad_input` 不是训练的计算瓶颈（attention 计算才是），因此 fallback 对端到端训练速度影响很小

**建议**：在 Ascend 上建议启用 `packing_samples` — 减少 padding 的收益通常远大于 fallback 函数带来的微小开销。

### 2.5 `AttributeError: module 'transformers.modeling_flash_attention_utils' has no attribute ...`

**原因**：transformers 版本不匹配。本框架要求 `transformers==4.51.3`。

**解决方案**：
```bash
pip install transformers==4.51.3
```

## 3. 调试技巧

### 3.1 检查当前注意力后端

```python
from openrlhf.models import Actor
model = Actor("model_path", use_flash_attention_2=False)
print(model.model.config._attn_implementation)
# 预期输出: "sdpa"
```

### 3.2 检查 flash_attn 兼容层状态

```python
from openrlhf.utils.flash_attn_compat import FLASH_ATTN_AVAILABLE
print(f"flash_attn available: {FLASH_ATTN_AVAILABLE}")

# 查看实际使用的函数实现
from openrlhf.utils.flash_attn_compat import unpad_input
print(f"unpad_input implementation: {unpad_input.__module__}.{unpad_input.__name__}")
```

### 3.3 验证通信后端

```python
import torch
_npu = hasattr(torch, 'npu') and torch.npu.is_available()
print(f"NPU available: {_npu}")
print(f"Expected comm backend: {'hccl' if _npu else 'nccl'}")
```

## 4. 修改记录

| 日期 | 修改内容 |
|------|----------|
| 2026-03-06 | 初始 Ascend 移植：创建 flash_attn 兼容层，消除硬依赖，支持 SDPA 后端 |
