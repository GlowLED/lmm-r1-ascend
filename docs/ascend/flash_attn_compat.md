# flash_attn 兼容层技术文档

## 1. 调研结论：flash_attn 在代码中的使用

### 1.1 直接导入总表

| 文件 | 行号 | 导入内容 | 导入方式 | 有 fallback? |
|------|------|----------|----------|-------------|
| `openrlhf/models/ring_attn_utils.py` | L3 | `flash_attn.bert_padding.{index_first_axis, pad_input, rearrange, unpad_input}` | **顶层硬导入** | ❌ → 已修复 |
| `openrlhf/models/ring_attn_utils.py` | L4 | `flash_attn.utils.distributed.all_gather` | **顶层硬导入** | ❌ → 已修复 |
| `openrlhf/models/utils.py` | L92 | `flash_attn.ops.triton.cross_entropy.cross_entropy_loss` | `try/except ImportError` | ✅ 原生 PyTorch fallback |
| `openrlhf/models/lmm_kits/phi3_v/src/modeling_phi3_v.py` | L50-54 | `flash_attn.{flash_attn_func, flash_attn_varlen_func}` + `bert_padding` | `try/except ImportError: pass` | ✅ |
| `openrlhf/models/lmm_kits/phi4mm/src/vision_siglip_navit.py` | L334-336 | `flash_attn.{flash_attn_func, flash_attn_varlen_func}` + `bert_padding` | `if is_flash_attn_2_available()` | ✅ |

### 1.2 间接依赖

| 文件 | 内容 | 说明 |
|------|------|------|
| `openrlhf/utils/deepspeed/deepspeed.py` | `import transformers.modeling_flash_attention_utils` | transformers 自有模块，不依赖 flash_attn 安装 |
| `openrlhf/utils/deepspeed/deepspeed.py` | `from ring_flash_attn import substitute_hf_flash_attn` | 仅 `ring_attn_size > 1` 时调用 → 已加 try/except |
| `openrlhf/models/lmm_kits/phi4mm/src/modeling_phi4mm.py` | `from transformers.modeling_flash_attention_utils import _flash_attention_forward` | transformers 内部有 fallback |

### 1.3 CLI 参数传递链

```
CLI (--flash_attn) → args.flash_attn → use_flash_attention_2 → attn_implementation="flash_attention_2"|"sdpa"
```

涉及：`train_sft.py`, `train_dpo.py`, `train_rm.py`, `train_kto.py`, `train_prm.py`, `train_ppo_ray.py`, `batch_inference.py`, `serve_rm.py`, `interactive_chat.py`

## 2. 解决方案设计

### 2.1 架构

```
openrlhf/utils/flash_attn_compat.py
├── 检测 flash_attn 是否可用 → FLASH_ATTN_AVAILABLE (bool)
├── 如果可用: 直接 re-export flash_attn 原生函数
└── 如果不可用: 提供纯 PyTorch fallback 实现
    ├── unpad_input()      — torch.nonzero + indexing
    ├── pad_input()        — scatter via indexing
    ├── index_first_axis() — simple x[indices]
    ├── rearrange()        — 从 einops 导入 (非 flash_attn 专有)
    └── all_gather()       — torch.distributed.all_gather
```

### 2.2 设计原则

1. **零侵入性**：兼容层是一个独立模块，不修改 flash_attn 本身的行为
2. **透明切换**：调用方无需关心当前使用的是原生还是 fallback 实现
3. **性能优先**：当 flash_attn 可用时，始终使用原生实现（CUDA kernel 优化）
4. **最小改动**：只修改导入路径，不改变业务逻辑

## 3. 替代函数实现说明

### 3.1 `unpad_input(hidden_states, attention_mask) → (unpadded, indices, cu_seqlens, max_seqlen, seqlens)`

**原版** (`flash_attn.bert_padding.unpad_input`)：
- 使用 CUDA kernel 高效处理
- 返回 5 个值：`(hidden_states_unpadded, indices, cu_seqlens, max_seqlen_in_batch, seqlens_in_batch)`

**fallback 实现**：
```python
seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
max_seqlen_in_batch = int(seqlens_in_batch.max().item())
cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0), (1, 0))  # 在 CPU 上
indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
hidden_states_unpadded = hidden_states.reshape(-1, *hidden_states.shape[2:])[indices]
```

**性能差异**：fallback 版本在大 batch size 时可能比 CUDA kernel 慢约 2-5x。但此函数通常不是训练瓶颈，影响可控。

### 3.2 `pad_input(hidden_states, indices, batch, seqlen) → padded`

**原版**：CUDA kernel scatter 操作

**fallback 实现**：
```python
output = torch.zeros(batch * seqlen, *other_dims, ...)
output[indices] = hidden_states
return output.reshape(batch, seqlen, *other_dims)
```

**差异**：功能完全等价，性能差异来自 Python indexing vs CUDA kernel。

### 3.3 `index_first_axis(x, indices) → selected`

**原版**：自定义 CUDA kernel，对第一维做高效索引

**fallback 实现**：
```python
return x[indices]
```

**差异**：PyTorch 的 advanced indexing 会创建新 tensor（contiguous copy），与原版语义一致。

### 3.4 `rearrange`

直接从 `einops` 导入。`flash_attn` 中的 `rearrange` 实际上就是 `einops.rearrange` 的 re-export。`einops` 已在 `requirements.txt` 中声明。

### 3.5 `all_gather(tensor, group) → gathered`

**原版** (`flash_attn.utils.distributed.all_gather`)：
- 在 process group 内 all_gather，沿 dim 0 拼接

**fallback 实现**：
```python
gathered_list = [torch.empty_like(tensor) for _ in range(world_size)]
dist.all_gather(gathered_list, tensor, group=group)
return torch.cat(gathered_list, dim=0)
```

**差异**：功能完全等价。原版可能使用了 `all_gather_into_tensor` 避免额外分配，但差异极小。

## 4. 注意力后端选择

### 4.1 三种注意力后端对比

| 后端 | 配置值 | CUDA 支持 | Ascend 支持 | 性能 |
|------|--------|-----------|-------------|------|
| Flash Attention 2 | `"flash_attention_2"` | ✅ (需 flash_attn) | ❌ | 最优 |
| SDPA | `"sdpa"` | ✅ (PyTorch 原生) | ✅ (torch_npu dispatch) | 良好 |
| Eager | `"eager"` | ✅ | ✅ | 基础 |

### 4.2 为什么选 SDPA 作为默认后端

1. **torch_npu 支持**：`torch_npu` 会将 `torch.nn.functional.scaled_dot_product_attention` dispatch 到 NPU 专用融合算子（如 `npu_fusion_attention`），性能接近手写算子
2. **CUDA 上也有加速**：即使在 CUDA 上不使用 flash_attn，SDPA 也会自动选择最优后端（Flash Attention / Memory-Efficient Attention / Math），比 eager 快很多
3. **transformers 原生支持**：HuggingFace transformers 从 v4.36 开始支持 `attn_implementation="sdpa"`，无需额外依赖
4. **向前兼容**：SDPA 是 PyTorch 2.0+ 的标准 API，未来各硬件平台都会优先支持

### 4.3 配置方式

```python
# 修改前 (原始代码)
attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

# 修改后
attn_implementation = "flash_attention_2" if use_flash_attention_2 else "sdpa"
```

用户在 Ascend 上不传 `--flash_attn` 参数时，自动使用 SDPA 后端。

## 5. 影响范围矩阵

### 功能可用性（Ascend NPU）

| 功能 | 状态 | 说明 |
|------|------|------|
| SFT 训练 | ✅ 完全可用 | 使用 SDPA 后端 |
| DPO 训练 | ✅ 完全可用 | 使用 SDPA 后端 |
| RM 训练 | ✅ 完全可用 | 使用 SDPA 后端 |
| KTO 训练 | ✅ 完全可用 | 使用 SDPA 后端 |
| PRM 训练 | ✅ 完全可用 | 使用 SDPA 后端 |
| PPO 训练 | ✅ 完全可用 | 使用 SDPA 后端 |
| packing_samples | ✅ 完全可用 | 使用 fallback unpad/pad 函数 |
| ring_attn_size=1 | ✅ 完全可用 | 不依赖 ring_flash_attn |
| ring_attn_size>1 | ❌ 不可用 | 需要 ring_flash_attn 包，该包依赖 flash_attn |
| Triton cross_entropy 加速 | ⚠️ 自动降级 | fallback 到纯 PyTorch logsumexp 实现 |
| Phi3V FlashAttention2 类 | ⚠️ 不可用 | 切换到 SDPA/Eager 注意力类 |
| Phi4MM FlashAttention2 类 | ⚠️ 不可用 | 切换到 SDPA/Eager 注意力类 |
| batch_inference | ✅ 完全可用 | 不传 --flash_attn 即可 |

### CUDA 环境回归兼容性

| 场景 | 状态 | 说明 |
|------|------|------|
| `--flash_attn` 启用 | ✅ 行为不变 | 使用原生 flash_attn |
| `--flash_attn` 未启用 | ✅ 性能改善 | 从 eager 升级为 sdpa |
| `--packing_samples --flash_attn` | ✅ 行为不变 | 使用原生 flash_attn 的 unpad/pad |
| `--packing_samples` (无 flash_attn) | ✅ 自动启用 | 检测到 flash_attn 可用时自动开启 |
| ring_attn_size > 1 | ✅ 行为不变 | ring_flash_attn 可用时正常工作 |
