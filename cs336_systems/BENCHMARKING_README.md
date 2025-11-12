# Transformer Model Benchmarking Script

这个脚本用于对 Transformer 模型进行端到端的前向和反向传播基准测试。

## 功能特性

- ✅ 使用给定超参数初始化模型
- ✅ 生成随机批次数据
- ✅ 支持预热步骤（warm-up steps）
- ✅ 使用 `timeit.default_timer()` 进行高精度计时
- ✅ 每步后调用 `torch.cuda.synchronize()` 确保准确计时
- ✅ 支持仅前向传播或前向+反向传播模式
- ✅ 详细的统计信息（平均值、中位数、标准差等）

## 使用方法

### 基本用法

```bash
# 默认配置（前向+反向传播）
python -m cs336_systems.benchmarking_script

# 仅前向传播
python -m cs336_systems.benchmarking_script --mode forward

# CPU 模式
python -m cs336_systems.benchmarking_script --device cpu
```

### 自定义模型配置

```bash
python -m cs336_systems.benchmarking_script \
    --num-layers 24 \
    --d-model 1024 \
    --num-heads 16 \
    --d-ff 4096 \
    --batch-size 4 \
    --context-length 2048
```

### 调整基准测试参数

```bash
python -m cs336_systems.benchmarking_script \
    --warmup-steps 10 \
    --num-steps 50 \
    --batch-size 16
```

## 命令行参数

### 模型超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vocab-size` | 50257 | 词汇表大小 |
| `--context-length` | 1024 | 最大上下文长度 |
| `--d-model` | 768 | 模型维度 |
| `--num-layers` | 12 | Transformer 层数 |
| `--num-heads` | 12 | 注意力头数量 |
| `--d-ff` | 3072 | 前馈网络维度 |
| `--rope-theta` | 10000.0 | RoPE 位置编码参数 |

### 基准测试参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 8 | 批次大小 |
| `--warmup-steps` | 5 | 预热步骤数（不计时） |
| `--num-steps` | 20 | 计时步骤数 |
| `--mode` | forward_backward | 模式：`forward` 或 `forward_backward` |
| `--device` | cuda | 设备：`cuda` 或 `cpu` |
| `--learning-rate` | 3e-4 | 优化器学习率 |

## 输出示例

```
================================================================================
BENCHMARK CONFIGURATION
================================================================================
Model Configuration:
  vocab_size: 50257
  context_length: 1024
  d_model: 768
  num_layers: 12
  num_heads: 12
  d_ff: 3072
  rope_theta: 10000.0

Benchmark Configuration:
  batch_size: 8
  warmup_steps: 5
  num_steps: 20
  mode: forward_backward
  device: cuda
================================================================================

Initializing model...
Total parameters: 124,439,808 (124.44M)
Initializing AdamW optimizer with lr=0.0003

Starting benchmark in forward_backward mode...
Running 5 warm-up steps...
Running 20 timed steps...
  Completed 10/20 steps
  Completed 20/20 steps

================================================================================
BENCHMARK RESULTS
================================================================================
Mode: forward_backward
Total time: 3.4567 seconds
Number of steps: 20

Per-step statistics:
  Mean time:   0.1728 seconds (5.79 steps/sec)
  Median time: 0.1715 seconds
  Std dev:     0.0123 seconds
  Min time:    0.1598 seconds
  Max time:    0.1945 seconds

Throughput:
  Tokens per step: 8,192
  Tokens per second: 47,407

GPU Memory:
  Allocated: 1.23 GB
  Reserved:  1.45 GB
  Max allocated: 1.35 GB
================================================================================
```

## 技术细节

### 计时机制

脚本使用 `timeit.default_timer()` 进行高精度计时，这是系统中分辨率最高的时钟，比 `time.time()` 更适合基准测试。

### 同步机制

每步执行后都会调用 `torch.cuda.synchronize()`，确保所有 CUDA 操作完成后再停止计时，避免异步执行导致的计时不准确。

### 预热步骤

预热步骤用于：
- 初始化 CUDA 内核
- 稳定 GPU 频率
- 填充缓存
- 确保后续测量的准确性

### 统计信息

脚本计算以下统计信息：
- **平均时间**：所有步骤的算术平均值
- **中位数时间**：排序后的中间值，更能抵抗异常值
- **标准差**：衡量时间的波动性
- **最小/最大时间**：边界情况
- **吞吐量**：每秒处理的 token 数量

## 使用建议

1. **预热步骤**：对于更准确的测量，可以增加预热步骤（例如 `--warmup-steps 10`）
2. **测量步骤**：增加计时步骤数可以获得更稳定的统计（例如 `--num-steps 50`）
3. **GPU 模式**：确保没有其他进程占用 GPU 资源
4. **一致性**：在不同配置间比较时，保持批次大小和序列长度一致

## 示例用例

### 1. 比较不同层数的性能

```bash
# 12 层
python -m cs336_systems.benchmarking_script --num-layers 12 --num-steps 30

# 24 层
python -m cs336_systems.benchmarking_script --num-layers 24 --num-steps 30
```

### 2. 测试扩展性

```bash
# 小批次
python -m cs336_systems.benchmarking_script --batch-size 4

# 大批次
python -m cs336_systems.benchmarking_script --batch-size 32
```

### 3. 前向传播性能（推理）

```bash
python -m cs336_systems.benchmarking_script \
    --mode forward \
    --batch-size 32 \
    --num-steps 100
```
