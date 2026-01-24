---
title: "Estimation as Discipline"
subtitle: "Back-of-Envelope Calculations for Distributed Training"
---

<div class="chapter-opener" markdown>
Before running expensive experiments, estimate. A capacity engineer should predict training time, memory usage, and communication costs within 2× of actual values using only basic arithmetic.
</div>

<div class="investigation-question" markdown>
**The Question**: You're planning to train a 30B parameter model on 128 H100s. Without running anything, estimate: (1) memory per GPU, (2) time per step, (3) tokens per second. Your estimates should guide hardware selection before spending a dollar.
</div>

## The Estimation Mindset

Good estimates are:

- **Fast**: 5 minutes of calculation, not 5 days of profiling
- **Approximate**: Within 2× is usually sufficient for planning
- **Conservative**: Overestimate costs, underestimate throughput

The goal is to identify infeasible configurations quickly and focus experimentation on viable options.

## Memory Estimation

### Model Memory

For a transformer with:

- $L$ layers
- $H$ hidden dimension
- $V$ vocabulary size
- $A$ attention heads

Total parameters:
$$\Psi \approx 12LH^2 + 2VH$$

Memory for parameters (mixed precision):
$$M_{\text{params}} = 2\Psi \text{ bytes (FP16)}$$

Memory for optimizer (Adam, FP32):
$$M_{\text{opt}} = 12\Psi \text{ bytes (params + momentum + variance)}$$

Total static memory:
$$M_{\text{static}} = 2\Psi + 2\Psi + 12\Psi = 16\Psi$$

### Activation Memory

Per-layer activation memory (without checkpointing):
$$M_{\text{act}}^{\text{layer}} \approx BSH \cdot (34 + 5\frac{AS}{H})$$

Where $B$ = batch, $S$ = sequence, $H$ = hidden, $A$ = heads.

Total activation memory:
$$M_{\text{act}} = L \cdot M_{\text{act}}^{\text{layer}}$$

With activation checkpointing (recompute every $k$ layers):
$$M_{\text{act}}^{\text{ckpt}} = \frac{L}{k} \cdot M_{\text{act}}^{\text{layer}}$$

### Quick Estimation Table

| Component | Formula | 70B Model |
|-----------|---------|-----------|
| Parameters | $2\Psi$ | 140 GB |
| Gradients | $2\Psi$ | 140 GB |
| Optimizer | $12\Psi$ | 840 GB |
| **Total Static** | $16\Psi$ | **1.12 TB** |

This immediately tells us: 70B requires ≥14 H100s just for static memory.

## Compute Estimation

### FLOPs per Token

Forward pass:
$$F_{\text{fwd}} \approx 2\Psi$$

Backward pass:
$$F_{\text{bwd}} \approx 4\Psi$$

Total per token:
$$F_{\text{total}} = 6\Psi$$

### Time per Step

$$T_{\text{step}} = \frac{F_{\text{step}}}{\text{MFU} \times \text{Peak FLOP/s} \times P}$$

Where:

- $F_{\text{step}} = 6\Psi \cdot B \cdot S$ (batch × sequence tokens)
- MFU ≈ 0.40-0.50 for well-optimized training
- P = number of GPUs

**Example**: 70B model, batch=1M tokens, 128 H100s, 45% MFU:
$$T_{\text{step}} = \frac{6 \times 70 \times 10^9 \times 10^6}{0.45 \times 1979 \times 10^{12} \times 128} = 3.67 \text{ seconds}$$

### Tokens per Second

$$\text{Tokens/s} = \frac{B \cdot S}{T_{\text{step}}} = \frac{10^6}{3.67} \approx 272,000 \text{ tokens/s}$$

### Training Time

$$T_{\text{train}} = \frac{D}{B \cdot S} \times T_{\text{step}} = \frac{D \times 6\Psi}{\text{MFU} \times F_{\text{peak}} \times P}$$

For 2T tokens:
$$T_{\text{train}} = \frac{2 \times 10^{12}}{272,000} \approx 7.35 \times 10^6 \text{ seconds} \approx 85 \text{ days}$$

## Communication Estimation

### Data Parallelism

AllReduce volume per step: $2\Psi$ bytes

AllReduce time (ring, P GPUs, bandwidth $\beta$):
$$T_{\text{AR}} = \frac{2\Psi}{\beta} \times \frac{P-1}{P}$$

For 70B across 128 GPUs at 50 GB/s:
$$T_{\text{AR}} = \frac{140 \times 10^9}{50 \times 10^9} \times \frac{127}{128} \approx 2.8 \text{ seconds}$$

### Tensor Parallelism

AllReduce per layer: $2 \times B \times S \times H$ bytes

For 8-way TP, B=4, S=4096, H=8192:
$$T_{\text{TP}}^{\text{layer}} = \frac{2 \times 4 \times 4096 \times 8192}{900 \times 10^9} \approx 0.3 \text{ ms}$$

Total per step (80 layers, 2 AllReduce each): ~48ms

### Pipeline Parallelism

Bubble fraction:
$$\text{Bubble} = \frac{p - 1}{m + p - 1}$$

Where $p$ = pipeline stages, $m$ = micro-batches.

For p=8, m=32: Bubble = 7/39 ≈ 18%

## The Estimation Workflow

1. **Memory check**: Does the model fit? How much parallelism is required?
2. **Compute estimate**: What's the theoretical throughput?
3. **Communication estimate**: What fraction of time is communication?
4. **Bottleneck identification**: Which ceiling dominates?
5. **Sanity check**: Compare to similar published runs

## Common Estimation Errors

| Error | Consequence | Fix |
|-------|-------------|-----|
| Forgetting optimizer states | 3× memory underestimate | Always include 12Ψ |
| Ignoring activations | OOM during training | Account for batch × seq × hidden |
| Assuming 100% MFU | 2× time underestimate | Use 40-50% MFU |
| Ignoring communication | Works in theory, fails in practice | Add AllReduce/AllGather time |

## Exercises

1. Estimate the memory per GPU for a 13B model with TP=4, ZeRO-3 across 32 GPUs. Assume batch=8, sequence=4096, hidden=5120, layers=40.

??? success "Solution"
    **Parallelism configuration:**

    - Total GPUs: 32
    - TP = 4 (tensor parallel groups)
    - DP = 32/4 = 8 (data parallel replicas)
    - ZeRO-3 shards across DP dimension

    **Static memory (ZeRO-3 shards across 32 GPUs):**

    | Component | Formula | Per-GPU |
    |-----------|---------|---------|
    | Parameters | $\frac{2\Psi}{32}$ | $\frac{2 \times 13 \times 10^9}{32} = 0.81$ GB |
    | Gradients | $\frac{2\Psi}{32}$ | 0.81 GB |
    | Optimizer | $\frac{12\Psi}{32}$ | 4.87 GB |
    | **Total Static** | | **6.49 GB** |

    **Activation memory:**

    Using formula: $M_{act}^{layer} \approx BSH \times (34 + 5\frac{AS}{H})$

    Assuming 40 attention heads ($A = 40$):

    $$BSH = 8 \times 4096 \times 5120 = 167.8\text{M}$$

    $$34 + 5 \times \frac{40 \times 4096}{5120} = 34 + 160 = 194$$

    $$M_{act}^{layer} = 167.8\text{M} \times 194 \text{ bytes} \approx 32.5 \text{ GB per layer}$$

    **With TP=4**: Activations are distributed, reducing per-GPU cost by ~4×:

    $$M_{act}^{layer, TP} \approx 8.1 \text{ GB per layer}$$

    **With activation checkpointing** (checkpoint every 4 layers):

    $$M_{act}^{total} = \frac{40}{4} \times 8.1 \approx 81 \text{ GB}$$

    Wait—this exceeds GPU memory! Let's apply more aggressive checkpointing (every layer):

    $$M_{act}^{total} \approx 2 \times 8.1 = 16.2 \text{ GB}$$

    **Total memory per GPU:**

    | Component | Memory |
    |-----------|--------|
    | Static (ZeRO-3) | 6.5 GB |
    | Activations (TP=4, checkpointing) | ~15-20 GB |
    | Temporary buffers | ~5 GB |
    | **Total** | **~27-32 GB** |

    Fits comfortably in 80GB H100.

2. A training run achieves 150K tokens/s on 64 H100s for a 7B model. Calculate the MFU.

??? success "Solution"
    **FLOPs per token:**

    $$F_{token} = 6\Psi = 6 \times 7 \times 10^9 = 42 \times 10^9 \text{ FLOPs/token}$$

    **Achieved FLOPs/s:**

    $$F_{achieved} = 42 \times 10^9 \times 150 \times 10^3 = 6.3 \times 10^{15} \text{ FLOP/s} = 6,300 \text{ TFLOP/s}$$

    **Peak FLOPs/s (64 H100s):**

    $$F_{peak} = 64 \times 1979 \text{ TFLOP/s} = 126,656 \text{ TFLOP/s}$$

    **MFU:**

    $$\text{MFU} = \frac{6,300}{126,656} = \boxed{4.97\% \approx 5\%}$$

    **Analysis:** This is a very low MFU, indicating significant inefficiency. Possible causes:

    | Issue | Likely Impact |
    |-------|---------------|
    | Small batch size | Underutilized compute |
    | Excessive pipeline bubbles | Idle time between stages |
    | Unoptimized kernels | Low SM utilization |
    | Communication not overlapped | GPUs waiting for network |

    **Expected MFU for well-optimized 7B training:** 40-50%

    At 45% MFU, expected throughput would be:

    $$\text{tokens/s} = \frac{0.45 \times 126,656 \times 10^{12}}{42 \times 10^9} = 1.36\text{M tokens/s}$$

    The observed 150K is only 11% of this potential.

3. You need to train a 30B model in 30 days on a budget of $2M. Estimate the minimum number of H100s required (at $4/hr) and the required MFU to meet the timeline.

??? success "Solution"
    **Budget constraint:**

    $$\text{Max GPU-hours} = \frac{\$2,000,000}{\$4/\text{hr}} = 500,000 \text{ GPU-hours}$$

    **Time constraint:**

    $$T_{max} = 30 \text{ days} = 720 \text{ hours}$$

    **GPU constraint from budget:**

    $$P_{max} = \frac{500,000}{720} \approx 694 \text{ GPUs}$$

    Round to practical value: **P = 640 GPUs** (or 512 for power-of-2)

    **Training FLOPs (assuming Chinchilla-optimal 600B tokens):**

    $$F_{total} = 6 \times 30 \times 10^9 \times 600 \times 10^9 = 1.08 \times 10^{23} \text{ FLOPs}$$

    **Required throughput:**

    $$F_{required} = \frac{1.08 \times 10^{23}}{720 \times 3600} = 4.17 \times 10^{16} \text{ FLOP/s}$$

    **Peak throughput with 640 H100s:**

    $$F_{peak} = 640 \times 1979 \times 10^{12} = 1.27 \times 10^{18} \text{ FLOP/s}$$

    **Required MFU:**

    $$\text{MFU}_{required} = \frac{4.17 \times 10^{16}}{1.27 \times 10^{18}} = \boxed{3.3\%}$$

    **Conclusion:** This is easily achievable—well-optimized runs achieve 40-50% MFU.

    **Alternative: Train for more tokens (2T)**

    $$F_{total}^{2T} = 6 \times 30 \times 10^9 \times 2 \times 10^{12} = 3.6 \times 10^{23} \text{ FLOPs}$$

    $$\text{MFU}_{required}^{2T} = \frac{3.6 \times 10^{23} / (720 \times 3600)}{1.27 \times 10^{18}} = 11\%$$

    Still very achievable with standard optimization.

    **Summary:**

    | Scenario | Tokens | GPUs | Required MFU |
    |----------|--------|------|--------------|
    | Chinchilla (600B) | 600B | 640 | 3.3% |
    | Extended (2T) | 2T | 640 | 11% |
    | Budget-limited | 2T | 512 | 14% |
