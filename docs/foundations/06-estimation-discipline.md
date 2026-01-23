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

2. A training run achieves 150K tokens/s on 64 H100s for a 7B model. Calculate the MFU.

3. You need to train a 30B model in 30 days on a budget of $2M. Estimate the minimum number of H100s required (at $4/hr) and the required MFU to meet the timeline.
