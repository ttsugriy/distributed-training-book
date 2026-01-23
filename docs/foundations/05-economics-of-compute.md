---
title: "The Economics of Compute"
subtitle: "Cost Models for Distributed Training"
---

<div class="chapter-opener" markdown>
Hardware is expensive. Time is expensive. Inefficiency is waste. Capacity Engineers must reason about cost as fluently as they reason about performance.
</div>

<div class="investigation-question" markdown>
**The Question**: You have a $10M budget to train a 70B parameter model. Do you rent 1000 H100s for 2 weeks or 500 H100s for 4 weeks? The answer depends on efficiency curves, not just total GPU-hours.
</div>

## The Basic Cost Equation

Total training cost:

$$C_{\text{total}} = \underbrace{P \cdot R \cdot T}_{\text{GPU cost}} + \underbrace{C_{\text{network}}}_{\text{networking}} + \underbrace{C_{\text{storage}}}_{\text{data/checkpoints}} + \underbrace{C_{\text{ops}}}_{\text{operations}}$$

Where:

- $P$: number of GPUs
- $R$: hourly rate per GPU
- $T$: training time in hours

GPU cost typically dominates (80%+ of total).

## GPU-Hour Economics

### Cloud Pricing (2024-2025)

| GPU | On-Demand | Reserved (1yr) | Spot |
|-----|-----------|----------------|------|
| H100 80GB | $4-5/hr | $2-3/hr | $1-2/hr |
| A100 80GB | $2-3/hr | $1.50-2/hr | $0.50-1/hr |
| H200 | ~$6/hr | TBD | Limited |

Spot instances can provide 2-3× cost reduction but require checkpoint resilience.

### On-Prem vs Cloud Break-Even

On-prem H100 costs ~$30,000 + infrastructure. At $4/hr cloud pricing:

$$\text{Break-even} = \frac{\$30,000}{\$4/\text{hr}} = 7,500 \text{ GPU-hours} \approx 10.4 \text{ months}$$

If you'll run GPUs >50% utilization for >1 year, on-prem may be cheaper.

## Efficiency Metrics

### Model FLOP Utilization (MFU)

$$\text{MFU} = \frac{\text{Achieved FLOP/s}}{\text{Peak FLOP/s}}$$

State-of-the-art MFU for large-scale training: 40-50%

### Hardware FLOP Utilization (HFU)

Counts re-materialization FLOPs:

$$\text{HFU} = \frac{\text{Total FLOPs executed (including recompute)}}{\text{Peak FLOP/s} \times \text{Time}}$$

HFU > MFU when using activation checkpointing.

### Cost per Token

$$C_{\text{token}} = \frac{C_{\text{total}}}{D}$$

Where $D$ is tokens trained. Chinchilla-optimal training at $0.01-0.05 per billion tokens for large models.

## The Efficiency-Scale Trade-off

Efficiency typically decreases with scale:

```
MFU
 ^
 |  *
 |    *
 |      *
 |        *  *
 |             *  *
 +-------------------> Number of GPUs
     "Efficiency cliff"
```

Causes:

- Communication overhead increases with scale
- Pipeline bubbles don't shrink proportionally
- Load imbalance across parallel dimensions

## Making Cost Decisions

### Fixed Budget, Variable Time

Given budget $B$, minimize training time:

$$\min T \quad \text{s.t.} \quad P \cdot R \cdot T \leq B$$

Solution: Maximize $P$ until efficiency cliff.

### Fixed Time, Variable Cost

Given deadline $T_{\max}$, minimize cost:

$$\min P \cdot R \cdot T \quad \text{s.t.} \quad T \leq T_{\max}$$

Solution: Find minimum $P$ that meets deadline, accounting for efficiency.

### Optimal Operating Point

For a given model, there's often a "sweet spot" of parallelism where $/FLOP is minimized. This requires empirical measurement of the efficiency curve.

## Case Study: DeepSeek's $5.6M Training

DeepSeek V3 (671B MoE, 37B active) trained for ~$5.6M:

- 2048 H800 GPUs
- 14.8T tokens
- FP8 training
- Aggressive MoE design (sparse activation)

Key cost optimizations:
1. MoE: 3× compute efficiency vs dense
2. FP8: ~2× memory/compute efficiency
3. Multi-Token Prediction: Better sample efficiency
4. Custom all-to-all kernels: Lower communication overhead

## Exercises

1. Calculate the GPU-hour cost to train a 70B dense model for 2T tokens on H100s at $4/hr, assuming 45% MFU.

2. You can choose between (a) 512 GPUs at 50% MFU or (b) 1024 GPUs at 35% MFU. Which is more cost-effective for a fixed training FLOP budget?

3. A spot instance costs $1.50/hr but has 5% chance of preemption per hour. On-demand costs $4/hr. If checkpointing overhead is 5% of training time, at what preemption frequency does spot become more expensive than on-demand?
