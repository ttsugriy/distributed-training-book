---
title: "The Economics of Compute"
subtitle: "Cost Models for Distributed Training"
---

<div class="chapter-opener" markdown>
Hardware is expensive. Time is expensive. Inefficiency is waste. Capacity Engineers must reason about cost as fluently as they reason about performance.
</div>

<div class="investigation-question" markdown>
**The Question**: You have a USD 10M budget to train a 70B parameter model. Do you rent 1000 H100s for 2 weeks or 500 H100s for 4 weeks? The answer depends on efficiency curves, not just total GPU-hours.
</div>

!!! warning "Time-sensitive estimates"
    Cloud prices and utilization assumptions in this chapter are market-dependent snapshots.
    **Last verified:** 2026-02-12. Re-check current provider pricing, reserved discounts, and spot interruption rates before making capacity decisions.

## The Basic Cost Equation

Total training cost:

$$C_{\text{total}} = \underbrace{P \cdot R \cdot T}_{\text{GPU cost}} + \underbrace{C_{\text{network}}}_{\text{networking}} + \underbrace{C_{\text{storage}}}_{\text{data/checkpoints}} + \underbrace{C_{\text{ops}}}_{\text{operations}}$$

Where:

- $P$: number of GPUs
- $R$: hourly rate per GPU (USD/hr)
- $T$: training time in hours
- $C_{\text{network}}$: networking costs (inter-node bandwidth, cross-region transfer)
- $C_{\text{storage}}$: storage costs (training data, checkpoints, logs)
- $C_{\text{ops}}$: operational costs (engineering time, monitoring, incident response)

GPU cost typically dominates (80%+ of total), so we often approximate $C_{\text{total}} \approx P \cdot R \cdot T$.

## GPU-Hour Economics

### Cloud Pricing (2024-2025)

| GPU | On-Demand | Reserved (1yr) | Spot |
|-----|-----------|----------------|------|
| H100 80GB | USD 4-5/hr | USD 2-3/hr | USD 1-2/hr |
| A100 80GB | USD 2-3/hr | USD 1.50-2/hr | USD 0.50-1/hr |
| H200 | ~USD 6/hr | ~USD 4-5/hr | Limited |

Spot instances can provide 2-3× cost reduction but require checkpoint resilience.

### On-Prem vs Cloud Break-Even

On-prem H100 total cost of ownership (TCO) is ~USD 50,000–60,000 per GPU when accounting for the server chassis, networking (InfiniBand switches), power, cooling, datacenter space, and operations staff—not just the ~USD 30,000 GPU cost. At USD 4/hr cloud pricing:

$$\text{Break-even} = \frac{\$55{,}000}{\$4/\text{hr}} = 13{,}750 \text{ GPU-hours} \approx 19 \text{ months at 100\% utilization}$$

At a more realistic ~50% utilization, break-even extends to ~3 years. The calculation is highly sensitive to utilization rate, power costs, and whether you need the latest hardware generation.

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

Where $D$ is the total number of tokens trained on. For Chinchilla-optimal training of large models, expect on the order of USD 100-10,000 per billion tokens depending on model size, hardware, and MFU.

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

For a given model, there's often a "sweet spot" of parallelism where cost per FLOP is minimized. This requires empirical measurement of the efficiency curve.

## Multi-Tenancy and Shared Clusters

Most practitioners don't have dedicated clusters. Shared infrastructure introduces costs that the basic equation $C = P \cdot R \cdot T$ does not capture.

### Contention Effects

When multiple jobs share network fabric, effective bandwidth drops:

$$\beta_{\text{eff}} = \frac{\beta_{\text{raw}}}{\sigma}$$

where $\sigma \geq 1$ is the oversubscription/contention factor. On shared clusters, $\sigma = 1.5$–$3.0$ is typical during peak hours. This directly increases AllReduce time and can shift a training run from compute-bound to communication-bound.

**Diagnosing contention vs. your own bottleneck**: If your NCCL bus bandwidth is 50% of what `nccl-tests` achieves on the same hardware at 2 AM (low contention), the gap is likely external traffic, not your configuration.

### Preemption and Gang Scheduling

Cloud spot/preemptible instances add a probabilistic cost:

$$C_{\text{effective}} = \frac{C_{\text{spot}}}{1 - p_{\text{preempt}} \cdot f_{\text{lost}}}$$

where $p_{\text{preempt}}$ is hourly preemption probability and $f_{\text{lost}}$ is the fraction of work lost per preemption (determined by checkpoint frequency).

**Gang scheduling** ensures all GPUs for a job start and stop together. Without it, partial allocation wastes the running GPUs while waiting for the rest. When sharing clusters:

- Request the minimum viable GPU count (e.g., 64 instead of 128 if efficiency is acceptable)
- Smaller allocations schedule faster and get preempted less
- Design for elastic restart: checkpoint frequently, resume on different node counts

### Scheduling Priority and Queue Time

Queue wait time $T_q$ adds to wall-clock but not GPU cost:

$$T_{\text{wall}} = T_q + T_{\text{train}}$$

For time-constrained projects, $T_q$ can dominate. Strategies:

- **Backfill-friendly sizing**: Jobs that fit in scheduling gaps run sooner
- **Off-peak scheduling**: Queue times can be 5–10× shorter at night
- **Incremental checkpointing**: Short jobs that checkpoint and re-queue avoid long queue waits

### Resource Allocation Decisions

| Scenario | Recommendation |
|----------|---------------|
| Shared fabric, communication-bound | Increase gradient accumulation to raise $I_{\text{net}}$ |
| Frequent preemption ($>10\%$/hr) | Checkpoint every 15–30 min; use async checkpointing |
| Long queue times ($>4$ hrs) | Split into shorter jobs; use backfill-friendly sizes |
| Multi-job contention | Schedule communication-heavy phases at off-peak times |

## Case Study: DeepSeek's USD 5.6M Training

DeepSeek V3 (671B MoE, 37B active) is reported/estimated to have trained for ~USD 5.6M:

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

1. Calculate the GPU-hour cost to train a 70B dense model for 2T tokens on H100s at USD 4/hr, assuming 45% MFU.

??? success "Solution"
    **Total FLOPs required** (using the $6\Psi D$ approximation where $\Psi$ = parameters, $D$ = tokens):

    $$F_{\text{total}} = 6 \times \Psi \times D = 6 \times 70 \times 10^9 \times 2 \times 10^{12} = 8.4 \times 10^{23} \text{ FLOPs}$$

    **Effective compute per GPU-hour:**

    $$F_{\text{GPU-hr}} = \text{MFU} \times \text{Peak} \times 3600 \text{ s}$$

    $$= 0.45 \times 989 \times 10^{12} \times 3600 = 1.60 \times 10^{18} \text{ FLOPs/GPU-hour}$$

    **Total GPU-hours required:**

    $$\text{GPU-hours} = \frac{8.4 \times 10^{23}}{1.60 \times 10^{18}} = 525,000 \text{ GPU-hours}$$

    **Total cost:**

    $$\text{Cost} = 525,000 \times \$4 = \boxed{\$2.10\text{M}}$$

    **Sanity check**: This is in the range of published training costs for models of this scale.

2. You can choose between (a) 512 GPUs at 50% MFU or (b) 1024 GPUs at 35% MFU. Which is more cost-effective for a fixed training FLOP budget?

??? success "Solution"
    **Key insight**: Cost depends on GPU-hours, not raw GPU count.

    For a fixed FLOP budget $F$:

    $$\text{Time} = \frac{F}{\text{GPUs} \times \text{MFU} \times \text{Peak}}$$

    $$\text{GPU-hours} = \text{GPUs} \times \text{Time} = \frac{F}{\text{MFU} \times \text{Peak}}$$

    **Cost is inversely proportional to MFU** (not GPU count!):

    **(a) 512 GPUs at 50% MFU:**
    $$\text{Cost}_a \propto \frac{1}{0.50} = 2.0$$

    **(b) 1024 GPUs at 35% MFU:**
    $$\text{Cost}_b \propto \frac{1}{0.35} = 2.86$$

    **Comparison:**
    $$\frac{\text{Cost}_b}{\text{Cost}_a} = \frac{2.86}{2.0} = 1.43\times$$

    **Option (a) is 43% cheaper** for the same training budget.

    | Configuration | Relative Cost | Training Time |
    |---------------|---------------|---------------|
    | 512 GPUs @ 50% MFU | 1.00× | Longer |
    | 1024 GPUs @ 35% MFU | 1.43× | Shorter |

    **Lesson**: Efficiency matters more than parallelism degree for cost optimization. Only scale up if you can maintain high MFU.

3. A spot instance costs USD 1.50/hr but has 5% chance of preemption per hour. On-demand costs USD 4/hr. If checkpointing overhead is 5% of training time, at what preemption frequency does spot become more expensive than on-demand?

??? success "Solution"
    **Setup:**

    - Spot rate: $R_s = \$1.50$/hr
    - On-demand rate: $R_d = \$4.00$/hr
    - Checkpoint overhead: 5% of training time
    - Base training time: $T$ hours

    **On-demand cost:**
    $$C_d = R_d \times T = 4T$$

    **Spot effective time:**

    With checkpoint overhead and preemption losses:

    $$T_{eff} = T \times (1 + \text{checkpoint overhead} + \text{preemption overhead})$$

    **Preemption model:**

    If checkpoint interval is $c$ hours and preemption rate is $p$ per hour:

    - Expected preemptions: $\approx p \times T_{eff}$
    - Average lost work per preemption: $c/2$ hours
    - Total lost time: $\frac{c}{2} \times p \times T_{eff}$

    For hourly checkpoints ($c = 1$) with 5% overhead:

    $$T_{eff} = 1.05T + 0.5p \times T_{eff}$$

    Solving: $T_{eff} = \frac{1.05T}{1 - 0.5p}$ (for small $pT$)

    **Break-even condition:**
    $$1.50 \times T_{eff} = 4.00 \times T$$

    $$T_{eff} = 2.67T$$

    Substituting:

    $$\frac{1.05}{1 - 0.5p} = 2.67$$

    $$1.05 = 2.67(1 - 0.5p)$$

    $$1.05 = 2.67 - 1.33p$$

    $$1.33p = 1.62$$

    $$p = 1.22 = \boxed{122\%\text{ per hour}}$$

    **Interpretation:** Spot remains cheaper even with very high preemption rates (>100%/hr) because the price differential is so large (2.67×). In practice, real preemption rates of 5-15% make spot instances highly cost-effective if you have robust checkpointing.

    **Practical guidance:**

    | Preemption Rate | Spot Multiplier | Still Cheaper? |
    |-----------------|-----------------|----------------|
    | 5%/hr | 1.08× | Yes (1.62 vs 4.00) |
    | 20%/hr | 1.17× | Yes (1.76 vs 4.00) |
    | 50%/hr | 1.40× | Yes (2.10 vs 4.00) |

## Key Takeaways

1. **Cost scales with utilization**: MFU and throughput dictate real $/token.
2. **Spot economics favor redundancy**: cheap instances win if checkpointing is robust.
3. **Infrastructure decisions are model decisions**: hardware, precision, and parallelism change training budgets by 2–10×.
