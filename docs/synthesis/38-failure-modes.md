---
title: "Failure Modes and Diagnostics"
subtitle: "What Breaks First and How to Tell"
---

<div class="chapter-opener" markdown>
Distributed training fails in recognizable ways. This chapter catalogs the most common failure modes and the fastest diagnostics for each.
</div>

<div class="investigation-question" markdown>
**The Question**: The run is slow, unstable, or OOMs. Which invariant is violated—memory, compute, or communication—and how do you prove it quickly?
</div>

## The Failure-Mode Triage

Start with the invariants:

1. **Memory**: OOMs, allocator fragmentation, activation spikes
2. **Compute**: Low MFU, under-occupied kernels, idle gaps
3. **Communication**: Exposed collectives, high latency sensitivity, stragglers

## Common Failure Modes

### 1. Out-of-Memory (OOM)

**Symptom**: Immediate crash, or late-stage OOM after several steps.

**Likely causes**:
- Activation growth from larger batch/sequence
- Fragmentation from dynamic shapes
- Excessive all-gather buffers

**Fast checks**:
- Compare peak allocated vs reserved memory
- Disable dynamic shapes; fix sequence length
- Enable activation checkpointing

### 2. Communication Stall

**Symptom**: NCCL/collective time dominates the step.

**Likely causes**:
- Low communication intensity
- Missing overlap with compute
- Suboptimal topology placement

**Fast checks**:
- Profile timeline for exposed AllReduce
- Increase local batch or accumulation
- Verify NIC saturation vs expected bandwidth

### 3. Pipeline Bubbles

**Symptom**: Periodic idle gaps in pipeline stages.

**Likely causes**:
- Too few micro-batches
- Imbalanced stage assignment

**Fast checks**:
- Compute bubble fraction: $(P-1)/(m+P-1)$
- Increase micro-batches or rebalance layers

### 4. Divergence at Scale

**Symptom**: Loss blows up after scaling GPUs or batch size.

**Likely causes**:
- Learning rate too high for new batch
- Reduced gradient noise

**Fast checks**:
- Reduce LR or use longer warmup
- Apply gradient clipping
- Switch to sqrt LR scaling above $B_{crit}$

### 5. Straggler Domination

**Symptom**: Step time equals slowest rank, high variance across ranks.

**Likely causes**:
- CPU data pipeline bottleneck
- Uneven expert routing
- Host-side overhead

**Fast checks**:
- Measure per-rank step time
- Inspect data-loader utilization
- Add expert load-balance loss

## Diagnostic Table

| Symptom | Invariant | First Test | Typical Fix |
|---|---|---|---|
| OOM | Memory | Peak allocated vs reserved | Checkpointing, ZeRO, offload |
| Low MFU | Compute | Kernel occupancy | Precision, fusion, better batching |
| Exposed NCCL | Communication | Timeline overlap | Bucketing, overlap, topology |
| Bubble gaps | Compute+Comm | Bubble fraction | More micro-batches |
| Divergence | Compute | LR sensitivity | Warmup, LR scaling |

## Closing Note

Failure modes are not random. Each one violates an invariant. Train yourself to map symptoms back to the invariant first, then apply the minimal fix.
