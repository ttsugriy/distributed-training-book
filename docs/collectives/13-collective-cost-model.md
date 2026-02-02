---
title: "The Collective Cost Model"
subtitle: "Predicting Communication Time for Any Operation"
---

<div class="chapter-opener" markdown>
Armed with α-β analysis and algorithm knowledge, we can predict the cost of any collective operation before running it. This predictive capability is essential for capacity planning.
</div>

<div class="investigation-question" markdown>
**The Question**: Your AllReduce takes 500ms. Is that good or bad? How do you know if you're achieving near-optimal performance? What's the gap between theoretical and achieved, and where does the gap come from?
</div>

!!! abstract "Chapter Map"
    **Prerequisites**: [Chapter 4](../foundations/04-alpha-beta-model.md) (α-β model fundamentals), [Chapter 12](12-ring-tree-algorithms.md) (ring vs tree algorithms)

    **Key insight**: Every collective has a predictable cost formula. Compare your measured times against these formulas to identify inefficiencies—the gap reveals whether you're limited by latency, bandwidth, or implementation overhead.

## The Complete Cost Formulas

Using the α-β model, we can predict communication time for any collective.

!!! note "Theory"
    The formulas below are algorithmic lower bounds under idealized assumptions (perfect overlap, no contention).

!!! note "Practice"
    Real systems typically achieve ~70–90% of these bounds depending on topology, kernel efficiency, and overlap.

### Point-to-Point Operations

**Send/Recv**:

$$T = \alpha + \frac{n}{\beta}$$

This is the foundation. All collective costs derive from compositions of point-to-point operations.

### Broadcast and Reduce

**Broadcast** (tree algorithm):

$$T_{\text{broadcast}} = \log_2 P \cdot \alpha + \log_2 P \cdot \frac{n}{\beta}$$

**Reduce** (tree algorithm):

$$T_{\text{reduce}} = \log_2 P \cdot \alpha + \log_2 P \cdot \frac{n}{\beta}$$

**Note**: Both use tree algorithms because they have single source/destination.

### Scatter and Gather

**Scatter** (binomial tree):

$$T_{\text{scatter}} = \log_2 P \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

The bandwidth term improves because at each level, data size halves.

**Gather** (binomial tree):

$$T_{\text{gather}} = \log_2 P \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

### AllReduce

**Ring algorithm** (optimal for large messages):

$$T_{\text{AllReduce}}^{\text{ring}} = 2(P-1) \cdot \alpha + 2 \cdot \frac{P-1}{P} \cdot \frac{n}{\beta}$$

**Tree algorithm** (optimal for small messages):

$$T_{\text{AllReduce}}^{\text{tree}} = 2\log_2 P \cdot \alpha + 2\log_2 P \cdot \frac{n}{\beta}$$

**Recursive halving-doubling** (best of both for power-of-2 P):

$$T_{\text{AllReduce}}^{\text{RHD}} = 2\log_2 P \cdot \alpha + 2 \cdot \frac{P-1}{P} \cdot \frac{n}{\beta}$$

!!! note "Practice"
    Compare measured NCCL time to the formula above. If you're <70% of the model, you're likely latency-bound (too many small buckets) or bandwidth-bound (contention). Use bigger buckets for latency, or reduce overlap contention for bandwidth.

### AllGather

**Ring algorithm**:

$$T_{\text{AllGather}} = (P-1) \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

Where $n$ is the total output size (each process contributes $n/P$).

### ReduceScatter

**Ring algorithm**:

$$T_{\text{ReduceScatter}} = (P-1) \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

Where $n$ is the total input size (each process outputs $n/P$).

### All-to-All

**Pairwise exchange**:

$$T_{\text{All-to-All}} = (P-1) \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

Where $n$ is the total data per process (sends $n/P$ to each other process).

### Summary Table

| Collective | Latency Term | Bandwidth Term | Algorithm |
|------------|--------------|----------------|-----------|
| Broadcast | $\log_2 P \cdot \alpha$ | $\log_2 P \cdot \frac{n}{\beta}$ | Tree |
| Reduce | $\log_2 P \cdot \alpha$ | $\log_2 P \cdot \frac{n}{\beta}$ | Tree |
| Scatter | $\log_2 P \cdot \alpha$ | $\frac{P-1}{P} \cdot \frac{n}{\beta}$ | Binomial |
| Gather | $\log_2 P \cdot \alpha$ | $\frac{P-1}{P} \cdot \frac{n}{\beta}$ | Binomial |
| AllReduce | $2(P-1) \cdot \alpha$ | $2 \cdot \frac{P-1}{P} \cdot \frac{n}{\beta}$ | Ring |
| AllGather | $(P-1) \cdot \alpha$ | $\frac{P-1}{P} \cdot \frac{n}{\beta}$ | Ring |
| ReduceScatter | $(P-1) \cdot \alpha$ | $\frac{P-1}{P} \cdot \frac{n}{\beta}$ | Ring |
| All-to-All | $(P-1) \cdot \alpha$ | $\frac{P-1}{P} \cdot \frac{n}{\beta}$ | Pairwise |

## Algorithmic Bandwidth

The **algorithmic bandwidth** (algbw) measures data movement relative to a naive baseline.

### Definition

$$\text{algbw} = \frac{n}{T_{\text{collective}}}$$

This is the "effective" bandwidth: how fast data "appears to move" from the application's perspective.

### Bus Bandwidth

For NCCL, **bus bandwidth** (busbw) accounts for the fact that data must traverse multiple links:

$$\text{busbw} = \text{algbw} \times \text{correction factor}$$

The correction factor depends on the collective:

| Collective | Correction Factor |
|------------|-------------------|
| Broadcast | 1 |
| Reduce | 1 |
| AllReduce | $\frac{2(P-1)}{P}$ |
| AllGather | $\frac{P-1}{P}$ |
| ReduceScatter | $\frac{P-1}{P}$ |
| AlltoAll | $\frac{P-1}{P}$ |

### Example Calculation

AllReduce of 1 GB across 8 GPUs takes 50ms.

**Algorithmic bandwidth**:

$$\text{algbw} = \frac{10^9 \text{ bytes}}{0.05 \text{ s}} = 20 \text{ GB/s}$$

**Bus bandwidth**:

$$\text{busbw} = 20 \times \frac{2(8-1)}{8} = 20 \times \frac{14}{8} = 35 \text{ GB/s}$$

If your NIC is 400 Gbps = 50 GB/s, you're achieving 70% of peak—good performance!

## Hierarchical Cost Analysis

Real clusters have network hierarchy. A DGX cluster might have:

- **Intra-node**: 8 GPUs connected via NVLink (900 GB/s per GPU)
- **Inter-node**: 8× 400 Gbps NICs (400 GB/s per node)

### Two-Level AllReduce Model

For $G$ GPUs per node, $N$ nodes ($P = GN$ total):

**Phase 1: Intra-node ReduceScatter**
$$T_1 = (G-1) \cdot \alpha_{\text{NV}} + \frac{G-1}{G} \cdot \frac{n}{\beta_{\text{NV}}}$$

Each GPU now holds $n/G$ of the partial result.

**Phase 2: Inter-node AllReduce**
$$T_2 = 2(N-1) \cdot \alpha_{\text{net}} + 2 \cdot \frac{N-1}{N} \cdot \frac{n/G}{\beta_{\text{net}}}$$

Only $n/G$ bytes cross the network per GPU!

**Phase 3: Intra-node AllGather**
$$T_3 = (G-1) \cdot \alpha_{\text{NV}} + \frac{G-1}{G} \cdot \frac{n}{\beta_{\text{NV}}}$$

**Total**:

$$T_{\text{hier}} = T_1 + T_2 + T_3$$

### Numerical Example

**Setup**: 8 nodes × 8 GPUs = 64 GPUs, AllReduce 2 GB

**Parameters**:

- $\alpha_{\text{NV}} = 1 \mu s$, $\beta_{\text{NV}} = 300$ GB/s (NVSwitch)
- $\alpha_{\text{net}} = 5 \mu s$, $\beta_{\text{net}} = 50$ GB/s (400 GbE)

**Phase 1 (intra-node RS)**:

$$T_1 = 7 \times 10^{-6} + \frac{7}{8} \times \frac{2 \times 10^9}{3 \times 10^{11}} = 7\mu s + 5.8\text{ms} = 5.81\text{ms}$$

**Phase 2 (inter-node AR of 256 MB per GPU)**:

$$T_2 = 14 \times 5 \times 10^{-6} + \frac{14}{8} \times \frac{2.5 \times 10^8}{5 \times 10^{10}} = 70\mu s + 8.75\text{ms} = 8.82\text{ms}$$

**Phase 3 (intra-node AG)**:

$$T_3 = 7 \times 10^{-6} + \frac{7}{8} \times \frac{2 \times 10^9}{3 \times 10^{11}} = 5.81\text{ms}$$

**Total hierarchical**: $T_{\text{hier}} = 5.81 + 8.82 + 5.81 = 20.44$ ms

**Compare to flat ring** (64 GPUs, bottleneck is network):

$$T_{\text{flat}} = 126 \times 5\mu s + \frac{126}{64} \times \frac{2 \times 10^9}{5 \times 10^{10}} = 0.63\text{ms} + 78.75\text{ms} = 79.38\text{ms}$$

**Hierarchical is 3.9× faster** because only 1/8 of data crosses the slow network!

## The Efficiency Gap

In practice, achieved performance is less than theoretical. The gap comes from:

### 1. Software Overhead

NCCL, driver, and kernel launch add latency beyond $\alpha$:

$$T_{\text{actual}} = T_{\text{theory}} + T_{\text{sw}}$$

Typical software overhead: 10-50 μs per operation.

### 2. Protocol Overhead

Headers, checksums, retransmissions add bandwidth overhead:

$$\beta_{\text{effective}} = \beta_{\text{raw}} \times (1 - \text{overhead})$$

Typical overhead: 5-15% of raw bandwidth.

### 3. Contention

Multiple collectives or traffic from other jobs:

$$\beta_{\text{contention}} = \frac{\beta_{\text{raw}}}{\text{contention factor}}$$

Shared clusters often see 30-50% bandwidth reduction.

### 4. Memory Copy

GPU memory to NIC staging buffers adds time:

$$T_{\text{copy}} = \frac{n}{\beta_{\text{PCIe}}}$$

For 1 GB over PCIe Gen5: $\frac{10^9}{64 \times 10^9} \approx 16$ ms

### 5. Synchronization

BSP model requires all processes to wait for slowest:

$$T_{\text{sync}} = T_{\text{max}} - T_{\text{median}}$$

Straggler effects add 5-20% typically.

## Measuring Parameters

### Latency ($\alpha$)

Measure round-trip time for tiny messages:

```python
import torch.distributed as dist
import time

def measure_alpha(iterations=1000):
    tensor = torch.zeros(1, device='cuda')

    # Warmup
    for _ in range(100):
        dist.all_reduce(tensor)

    # Measure
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        dist.all_reduce(tensor)
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # For ring AllReduce: T ≈ 2(P-1)α for tiny messages
    P = dist.get_world_size()
    alpha = elapsed / (iterations * 2 * (P - 1))

    return alpha  # seconds
```

### Bandwidth ($\beta$)

Measure throughput for large messages:

```python
def measure_beta(size_gb=1.0, iterations=10):
    size = int(size_gb * 1e9 / 4)  # float32 elements
    tensor = torch.zeros(size, device='cuda')

    # Warmup
    for _ in range(3):
        dist.all_reduce(tensor)

    # Measure
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(iterations):
        dist.all_reduce(tensor)
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # For ring AllReduce: T ≈ 2(P-1)/P * n/β for large messages
    P = dist.get_world_size()
    n = size_gb * 1e9  # bytes
    factor = 2 * (P - 1) / P

    beta = factor * n * iterations / elapsed

    return beta  # bytes/second
```

### NCCL-tests

The standard tool for collective benchmarking:

```bash
# Build
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests && make

# Run AllReduce benchmark
./build/all_reduce_perf -b 1M -e 1G -f 2 -g 8

# Output interpretation:
#   busbw: bus bandwidth (accounting for algorithm)
#   algbw: algorithmic bandwidth (raw n/T)
```

## Predicting Training Communication Time

For a training step, total communication includes:

### Data Parallel AllReduce

$$T_{\text{DP}} = 2(P-1)\alpha + 2 \cdot \frac{P-1}{P} \cdot \frac{n_{\text{grad}}}{\beta}$$

Where $n_{\text{grad}}$ = total gradient size (≈ model parameters × 4 bytes for FP32).

### ZeRO-3 AllGather + ReduceScatter

Per layer, twice (forward + backward):

$$T_{\text{ZeRO}} = 2 \times \left[ (P-1)\alpha + \frac{P-1}{P} \cdot \frac{n_{\text{layer}}}{\beta} \right]$$

### Tensor Parallel AllReduce

Per transformer layer:

$$T_{\text{TP}} = 4 \times T_{\text{AllReduce}}(n_{\text{activation}})$$

(2 AllReduce in forward, 2 in backward per layer)

### Pipeline Parallel Send/Recv

Per micro-batch:

$$T_{\text{PP}} = 2 \times \left( \alpha + \frac{n_{\text{activation}}}{\beta} \right)$$

### Worked Example: 70B Model

**Setup**: 70B parameters, FP16, 512 GPUs with 8-way TP, 8-way DP, 8-way PP

**Parameters**:

- Gradient size: 70B × 2 bytes = 140 GB (but sharded 8×, so 17.5 GB per DP group)
- TP activation: 8192 × 4096 × 2 = 64 MB per layer
- PP activation: 8192 × 4096 × 2 = 64 MB per micro-batch
- 80 layers, 8 micro-batches

**TP communication** (8 GPUs, NVLink β = 300 GB/s):

$$T_{\text{TP}} = 80 \times 4 \times \left( 14 \times 1\mu s + \frac{14}{8} \times \frac{64 \times 10^6}{3 \times 10^{11}} \right)$$

$$= 320 \times (14\mu s + 0.37\text{ms}) = 123\text{ms}$$

**DP communication** (8 GPUs, network β = 50 GB/s):

$$T_{\text{DP}} = 14 \times 5\mu s + \frac{14}{8} \times \frac{17.5 \times 10^9}{5 \times 10^{10}}$$

$$= 70\mu s + 612.5\text{ms} = 612.6\text{ms}$$

**PP communication** (8 stages, 8 μbatches):

$$T_{\text{PP}} = 8 \times 2 \times \left( 5\mu s + \frac{64 \times 10^6}{5 \times 10^{10}} \right) = 16 \times 1.28\text{ms} = 20.5\text{ms}$$

**Total communication estimate**: 123 + 613 + 21 = **757 ms**

If compute time is 1500 ms, communication overhead is **50%**—needs optimization!

## Optimization Strategies

Given the cost model, we can reason about optimizations:

### Overlap Communication and Compute

Hide $T_{\text{comm}}$ behind $T_{\text{compute}}$:

$$T_{\text{total}} = \max(T_{\text{compute}}, T_{\text{comm}})$$

Instead of:

$$T_{\text{total}} = T_{\text{compute}} + T_{\text{comm}}$$

### Reduce Message Size

- **Gradient compression**: Reduce $n$ by 10-100×
- **Mixed precision**: FP16 vs FP32 halves $n$
- **Gradient accumulation**: Fewer AllReduce calls

### Increase Effective Bandwidth

- **Better algorithms**: Ring vs tree selection
- **Hierarchical collectives**: Exploit topology
- **Bucket fusion**: Amortize latency

### Reduce Parallelism Degree

If $P$ is large and messages are small:

$$T \approx P \cdot \alpha \gg \frac{n}{\beta}$$

Consider: smaller DP group, more TP/PP.

## Validation: Predict vs Measure

Always validate your model against measurements.

### Methodology

1. **Measure α and β** on your specific cluster
2. **Predict time** using formulas
3. **Measure actual time** with profiler
4. **Compute error**: $\frac{|T_{\text{pred}} - T_{\text{actual}}|}{T_{\text{actual}}}$

### Acceptable Error

- < 10%: Excellent model fit
- 10-30%: Useful for planning
- > 30%: Model assumptions violated

### Common Causes of Large Error

1. **Contention**: Other jobs on shared network
2. **Wrong topology**: α, β measured on different path
3. **Protocol effects**: Small message protocol overhead
4. **GPU memory pressure**: Thrashing affects copy bandwidth

!!! tip "Next"
    This chapter assumed a uniform network. The next chapter, [Topology-Aware Collectives](13a-topology-aware-collectives.md), extends the model to hierarchical topologies and contention-aware scheduling.

## Exercises

1. **Formula verification**: For P=16, n=100 MB, α=10μs, β=100 GB/s, calculate:

   - AllReduce time (ring algorithm)
   - AllGather time
   - Speedup of hierarchical (4 nodes × 4 GPUs) vs flat

??? success "Solution"
    **Given parameters:**

    - P = 16 GPUs
    - n = 100 MB = 10⁸ bytes
    - α = 10 μs = 10⁻⁵ s
    - β = 100 GB/s = 10¹¹ bytes/s

    **Part 1: AllReduce time (ring algorithm)**

    Using the ring AllReduce formula:

    $$T_{\text{AllReduce}} = 2(P-1) \cdot \alpha + 2 \cdot \frac{P-1}{P} \cdot \frac{n}{\beta}$$

    Latency term:

    $$T_\alpha = 2(16-1) \times 10^{-5} = 30 \times 10^{-5} = 0.3 \text{ ms}$$

    Bandwidth term:

    $$T_\beta = 2 \times \frac{15}{16} \times \frac{10^8}{10^{11}} = 2 \times 0.9375 \times 10^{-3} = 1.875 \text{ ms}$$

    $$\boxed{T_{\text{AllReduce}} = 0.3 + 1.875 = 2.175 \text{ ms}}$$

    **Part 2: AllGather time**

    Using the ring AllGather formula:

    $$T_{\text{AllGather}} = (P-1) \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

    $$T_{\text{AllGather}} = 15 \times 10^{-5} + \frac{15}{16} \times \frac{10^8}{10^{11}}$$

    $$= 0.15 \text{ ms} + 0.9375 \text{ ms} = \boxed{1.09 \text{ ms}}$$

    **Part 3: Hierarchical (4×4) vs Flat speedup**

    **Flat ring (P=16):** Already calculated: T_flat = 2.175 ms

    **Hierarchical (4 nodes × 4 GPUs per node):**

    Assuming same α and β for simplicity (in practice, NVLink would be faster intra-node):

    *Phase 1: Intra-node ReduceScatter (G=4):*
    $$T_1 = (4-1) \times 10^{-5} + \frac{3}{4} \times \frac{10^8}{10^{11}} = 0.03 + 0.75 = 0.78 \text{ ms}$$

    *Phase 2: Inter-node AllReduce (N=4, data = n/G = 25 MB):*
    $$T_2 = 2(4-1) \times 10^{-5} + 2 \times \frac{3}{4} \times \frac{2.5 \times 10^7}{10^{11}}$$

    $$= 0.06 + 0.375 = 0.435 \text{ ms}$$

    *Phase 3: Intra-node AllGather (G=4):*
    $$T_3 = (4-1) \times 10^{-5} + \frac{3}{4} \times \frac{10^8}{10^{11}} = 0.78 \text{ ms}$$

    $$T_{\text{hier}} = 0.78 + 0.435 + 0.78 = 1.995 \text{ ms}$$

    **Speedup:**
    $$\text{Speedup} = \frac{T_{\text{flat}}}{T_{\text{hier}}} = \frac{2.175}{1.995} = \boxed{1.09\times}$$

    **Note:** With uniform α and β, hierarchical shows modest speedup. The real benefit appears when network β is much slower than NVLink β (e.g., 50 GB/s vs 300 GB/s), giving speedups of 3-4×.

2. **Efficiency analysis**: Your 8-GPU AllReduce of 1 GB takes 80ms. Network is 400 Gbps. Calculate:

   - Algorithmic bandwidth
   - Bus bandwidth
   - Efficiency vs theoretical peak

??? success "Solution"
    **Given:**

    - P = 8 GPUs
    - n = 1 GB = 10⁹ bytes
    - T = 80 ms = 0.08 s
    - Network = 400 Gbps = 50 GB/s

    **Part 1: Algorithmic bandwidth**

    $$\text{algbw} = \frac{n}{T} = \frac{10^9}{0.08} = 12.5 \text{ GB/s}$$

    $$\boxed{\text{algbw} = 12.5 \text{ GB/s}}$$

    **Part 2: Bus bandwidth**

    For AllReduce, the correction factor is $\frac{2(P-1)}{P}$:

    $$\text{busbw} = \text{algbw} \times \frac{2(P-1)}{P} = 12.5 \times \frac{2 \times 7}{8}$$

    $$= 12.5 \times 1.75 = \boxed{21.875 \text{ GB/s}}$$

    **Part 3: Efficiency vs theoretical peak**

    The theoretical peak is the network bandwidth:

    $$\text{Efficiency} = \frac{\text{busbw}}{\beta_{\text{peak}}} = \frac{21.875}{50} = \boxed{43.75\%}$$

    **Analysis:**

    | Metric | Value |
    |--------|-------|
    | Algorithmic bandwidth | 12.5 GB/s |
    | Bus bandwidth | 21.875 GB/s |
    | Peak bandwidth | 50 GB/s |
    | Efficiency | 43.75% |

    **This efficiency is concerning.** Possible causes:

    1. **Software overhead**: NCCL kernel launch, synchronization
    2. **Protocol overhead**: Headers, checksums reducing effective bandwidth
    3. **Contention**: Other traffic on shared network
    4. **Memory copies**: PCIe bottleneck between GPU and NIC

    **Expected efficiency** for well-optimized systems: 70-85%

    **To investigate**, run NCCL-tests with varying message sizes to see if the issue is latency-bound (small messages) or bandwidth-bound (large messages).

3. **Training time prediction**: A 13B model has:

   - 4B parameters in attention (TP)
   - 9B parameters in FFN (TP)
   - Sequence length 4096, hidden 5120, batch 2M tokens
   - 64 GPUs: 8 TP × 8 DP

   Estimate communication time per step. Which parallelism dominates?

??? success "Solution"
    **Setup:**

    - 13B total parameters (4B attention + 9B FFN)
    - TP = 8, DP = 8 (64 total GPUs)
    - Sequence S = 4096, Hidden H = 5120
    - Batch = 2M tokens → micro-batch per DP replica = 2M/8 = 250K tokens
    - Per-GPU batch: B = 250K / 4096 ≈ 61 sequences

    **Assume:**

    - NVLink β = 300 GB/s (intra-node for TP)
    - Network β = 50 GB/s (inter-node for DP)
    - α_NV = 1 μs, α_net = 5 μs

    **Part 1: Tensor Parallel Communication**

    TP requires AllReduce of activations within each TP group.

    Activation size per layer:

    $$n_{\text{act}} = B \times S \times H \times 2 \text{ bytes (FP16)}$$

    $$= 61 \times 4096 \times 5120 \times 2 = 2.56 \text{ GB}$$

    Per transformer layer: 4 AllReduce operations (2 forward, 2 backward)

    Each AllReduce (ring, P=8, NVLink):

    $$T_{\text{AR}} = 2(8-1) \times 10^{-6} + 2 \times \frac{7}{8} \times \frac{2.56 \times 10^9}{3 \times 10^{11}}$$

    $$= 14 \mu s + 14.9 \text{ ms} = 14.9 \text{ ms}$$

    Assuming ~40 layers:

    $$T_{\text{TP}} = 40 \times 4 \times 14.9 = \boxed{2,384 \text{ ms}}$$

    **Part 2: Data Parallel Communication**

    DP requires AllReduce of gradients across DP groups.

    Gradient size: 13B × 2 bytes (FP16) = 26 GB total

    But with TP=8, each TP group only has 26/8 = 3.25 GB of unique gradients to sync.

    AllReduce across DP=8 groups (network):

    $$T_{\text{DP}} = 2(8-1) \times 5 \times 10^{-6} + 2 \times \frac{7}{8} \times \frac{3.25 \times 10^9}{5 \times 10^{10}}$$

    $$= 70 \mu s + 113.75 \text{ ms} = \boxed{114 \text{ ms}}$$

    **Summary:**

    | Parallelism | Communication Time | Percentage |
    |-------------|-------------------|------------|
    | Tensor Parallel | 2,384 ms | 95.4% |
    | Data Parallel | 114 ms | 4.6% |
    | **Total** | **2,498 ms** | 100% |

    $$\boxed{\text{Tensor Parallelism dominates by } 21\times}$$

    **Analysis:**

    TP dominates because:
    1. Activation tensors are large (2.56 GB per AllReduce)
    2. 4 AllReduce ops per layer × 40 layers = 160 operations
    3. Even with fast NVLink, sheer volume is massive

    **Optimizations to consider:**

    1. **Sequence parallelism**: Reduce activation size per GPU
    2. **Activation checkpointing**: Trade compute for memory (doesn't help communication directly)
    3. **Reduce TP degree**: TP=4 instead of TP=8 would halve TP communication, but increase per-GPU memory
    4. **Overlap TP AllReduce with compute**: Use async collectives

4. **Hierarchical benefit**: You're choosing between:

   - 64 GPUs: 8 nodes × 8 GPUs
   - 64 GPUs: 16 nodes × 4 GPUs

   For 4 GB AllReduce with NVLink 300 GB/s, network 50 GB/s, which is faster?

??? success "Solution"
    **Given:**

    - n = 4 GB = 4 × 10⁹ bytes
    - β_NV = 300 GB/s (NVLink, intra-node)
    - β_net = 50 GB/s (network, inter-node)
    - α_NV = 1 μs, α_net = 5 μs

    **Configuration A: 8 nodes × 8 GPUs (G=8, N=8)**

    *Phase 1: Intra-node ReduceScatter (G=8):*
    $$T_1 = (8-1) \times 10^{-6} + \frac{7}{8} \times \frac{4 \times 10^9}{3 \times 10^{11}}$$

    $$= 7 \mu s + 11.67 \text{ ms} = 11.67 \text{ ms}$$

    *Phase 2: Inter-node AllReduce (N=8, data = 4GB/8 = 500 MB per GPU):*
    $$T_2 = 2(8-1) \times 5 \times 10^{-6} + 2 \times \frac{7}{8} \times \frac{5 \times 10^8}{5 \times 10^{10}}$$

    $$= 70 \mu s + 17.5 \text{ ms} = 17.57 \text{ ms}$$

    *Phase 3: Intra-node AllGather (G=8):*
    $$T_3 = (8-1) \times 10^{-6} + \frac{7}{8} \times \frac{4 \times 10^9}{3 \times 10^{11}} = 11.67 \text{ ms}$$

    $$T_A = 11.67 + 17.57 + 11.67 = \boxed{40.91 \text{ ms}}$$

    **Configuration B: 16 nodes × 4 GPUs (G=4, N=16)**

    *Phase 1: Intra-node ReduceScatter (G=4):*
    $$T_1 = (4-1) \times 10^{-6} + \frac{3}{4} \times \frac{4 \times 10^9}{3 \times 10^{11}}$$

    $$= 3 \mu s + 10 \text{ ms} = 10 \text{ ms}$$

    *Phase 2: Inter-node AllReduce (N=16, data = 4GB/4 = 1 GB per GPU):*
    $$T_2 = 2(16-1) \times 5 \times 10^{-6} + 2 \times \frac{15}{16} \times \frac{10^9}{5 \times 10^{10}}$$

    $$= 150 \mu s + 37.5 \text{ ms} = 37.65 \text{ ms}$$

    *Phase 3: Intra-node AllGather (G=4):*
    $$T_3 = 10 \text{ ms}$$

    $$T_B = 10 + 37.65 + 10 = \boxed{57.65 \text{ ms}}$$

    **Comparison:**

    | Configuration | Phase 1 | Phase 2 | Phase 3 | Total |
    |---------------|---------|---------|---------|-------|
    | 8×8 (A) | 11.67 ms | 17.57 ms | 11.67 ms | **40.91 ms** |
    | 16×4 (B) | 10 ms | 37.65 ms | 10 ms | 57.65 ms |

    $$\boxed{\text{8×8 is 1.41× faster than 16×4}}$$

    **Why 8×8 wins:**

    1. **More GPUs per node** → smaller data crosses slow network
       - 8×8: 500 MB per GPU crosses network
       - 16×4: 1 GB per GPU crosses network (2× more!)

    2. **Fewer nodes** → fewer inter-node AllReduce steps
       - 8×8: N=8 → 2×7 = 14 latency steps
       - 16×4: N=16 → 2×15 = 30 latency steps

    **Key insight:** Maximize GPUs per node to minimize network traffic. The network is the bottleneck, so keeping more computation intra-node pays off.

5. **Overlap potential**: Compute takes 2000ms, communication takes 600ms. If you can overlap 80% of communication, what's the speedup?

??? success "Solution"
    **Given:**

    - $T_{\text{compute}} = 2000$ ms
    - $T_{\text{comm}} = 600$ ms
    - Overlap fraction = 80%

    **Without overlap (sequential execution):**

    $$T_{\text{sequential}} = T_{\text{compute}} + T_{\text{comm}} = 2000 + 600 = 2600 \text{ ms}$$

    **With 80% overlap:**

    The overlapped portion runs concurrently with compute. Only the non-overlapped portion adds to total time:

    $$T_{\text{overlapped}} = T_{\text{compute}} + (1 - 0.80) \times T_{\text{comm}}$$

    $$= 2000 + 0.20 \times 600 = 2000 + 120 = 2120 \text{ ms}$$

    **Speedup:**

    $$\text{Speedup} = \frac{T_{\text{sequential}}}{T_{\text{overlapped}}} = \frac{2600}{2120} = \boxed{1.23\times}$$

    **Alternative view - time saved:**

    $$\text{Time saved} = 2600 - 2120 = 480 \text{ ms}$$

    $$\text{Reduction} = \frac{480}{2600} = 18.5\%$$

    **Analysis:**

    | Scenario | Total Time | Speedup |
    |----------|------------|---------|
    | No overlap (0%) | 2600 ms | 1.00× |
    | 80% overlap | 2120 ms | 1.23× |
    | 100% overlap | 2000 ms | 1.30× |

    **Theoretical maximum speedup** (perfect overlap):

    $$\text{Speedup}_{\text{max}} = \frac{T_{\text{compute}} + T_{\text{comm}}}{T_{\text{compute}}} = \frac{2600}{2000} = 1.30\times$$

    We achieve $\frac{1.23 - 1.00}{1.30 - 1.00} = 77\%$ of the theoretical maximum benefit.

    **Practical considerations:**

    1. **Overlap techniques**: Gradient bucketing, async AllReduce, pipelining
    2. **Why not 100%?**: Some operations require synchronization (e.g., first/last layers, optimizer step)
    3. **Memory trade-off**: Overlapping requires buffering gradients during communication

6. **Parameter measurement**: Design an experiment to separately measure:

   - GPU-GPU latency (same node)
   - GPU-GPU latency (different nodes)
   - NVLink bandwidth
   - Network bandwidth

??? success "Solution"
    **Experiment Design for α-β Parameter Measurement**

    The key insight: use **tiny messages to isolate latency (α)** and **large messages to isolate bandwidth (β)**.

    **Experiment 1: GPU-GPU Latency (Same Node) - α_NV**

    ```python
    import torch
    import torch.distributed as dist
    import time

    def measure_intra_node_latency(iterations=10000):
        """Measure NVLink latency using tiny AllReduce."""
        # Tiny tensor - bandwidth term negligible
        tensor = torch.zeros(1, device='cuda')

        # Warmup
        for _ in range(100):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        # Measure
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        # For ring AllReduce: T ≈ 2(P-1)α when n→0
        P = dist.get_world_size()  # Should be GPUs on same node
        alpha_nv = elapsed / (iterations * 2 * (P - 1))

        return alpha_nv * 1e6  # Return in microseconds
    ```

    **Run configuration**: Single node, all 8 GPUs, NCCL with NVLink only.

    **Expected result**: α_NV ≈ 1-5 μs

    ---

    **Experiment 2: GPU-GPU Latency (Different Nodes) - α_net**

    ```python
    def measure_inter_node_latency(iterations=10000):
        """Measure network latency using tiny AllReduce across nodes."""
        # Use ONLY one GPU per node to isolate network latency
        tensor = torch.zeros(1, device='cuda')

        # Warmup
        for _ in range(100):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        # Measure
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        N = dist.get_world_size()  # Number of nodes
        alpha_net = elapsed / (iterations * 2 * (N - 1))

        return alpha_net * 1e6  # microseconds
    ```

    **Run configuration**: 1 GPU per node, multiple nodes, network only.

    **Expected result**: α_net ≈ 5-20 μs (depends on network topology)

    ---

    **Experiment 3: NVLink Bandwidth - β_NV**

    ```python
    def measure_nvlink_bandwidth(size_gb=2.0, iterations=20):
        """Measure NVLink bandwidth using large AllReduce."""
        size = int(size_gb * 1e9 / 4)  # float32 elements
        tensor = torch.zeros(size, device='cuda')

        # Warmup
        for _ in range(3):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        # Measure
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        # For large n: T ≈ 2(P-1)/P × n/β (latency negligible)
        P = dist.get_world_size()
        n = size_gb * 1e9  # bytes
        factor = 2 * (P - 1) / P

        # β = factor × n × iterations / elapsed
        beta_nv = factor * n * iterations / elapsed

        return beta_nv / 1e9  # Return in GB/s
    ```

    **Run configuration**: Single node, all 8 GPUs, NVLink only.

    **Expected result**: β_NV ≈ 200-300 GB/s per GPU (NVSwitch)

    ---

    **Experiment 4: Network Bandwidth - β_net**

    ```python
    def measure_network_bandwidth(size_gb=4.0, iterations=10):
        """Measure network bandwidth using large AllReduce across nodes."""
        size = int(size_gb * 1e9 / 4)
        tensor = torch.zeros(size, device='cuda')

        # Warmup
        for _ in range(3):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        # Measure
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        N = dist.get_world_size()
        n = size_gb * 1e9
        factor = 2 * (N - 1) / N

        beta_net = factor * n * iterations / elapsed

        return beta_net / 1e9  # GB/s
    ```

    **Run configuration**: 1 GPU per node (to avoid NVLink contribution), multiple nodes.

    **Expected result**: β_net ≈ 40-50 GB/s per GPU (400 GbE)

    ---

    **Complete Measurement Protocol**

    | Parameter | Configuration | Message Size | Iterations |
    |-----------|---------------|--------------|------------|
    | α_NV | 8 GPUs, 1 node | 4 bytes | 10,000 |
    | α_net | 1 GPU/node, N nodes | 4 bytes | 10,000 |
    | β_NV | 8 GPUs, 1 node | 2 GB | 20 |
    | β_net | 1 GPU/node, N nodes | 4 GB | 10 |

    **Validation steps:**

    1. **Consistency check**: Run each experiment 5 times, report mean ± std
    2. **Size sweep**: Vary message size from 1KB to 10GB, plot T vs n
    3. **Fit α-β model**: Linear regression on T = α + n/β
    4. **Compare to spec**: NVLink should be ~6× network bandwidth

    **Example results table:**

    | Parameter | Measured | Expected | Status |
    |-----------|----------|----------|--------|
    | α_NV | 2.3 μs | 1-5 μs | ✓ |
    | α_net | 8.7 μs | 5-20 μs | ✓ |
    | β_NV | 285 GB/s | 250-300 GB/s | ✓ |
    | β_net | 47 GB/s | 40-50 GB/s | ✓ |

## Key Takeaways

1. **Formulas exist for all collectives**: Use α-β model with algorithm-specific factors.

2. **Bus bandwidth normalizes comparisons**: Accounts for algorithmic data amplification.

3. **Hierarchy dramatically reduces network load**: Factor of G (GPUs per node) reduction.

4. **Real systems have overhead**: Plan for 70-80% of theoretical peak.

5. **Predict then measure**: Validate your model on actual infrastructure.

6. **Communication often dominates at scale**: Understanding costs enables optimization.
