---
title: "The Collective Cost Model"
subtitle: "Predicting Communication Time for Any Operation"
---

::: {.chapter-opener}
Armed with α-β analysis and algorithm knowledge, we can predict the cost of any collective operation before running it. This predictive capability is essential for capacity planning.
:::

::: {.investigation-question}
**The Question**: Your AllReduce takes 500ms. Is that good or bad? How do you know if you're achieving near-optimal performance? What's the gap between theoretical and achieved, and where does the gap come from?
:::

## The Complete Cost Formulas

Using the α-β model, we can predict communication time for any collective.

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

### AllGather

**Ring algorithm**:
$$T_{\text{AllGather}} = (P-1) \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

Where $n$ is the total output size (each process contributes $n/P$).

### ReduceScatter

**Ring algorithm**:
$$T_{\text{ReduceScatter}} = (P-1) \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

Where $n$ is the total input size (each process outputs $n/P$).

### AlltoAll

**Pairwise exchange**:
$$T_{\text{AlltoAll}} = (P-1) \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

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
| AlltoAll | $(P-1) \cdot \alpha$ | $\frac{P-1}{P} \cdot \frac{n}{\beta}$ | Pairwise |

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

Per microbatch:

$$T_{\text{PP}} = 2 \times \left( \alpha + \frac{n_{\text{activation}}}{\beta} \right)$$

### Worked Example: 70B Model

**Setup**: 70B parameters, FP16, 512 GPUs with 8-way TP, 8-way DP, 8-way PP

**Parameters**:
- Gradient size: 70B × 2 bytes = 140 GB (but sharded 8×, so 17.5 GB per DP group)
- TP activation: 8192 × 4096 × 2 = 64 MB per layer
- PP activation: 8192 × 4096 × 2 = 64 MB per microbatch
- 80 layers, 8 microbatches

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

## Exercises

1. **Formula verification**: For P=16, n=100 MB, α=10μs, β=100 GB/s, calculate:
   - AllReduce time (ring algorithm)
   - AllGather time
   - Speedup of hierarchical (4 nodes × 4 GPUs) vs flat

2. **Efficiency analysis**: Your 8-GPU AllReduce of 1 GB takes 80ms. Network is 400 Gbps. Calculate:
   - Algorithmic bandwidth
   - Bus bandwidth
   - Efficiency vs theoretical peak

3. **Training time prediction**: A 13B model has:
   - 4B parameters in attention (TP)
   - 9B parameters in FFN (TP)
   - Sequence length 4096, hidden 5120, batch 2M tokens
   - 64 GPUs: 8 TP × 8 DP

   Estimate communication time per step. Which parallelism dominates?

4. **Hierarchical benefit**: You're choosing between:
   - 64 GPUs: 8 nodes × 8 GPUs
   - 64 GPUs: 16 nodes × 4 GPUs

   For 4 GB AllReduce with NVLink 300 GB/s, network 50 GB/s, which is faster?

5. **Overlap potential**: Compute takes 2000ms, communication takes 600ms. If you can overlap 80% of communication, what's the speedup?

6. **Parameter measurement**: Design an experiment to separately measure:
   - GPU-GPU latency (same node)
   - GPU-GPU latency (different nodes)
   - NVLink bandwidth
   - Network bandwidth

## Key Takeaways

1. **Formulas exist for all collectives**: Use α-β model with algorithm-specific factors.

2. **Bus bandwidth normalizes comparisons**: Accounts for algorithmic data amplification.

3. **Hierarchy dramatically reduces network load**: Factor of G (GPUs per node) reduction.

4. **Real systems have overhead**: Plan for 70-80% of theoretical peak.

5. **Predict then measure**: Validate your model on actual infrastructure.

6. **Communication often dominates at scale**: Understanding costs enables optimization.
