---
title: "The Memory Hierarchy of a Cluster"
subtitle: "From Registers to the Datacenter"
---

<div class="chapter-opener" markdown>
A single machine has a memory hierarchy: registers, L1, L2, L3, DRAM. A cluster extends this hierarchy across machines: HBM, NVLink, InfiniBand, Ethernet. Understanding the bandwidth and latency at each level determines where we can afford to place data.
</div>

<div class="investigation-question" markdown>
**The Question**: Why does tensor parallelism work within a node but struggle across nodes? Why is pipeline parallelism preferred for cross-node communication? The answer lies in the 20× bandwidth gap between NVLink and InfiniBand.
</div>

## The Extended Hierarchy

| Level | Technology | Bandwidth | Latency | Typical Use |
|-------|-----------|-----------|---------|-------------|
| On-chip | SRAM (L1/L2) | ~80 TB/s | ~1 ns | Kernel intermediates |
| Device memory | HBM3 | ~3.35 TB/s | ~100 ns | Model weights, activations |
| Intra-node | NVLink 4.0 | ~900 GB/s | ~1 μs | Tensor parallelism |
| Inter-node (fast) | InfiniBand NDR | ~50 GB/s | ~1-2 μs | Pipeline, data parallelism |
| Inter-node (slow) | Ethernet 100G | ~12 GB/s | ~5 μs | Fallback, some clouds |

The bandwidth drops by **~20×** from NVLink to InfiniBand. This single fact explains most placement decisions.

## Intra-Node Topology

### DGX H100 (8 GPUs)

```
    GPU0 ←→ GPU1 ←→ GPU2 ←→ GPU3
      ↕        ↕        ↕        ↕
    GPU4 ←→ GPU5 ←→ GPU6 ←→ GPU7

    All connected via NVSwitch
    Full bisection bandwidth: 900 GB/s per GPU
```

NVSwitch provides **non-blocking full bisection bandwidth**: any GPU can communicate with any other at full speed simultaneously. This is why 8-way tensor parallelism within a node works well.

### PCIe-Only Systems

Without NVLink, GPUs communicate through PCIe (64 GB/s for Gen5) and potentially through CPU memory. This adds latency and reduces bandwidth, making tensor parallelism across GPUs less attractive.

## Inter-Node Topology

### Fat-Tree

```
        Core Switch
       /    |    \
      /     |     \
   Agg    Agg    Agg  (Aggregation switches)
   /|\    /|\    /|\
  N N N  N N N  N N N  (Nodes)
```

- Provides full bisection bandwidth at each level
- Expensive (many switches)
- Standard in high-performance clusters

### Rail-Optimized (Dragonfly+)

Modern clusters like Meta's Grand Teton use rail-optimized topologies where GPUs at the same position across nodes share a network rail:

```
Node 0:  GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
           |     |     |     |     |     |     |     |
Node 1:  GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
           |     |     |     |     |     |     |     |
         Rail0 Rail1 Rail2 Rail3 Rail4 Rail5 Rail6 Rail7
```

This optimizes for the common case where same-rank GPUs communicate (data parallelism).

## Bandwidth-Latency Trade-offs

### The α-β Model (Preview)

Communication time follows:

$$T(n) = \alpha + \frac{n}{\beta}$$

Where:

- $\alpha$: latency (fixed startup cost)
- $n$: message size in bytes
- $\beta$: bandwidth in bytes/second

For small messages, latency dominates. For large messages, bandwidth dominates.

**Crossover point**: $n^* = \alpha \cdot \beta$

| Link Type | α (latency) | β (bandwidth) | Crossover |
|-----------|-------------|---------------|-----------|
| NVLink | ~1 μs | 900 GB/s | ~1 MB |
| InfiniBand | ~2 μs | 50 GB/s | ~100 KB |

Messages smaller than the crossover point are latency-bound.

## Implications for Parallelism Strategy

### Tensor Parallelism

Requires AllReduce of activation tensors **every layer**. Communication volume is proportional to batch × sequence × hidden.

- Within node (NVLink): ~900 GB/s → viable
- Across nodes (InfiniBand): ~50 GB/s → often bottleneck

**Guideline**: Limit TP degree to GPUs within a node (typically 8).

### Pipeline Parallelism

Only communicates activations at stage boundaries. Point-to-point, not collective. Can tolerate higher latency.

- Activation size: batch × sequence × hidden (often 10s of MB)
- At crossover point for InfiniBand: latency matters less
- Lower bandwidth requirement overall

**Guideline**: Use PP across nodes.

### Data Parallelism

Communicates gradients once per step. Volume is O(parameters), not O(activations × layers).

- Can use gradient accumulation to increase communication intensity
- Compression techniques reduce bandwidth needs
- AllReduce can overlap with backward pass

**Guideline**: DP as the outer parallelism dimension.

## The 3D Parallelism Placement

```
Within DGX Node (8 GPUs):
├── Tensor Parallelism (TP=8)
│   └── Uses NVLink at 900 GB/s

Across Nodes:
├── Pipeline Parallelism (PP=k)
│   └── Uses InfiniBand for P2P
│   └── Point-to-point, latency-tolerant
│
└── Data Parallelism (DP=N/(8*k))
    └── Uses InfiniBand for AllReduce
    └── Can overlap with compute
```

This hierarchy emerges directly from the bandwidth hierarchy of the cluster.

## Exercises

1. You have a 256-GPU cluster with 8 GPUs per node (NVLink) and 400 Gbps InfiniBand between nodes. Calculate the maximum collective bandwidth for (a) 8-way AllReduce within a node, (b) 32-way AllReduce across 4 nodes.

2. A transformer layer does two AllReduce operations for tensor parallelism. The hidden dimension is 8192, batch size is 32, sequence length is 2048. Calculate the total bytes transferred per layer and the time required on (a) NVLink, (b) InfiniBand.

3. Why does pipeline parallelism use point-to-point communication rather than collectives? What would happen if each stage had to AllReduce with all other stages?
