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
| On-chip | SRAM (L1/L2) | ~80 TB/s | ~1-10 ns | Kernel intermediates |
| Device memory | HBM3 | ~3.35 TB/s | ~200-500 ns | Model weights, activations |
| Intra-node | NVLink 4.0 | ~900 GB/s | ~1 μs | Tensor parallelism |
| Inter-node (fast) | InfiniBand NDR | ~50 GB/s | ~1-2 μs | Pipeline, data parallelism |
| Inter-node (slow) | Ethernet 100G | ~12 GB/s | ~10-50 μs | Fallback, some clouds |

The bandwidth drops by **~20×** from NVLink to InfiniBand. This, along with latency and topology, explains most placement decisions.

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

Without NVLink, GPUs communicate through PCIe (Gen5 x16: ~32 GB/s per direction, ~64 GB/s bidirectional) and potentially through CPU memory. This adds latency and reduces bandwidth, making tensor parallelism across GPUs less attractive.

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

Requires collective communication of activation tensors **every layer** (AllGather/ReduceScatter in common TP schemes). Communication volume is proportional to batch × sequence × hidden.

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

??? success "Solution"
    **Ring AllReduce effective bandwidth:**

    For ring AllReduce, effective bandwidth ≈ $\beta \times \frac{P}{2(P-1)}$ where $\beta$ is link bandwidth.

    **(a) 8-way AllReduce within node (NVLink):**

    - NVLink bandwidth: $\beta = 900$ GB/s
    - $P = 8$ GPUs

    $$BW_{eff} = 900 \times \frac{8}{2 \times 7} = 900 \times \frac{8}{14} \approx 514 \text{ GB/s}$$

    In practice, NCCL can reach ~500-600 GB/s on large messages, depending on topology and measurement definition.

    **(b) 32-way AllReduce across 4 nodes:**

    Uses hierarchical AllReduce:

    1. Reduce within each node (NVLink, fast)
    2. Reduce across 4 node representatives (InfiniBand, slow)
    3. Broadcast within each node (NVLink, fast)

    Inter-node bottleneck:

    - InfiniBand: 400 Gbps = 50 GB/s per direction
    - 4-way AllReduce across nodes

    $$BW_{eff}^{inter} = 50 \times \frac{4}{2 \times 3} \approx 33 \text{ GB/s}$$

    The inter-node communication dominates, limiting effective bandwidth to **~30-40 GB/s** for the full 32-way collective.

2. A transformer layer does two AllReduce operations for tensor parallelism. The hidden dimension is 8192, batch size is 32, sequence length is 2048. Calculate the total bytes transferred per layer and the time required on (a) NVLink, (b) InfiniBand.

??? success "Solution"
    **Activation tensor size (BF16):**

    $$\text{Size} = B \times S \times H \times 2 = 32 \times 2048 \times 8192 \times 2 = 1.07 \text{ GB}$$

    **Per AllReduce volume (ring, large P):**

    $$V_{AR} \approx 2 \times \text{Size} = 2.14 \text{ GB}$$

    **Total for 2 AllReduces:**

    $$V_{total} = 2 \times 2.14 = 4.28 \text{ GB per layer}$$

    **Time calculations:**

    (a) **NVLink** (900 GB/s):

    $$T_{NVLink} = \frac{4.28 \text{ GB}}{900 \text{ GB/s}} = 4.8 \text{ ms}$$

    (b) **InfiniBand** (50 GB/s):

    $$T_{IB} = \frac{4.28 \text{ GB}}{50 \text{ GB/s}} = 86 \text{ ms}$$

    **Conclusion:** Tensor parallelism over InfiniBand is 18× slower than NVLink. This is why TP stays within NVLink domains.

3. Why does pipeline parallelism use point-to-point communication rather than collectives? What would happen if each stage had to AllReduce with all other stages?

??? success "Solution"
    **Why point-to-point:**

    Pipeline parallelism has a linear data flow: Stage 0 → Stage 1 → Stage 2 → ... → Stage N-1.

    Each stage only needs to:

    - **Receive** activations from the previous stage
    - **Send** activations to the next stage

    This is inherently **point-to-point** (Send/Recv), not collective.

    **If AllReduce were required:**

    | Problem | Impact |
    |---------|--------|
    | Global synchronization | All stages wait for slowest |
    | Latency scales with P | $O(P)$ instead of $O(1)$ |
    | Defeats pipelining | Can't overlap stages |
    | Unnecessary data movement | Stages don't need each other's data |

    **Quantitative example (8 stages):**

    - Point-to-point: 1 hop latency (~1-5 μs per activation transfer)
    - AllReduce: 7 hops minimum, all stages synchronized

    Pipeline bubbles would grow from $\frac{P-1}{M+P-1}$ to a much larger fraction, potentially overwhelming useful work for typical $M$.

    **Key insight:** PP exploits the *sequential* nature of neural network layers. Collectives are for *parallel* operations on the same data.

## Key Takeaways

1. **Clusters are hierarchical**: NVLink, PCIe, NICs, and switches define bandwidth tiers.
2. **Parallelism placement must respect bandwidth gaps**: TP within nodes, DP across nodes, PP across stages.
3. **Topology awareness prevents hidden slowdowns**: mismatching strategy to link speeds can add 10–20× overhead.