---
title: "Ring and Tree Algorithms"
subtitle: "Bandwidth-Optimal and Latency-Optimal Collectives"
---

::: {.chapter-opener}
The same logical operation—AllReduce—can be implemented many ways. Ring algorithms optimize for bandwidth; tree algorithms optimize for latency. Choosing correctly depends on message size.
:::

::: {.investigation-question}
**The Question**: Why is ring AllReduce bandwidth-optimal? Can we prove it achieves the theoretical minimum communication volume? And when should we use tree algorithms instead?
:::

## The Algorithm Design Space

AllReduce takes $P$ input vectors and produces $P$ copies of their sum. The question: how do we move and combine this data?

Two fundamental approaches:

1. **Ring**: Arrange processes in a logical ring, pass data around
2. **Tree**: Arrange processes in a logical tree, reduce up then broadcast down

These optimize for different regimes in the α-β model.

## Ring AllReduce

### The Two-Phase Algorithm

Ring AllReduce consists of two phases:

**Phase 1: ReduceScatter** — Each process ends with 1/P of the final result

**Phase 2: AllGather** — Each process collects all pieces

### Phase 1: ReduceScatter via Ring

Partition each process's data into $P$ chunks. In $P-1$ steps, each process:
1. Sends one chunk clockwise
2. Receives one chunk from counterclockwise neighbor
3. Reduces received chunk with local chunk

```
Initial state (P=4, data partitioned into 4 chunks):
P0: [A0 A1 A2 A3]    P1: [B0 B1 B2 B3]
P2: [C0 C1 C2 C3]    P3: [D0 D1 D2 D3]

Step 1: Each sends chunk[i] to (i+1) mod P, receives from (i-1) mod P
P0: sends A3→P1, recv D0     P1: sends B0→P2, recv A3
P2: sends C1→P3, recv B0     P3: sends D2→P0, recv C1

After Step 1 (with reduction):
P0: [A0+D0 A1 A2 A3]         P1: [B0 B1+A3 B2 B3]
P2: [C0 C1 C2+B0 C3]         P3: [D0 D1 D2 D3+C1]
    ↑reduced                      ↑reduced

Step 2: Send next chunk (the one just reduced)
P0: sends (A0+D0)→P1, recv (D3+C2)
...

After Step 2:
P0: [A0+D0 A1 A2+D3+C2 A3]
P1: [B0+A0+D0 B1+A3 B2 B3]
...

After P-1=3 steps:
P0: has complete reduction of chunk 0: A0+B0+C0+D0
P1: has complete reduction of chunk 1: A1+B1+C1+D1
P2: has complete reduction of chunk 2: A2+B2+C2+D2
P3: has complete reduction of chunk 3: A3+B3+C3+D3
```

Each process now holds 1/P of the fully reduced result.

### Phase 2: AllGather via Ring

In $P-1$ steps, each process:
1. Sends its reduced chunk clockwise
2. Receives a reduced chunk from counterclockwise neighbor
3. Stores received chunk (no reduction needed)

```
After ReduceScatter:
P0: [Σ0 . . .]    P1: [. Σ1 . .]
P2: [. . Σ2 .]    P3: [. . . Σ3]

Step 1: Each sends its chunk clockwise
P0: sends Σ0→P1    P1: sends Σ1→P2
P2: sends Σ2→P3    P3: sends Σ3→P0

After Step 1:
P0: [Σ0 . . Σ3]    P1: [Σ0 Σ1 . .]
P2: [. Σ1 Σ2 .]    P3: [. . Σ2 Σ3]

Step 2: Send what was just received
...

After P-1=3 steps:
P0: [Σ0 Σ1 Σ2 Σ3]    P1: [Σ0 Σ1 Σ2 Σ3]
P2: [Σ0 Σ1 Σ2 Σ3]    P3: [Σ0 Σ1 Σ2 Σ3]
```

### Communication Analysis

**Per step**:
- Each process sends: $n/P$ bytes
- Each process receives: $n/P$ bytes

**Total steps**: $2(P-1)$ (ReduceScatter + AllGather)

**Total per process**:
$$\text{Send} = \text{Recv} = 2(P-1) \cdot \frac{n}{P} = 2 \cdot \frac{P-1}{P} \cdot n$$

**Time using α-β model**:
$$T_{\text{ring}} = 2(P-1) \cdot \alpha + 2 \cdot \frac{P-1}{P} \cdot \frac{n}{\beta}$$

For large $P$:
$$T_{\text{ring}} \approx 2P\alpha + \frac{2n}{\beta}$$

### Bandwidth Optimality Proof

**Theorem**: Ring AllReduce is bandwidth-optimal.

**Proof**:

Consider the total data that must be communicated. The AllReduce result is a vector of size $n$ that must exist on all $P$ processes.

**Lower bound argument**:

Consider any process $p$. Initially, $p$ has only its own contribution (size $n$). After AllReduce, $p$ must have the complete reduced result (size $n$), which incorporates data from all other $P-1$ processes.

The minimum data $p$ must receive: at least $(P-1)/P \cdot n$ bytes, because each of the other $P-1$ processes contributes $n/P$ bytes to the final result that $p$ doesn't initially have.

Similarly, $p$ must send at least $(P-1)/P \cdot n$ bytes so that other processes can compute the portions of the result that depend on $p$'s contribution.

Total communication per process: at least $2(P-1)/P \cdot n$ bytes.

**Ring AllReduce achieves exactly this bound**. $\square$

### Bidirectional Ring

The standard ring uses unidirectional communication. Bidirectional ring sends different chunks in each direction simultaneously:

```
Standard ring:
P0 → P1 → P2 → P3 → P0

Bidirectional ring:
P0 ⟷ P1 ⟷ P2 ⟷ P3 ⟷ P0
```

**Benefit**: Utilizes both directions of full-duplex links.

**Time**:
$$T_{\text{bidir}} = (P-1) \cdot \alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

Half the steps of unidirectional ring.

## Tree AllReduce

### The Two-Phase Algorithm

Tree AllReduce also has two phases:

**Phase 1: Reduce** — Aggregate data to root via tree reduction

**Phase 2: Broadcast** — Distribute result from root via tree broadcast

### Recursive Halving (Reduce Phase)

Arrange processes as a binary tree. In $\log_2 P$ steps:

```
Step 1: Pairs reduce (P0↔P1, P2↔P3, ...)
        P0 ←── P1        P2 ←── P3
        P0 holds sum     P2 holds sum
        of (P0,P1)       of (P2,P3)

Step 2: Pairs of pairs reduce
        P0 ←────────── P2
        P0 holds sum of all

After log₂(P) steps: Root has complete reduction
```

At each step:
- Half the active processes send their entire data
- Other half receives and reduces

### Recursive Doubling (Broadcast Phase)

Reverse the tree structure:

```
Step 1: Root sends to partner
        P0 ──────────→ P2
        Both have sum

Step 2: Each sends to partner
        P0 ──→ P1      P2 ──→ P3
        All have sum
```

### Communication Analysis

**Reduce phase**:
- $\log_2 P$ steps
- Each step: send/recv $n$ bytes (full data)
- Time: $\log_2 P \cdot (\alpha + n/\beta)$

**Broadcast phase**:
- $\log_2 P$ steps
- Each step: send/recv $n$ bytes
- Time: $\log_2 P \cdot (\alpha + n/\beta)$

**Total**:
$$T_{\text{tree}} = 2 \log_2 P \cdot \alpha + 2 \log_2 P \cdot \frac{n}{\beta}$$

### Latency Optimality

**Theorem**: Tree AllReduce is latency-optimal.

**Proof**:

Consider the information flow constraint. Process $P_{P-1}$'s contribution must reach process $P_0$, and $P_0$'s contribution must reach $P_{P-1}$.

In any communication graph, the minimum number of hops between two arbitrary processes is $\Omega(\log P)$ when using only pairwise communication (each step involves pairs).

The tree achieves exactly $\log_2 P$ steps in each direction. $\square$

## Comparison: Ring vs Tree

| Aspect | Ring | Tree |
|--------|------|------|
| Latency term | $2(P-1) \cdot \alpha$ | $2\log_2 P \cdot \alpha$ |
| Bandwidth term | $2 \cdot \frac{P-1}{P} \cdot \frac{n}{\beta}$ | $2\log_2 P \cdot \frac{n}{\beta}$ |
| Latency-optimal | No | Yes |
| Bandwidth-optimal | Yes | No |

**Critical observation**:
- Ring: bandwidth term ≈ $2n/\beta$ (independent of $P$)
- Tree: bandwidth term = $2\log_2 P \cdot n/\beta$ (grows with $P$)

For large $P$, tree uses $\log_2 P$ times more bandwidth!

### The Crossover Point

Setting $T_{\text{ring}} = T_{\text{tree}}$:

$$2(P-1)\alpha + \frac{2(P-1)}{P} \cdot \frac{n}{\beta} = 2\log_2 P \cdot \alpha + 2\log_2 P \cdot \frac{n}{\beta}$$

Solving for $n$:

$$n^* = \frac{2\alpha\beta(P-1 - \log_2 P)}{2\log_2 P - 2(P-1)/P}$$

For large $P$:
$$n^* \approx \frac{\alpha\beta \cdot P}{\log_2 P - 1}$$

**Example**: $P = 64$, $\alpha = 1\mu s$, $\beta = 100$ GB/s

$$n^* \approx \frac{10^{-6} \times 10^{11} \times 64}{6 - 1} = \frac{6.4 \times 10^6}{5} = 1.28 \text{ MB}$$

- Messages < 1.28 MB: use tree
- Messages > 1.28 MB: use ring

### Visualization

```
Time
  │
  │     Tree: T = 2log(P)·α + 2log(P)·n/β
  │              ╱
  │             ╱
  │            ╱
  │           ╱      Ring: T = 2(P-1)·α + 2·(P-1)/P·n/β
  │          ╱       ╱
  │    ─────●──────╱──────────────
  │        ╱     ╱ ↑
  │       ╱    ╱   crossover
  │      ╱   ╱
  │     ╱  ╱
  │    ╱ ╱
  │   ╱╱
  └──●────────────────────────────→ n (message size)
     0
```

At small $n$: Ring pays high latency penalty $(P-1)$ steps
At large $n$: Tree pays high bandwidth penalty $(\log P)$ factor

## Recursive Halving-Doubling (Rabenseifner's Algorithm)

Combines best of both approaches:

### Algorithm

**Phase 1: Recursive Halving + Reduction**

Like tree reduce, but only exchange $n/2$ data at each step:

```
Step 1: Pairs exchange opposite halves and reduce
        P0: [A B] ↔ P1: [C D]
        P0 sends [B], recvs [D], computes [A A+C+D]
        P1 sends [A], recvs [C], computes [B+C B]

Wait, this isn't quite right. Let me be more precise:

Step 1: P0 sends second half [B] to P1
        P1 sends first half [C] to P0
        P0: reduces received C with local A → [A+C, B]
        P1: reduces received B with local D → [C, B+D]
```

Actually, the Rabenseifner algorithm for reduce-scatter:

```
Step k (of log₂P steps):
- Pair up with process distance 2^(log₂P - k) away
- Send half the data you're responsible for
- Receive the other half
- Reduce locally

After log₂P steps: each process has 1/P of reduced result
```

**Phase 2: Recursive Doubling (AllGather)**

Reverse the communication pattern to collect all pieces.

### Analysis

$$T_{\text{RHD}} = 2\log_2 P \cdot \alpha + \frac{2(P-1)}{P} \cdot \frac{n}{\beta}$$

**Best of both worlds**:
- Latency of tree: $O(\log P)$ steps
- Bandwidth of ring: $O(n)$ total data

**Constraint**: Requires $P$ to be a power of 2.

## Hierarchical Algorithms

Modern clusters have hierarchy: GPUs within a node, nodes within a rack, racks within a cluster.

### 2D Ring (Ring-Ring)

Arrange processes in a 2D grid. AllReduce in two phases:

**Phase 1**: AllReduce within rows (intra-node, fast)
**Phase 2**: AllReduce across rows (inter-node, slow)

```
         Node 0           Node 1           Node 2
        ┌───────────┐   ┌───────────┐   ┌───────────┐
        │ G0 ↔ G1 ↔│G2↔│ G3 ↔ G4 ↔│G5↔│ G6 ↔ G7 │
        │  ↕    ↕    │   │  ↕    ↕    │   │  ↕    ↕   │
        │ G8 ↔ G9 ↔│...│           ...│   │          │
        └───────────┘   └───────────┘   └───────────┘
              ↑               ↑               ↑
              └───────────────┼───────────────┘
                    Inter-node ring
```

**Phase 1** (intra-node): Ring AllReduce among GPUs in each node
- Uses NVLink (high bandwidth, low latency)
- Each GPU gets 1/G portion of result (G = GPUs per node)

**Phase 2** (inter-node): Ring AllReduce of corresponding chunks across nodes
- Uses network (lower bandwidth, higher latency)
- Only 1/G of data crosses network

**Analysis**: Let $G$ = GPUs per node, $N$ = nodes, $P = GN$

$$T_{\text{intra}} = 2(G-1)\alpha_{\text{NV}} + \frac{2(G-1)}{G} \cdot \frac{n}{\beta_{\text{NV}}}$$

$$T_{\text{inter}} = 2(N-1)\alpha_{\text{net}} + \frac{2(N-1)}{N} \cdot \frac{n/G}{\beta_{\text{net}}}$$

**Key insight**: Inter-node phase transfers only $n/G$ bytes, reducing network bandwidth requirement by factor of $G$.

### Hierarchical Ring-Tree

- **Intra-node**: Ring (high bandwidth utilization of NVLink)
- **Inter-node**: Tree (low latency for small messages after local reduction)

```
Within each node: Ring AllReduce
Between nodes: Tree AllReduce of local results
```

### NCCL's Double Binary Tree

NCCL uses two overlapping binary trees to fully utilize bidirectional links:

```
Tree 1:        Tree 2:
    0              7
   / \            / \
  1   2          6   5
 / \   \        /   / \
3   4   5      4   3   0
               ...
```

Both trees run simultaneously, each handling half the data. This achieves:
- $\log P$ latency (tree property)
- Full bidirectional bandwidth utilization

## Algorithm Selection in Practice

### NCCL's Heuristics

NCCL selects algorithms based on:

1. **Message size**
2. **Number of GPUs**
3. **Network topology**

Typical thresholds:

| Message Size | Algorithm |
|-------------|-----------|
| < 256 KB | Tree (latency-bound) |
| 256 KB - 4 MB | Hybrid selection |
| > 4 MB | Ring (bandwidth-bound) |

### Environment Variables

```bash
# Force specific algorithm
export NCCL_ALGO=Ring   # or Tree, or CollnetDirect
export NCCL_ALGO=Tree

# Set protocol
export NCCL_PROTO=Simple  # or LL (low-latency) or LL128

# Topology detection
export NCCL_TOPO_FILE=/path/to/topology.xml
```

## Implementation: Ring AllReduce

```python
import torch
import torch.distributed as dist

def ring_allreduce(tensor, group):
    """Ring AllReduce implementation."""
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    # Partition tensor into world_size chunks
    chunks = tensor.chunk(world_size)
    chunks = list(chunks)  # Make mutable

    # Phase 1: ReduceScatter
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size

        send_to = (rank + 1) % world_size
        recv_from = (rank - 1) % world_size

        # Exchange chunks
        recv_buffer = torch.empty_like(chunks[recv_idx])

        send_op = dist.isend(chunks[send_idx], send_to, group)
        recv_op = dist.irecv(recv_buffer, recv_from, group)

        send_op.wait()
        recv_op.wait()

        # Reduce
        chunks[recv_idx] += recv_buffer

    # Phase 2: AllGather
    for step in range(world_size - 1):
        send_idx = (rank - step + 1) % world_size
        recv_idx = (rank - step) % world_size

        send_to = (rank + 1) % world_size
        recv_from = (rank - 1) % world_size

        recv_buffer = torch.empty_like(chunks[recv_idx])

        send_op = dist.isend(chunks[send_idx], send_to, group)
        recv_op = dist.irecv(recv_buffer, recv_from, group)

        send_op.wait()
        recv_op.wait()

        chunks[recv_idx] = recv_buffer

    # Reconstruct tensor
    return torch.cat(chunks)
```

## Bucket Fusion

For many small tensors (e.g., gradients), individual AllReduce is inefficient. Solution: bucket fusion.

```python
class BucketedAllReduce:
    def __init__(self, bucket_size_mb=25):
        self.bucket_size = bucket_size_mb * 1024 * 1024
        self.buckets = []
        self.current_bucket = []
        self.current_size = 0

    def add_tensor(self, tensor):
        tensor_size = tensor.numel() * tensor.element_size()

        if self.current_size + tensor_size > self.bucket_size:
            # Flush current bucket
            self._flush_bucket()

        self.current_bucket.append(tensor)
        self.current_size += tensor_size

    def _flush_bucket(self):
        if not self.current_bucket:
            return

        # Flatten all tensors into single buffer
        flat = torch.cat([t.view(-1) for t in self.current_bucket])

        # Single AllReduce
        dist.all_reduce(flat)

        # Unflatten back
        offset = 0
        for t in self.current_bucket:
            numel = t.numel()
            t.copy_(flat[offset:offset+numel].view(t.shape))
            offset += numel

        self.current_bucket = []
        self.current_size = 0
```

PyTorch DDP uses 25MB buckets by default.

## Exercises

1. **Ring correctness**: Trace through ring ReduceScatter with P=4 and data `P0=[1,2,3,4], P1=[5,6,7,8], P2=[9,10,11,12], P3=[13,14,15,16]`. What does each process hold after the phase?

2. **Crossover calculation**: For P=256, α=5μs, β=200 GB/s, calculate the crossover point between ring and tree.

3. **Hierarchical analysis**: You have 8 nodes with 8 GPUs each (64 total). Intra-node bandwidth is 600 GB/s (NVLink), inter-node is 100 GB/s. Compare total time for flat ring vs 2D ring for a 1GB AllReduce.

4. **Bucket sizing**: You have 1000 gradient tensors, each 1MB. With α=10μs, β=100 GB/s, compare:
   - 1000 individual AllReduce calls
   - Bucketed into 25MB chunks

5. **Power of 2 constraint**: Rabenseifner's algorithm requires $P = 2^k$. Describe how to handle P=6 (not a power of 2).

6. **Bidirectional ring**: Prove that bidirectional ring halves the number of steps while maintaining bandwidth optimality.

## Key Takeaways

1. **Ring is bandwidth-optimal**: Total data = $2(P-1)/P \cdot n$ per process.

2. **Tree is latency-optimal**: Only $\log_2 P$ communication rounds.

3. **Crossover exists**: Small messages → tree; large messages → ring.

4. **Hierarchical adapts to topology**: Match algorithm to network structure.

5. **Bucket fusion amortizes latency**: Combine small tensors before communicating.

6. **NCCL handles selection**: Automatic algorithm choice based on heuristics.
