---
title: "Collective Primitives and Properties"
subtitle: "The Building Blocks of Distributed Communication"
---

<div class="chapter-opener" markdown>
Every distributed training algorithm is built from a small set of collective operations. Understanding their algebraic properties reveals why certain compositions work and others fail.
</div>

<div class="investigation-question" markdown>
**The Question**: Why is AllReduce = ReduceScatter ∘ AllGather? What properties must hold for this decomposition to be valid? What happens when these properties are violated?
</div>

## The Seven Primitives

Collective operations are the vocabulary of distributed systems. All distributed training communication can be expressed using these primitives.

### 1. Broadcast

One process sends data to all others.

```
Before:              After:
P0: [A B C D]        P0: [A B C D]
P1: [. . . .]   →    P1: [A B C D]
P2: [. . . .]        P2: [A B C D]
P3: [. . . .]        P3: [A B C D]
```

**Use case**: Distributing model weights at initialization.

**Volume**: $n$ bytes total sent by root.

### 2. Reduce

All processes contribute data; one receives the reduced result.

```
Before:              After (sum):
P0: [1 2 3 4]        P0: [10 20 30 40]
P1: [2 4 6 8]   →    P1: [. . . .]
P2: [3 6 9 12]       P2: [. . . .]
P3: [4 8 12 16]      P3: [. . . .]
```

**Use case**: Aggregating gradients to a parameter server.

**Volume**: Each non-root sends $n$ bytes.

### 3. AllReduce

All processes contribute; all receive the reduced result.

```
Before:              After (sum):
P0: [1 2 3 4]        P0: [10 20 30 40]
P1: [2 4 6 8]   →    P1: [10 20 30 40]
P2: [3 6 9 12]       P2: [10 20 30 40]
P3: [4 8 12 16]      P3: [10 20 30 40]
```

**Use case**: Synchronizing gradients in data parallelism.

**Volume**: $2(P-1)/P \cdot n$ bytes per process (ring algorithm).

### 4. Scatter

One process distributes different chunks to each process.

```
Before:              After:
P0: [A B C D]        P0: [A]
P1: [. . . .]   →    P1: [B]
P2: [. . . .]        P2: [C]
P3: [. . . .]        P3: [D]
```

**Use case**: Distributing different data batches.

**Volume**: Root sends $n$ bytes total.

### 5. Gather

Each process contributes a chunk; one receives concatenated result.

```
Before:              After:
P0: [A]              P0: [A B C D]
P1: [B]         →    P1: [.]
P2: [C]              P2: [.]
P3: [D]              P3: [.]
```

**Use case**: Collecting outputs from workers.

**Volume**: Root receives $n$ bytes total.

### 6. AllGather

Each process contributes a chunk; all receive concatenated result.

```
Before:              After:
P0: [A]              P0: [A B C D]
P1: [B]         →    P1: [A B C D]
P2: [C]              P2: [A B C D]
P3: [D]              P3: [A B C D]
```

**Use case**: Reconstructing full tensors in ZeRO-3.

**Volume**: Each process receives $(P-1) \cdot n/P$ bytes.

### 7. ReduceScatter

All processes contribute full data; each receives a shard of the reduced result.

```
Before:              After (sum):
P0: [1 2 3 4]        P0: [10]
P1: [2 4 6 8]   →    P1: [20]
P2: [3 6 9 12]       P2: [30]
P3: [4 8 12 16]      P3: [40]
```

**Use case**: Gradient reduction + sharding in ZeRO.

**Volume**: Each process sends $(P-1)/P \cdot n$ bytes.

### Bonus: AlltoAll

Each process sends different data to each other process.

```
Before:              After:
P0: [A0 A1 A2 A3]    P0: [A0 B0 C0 D0]
P1: [B0 B1 B2 B3] →  P1: [A1 B1 C1 D1]
P2: [C0 C1 C2 C3]    P2: [A2 B2 C2 D2]
P3: [D0 D1 D2 D3]    P3: [A3 B3 C3 D3]
```

**Use case**: Expert routing in Mixture-of-Experts.

**Volume**: Each process sends and receives $(P-1)/P \cdot n$ bytes.

## Summary Table

| Primitive | Input/Output | Data Movement | Optimal Complexity |
|-----------|--------------|---------------|-------------------|
| Broadcast | 1 → N copies | root → all | $\alpha \log P + n/\beta$ |
| Reduce | N → 1 sum | all → root | $\alpha \log P + n/\beta$ |
| AllReduce | N → N sum | all ↔ all | $2\alpha(P-1) + 2\frac{P-1}{P}\frac{n}{\beta}$ |
| Scatter | 1 → N shards | root → all | $\alpha \log P + \frac{P-1}{P}\frac{n}{\beta}$ |
| Gather | N → 1 concat | all → root | $\alpha \log P + \frac{P-1}{P}\frac{n}{\beta}$ |
| AllGather | N → N concat | all ↔ all | $\alpha(P-1) + \frac{P-1}{P}\frac{n}{\beta}$ |
| ReduceScatter | N → N sharded sum | all ↔ all | $\alpha(P-1) + \frac{P-1}{P}\frac{n}{\beta}$ |
| AlltoAll | N → N transpose | all ↔ all | $\alpha(P-1) + \frac{P-1}{P}\frac{n}{\beta}$ |

## Algebraic Properties

Collective operations form an algebra. Understanding this algebra reveals valid transformations.

### Associativity

A binary operation $\oplus$ is **associative** if:

$$(a \oplus b) \oplus c = a \oplus (b \oplus c)$$

**Why it matters**: Reduction can happen in any grouping order.

```
Tree reduction (associative op):
      sum
     /   \
   sum   sum
   / \   / \
  P0 P1 P2 P3

Valid if (g0 ⊕ g1) ⊕ (g2 ⊕ g3) = ((g0 ⊕ g1) ⊕ g2) ⊕ g3
```

**Floating-point caveat**: Addition is not perfectly associative!

$$\text{float}((a + b) + c) \neq \text{float}(a + (b + c))$$

Example: $a = 10^{20}$, $b = -10^{20}$, $c = 1$
- $(a + b) + c = 0 + 1 = 1$
- $a + (b + c) = 10^{20} + (-10^{20} + 1) = 10^{20} - 10^{20} = 0$

This affects reproducibility in distributed training.

### Commutativity

A binary operation $\oplus$ is **commutative** if:

$$a \oplus b = b \oplus a$$

**Why it matters**: Order of contributions doesn't affect result.

**Examples**:

- Sum: commutative ✓
- Max: commutative ✓
- Concatenation: not commutative ✗

### Identity Element

An **identity element** $e$ satisfies:

$$a \oplus e = e \oplus a = a$$

**Examples**:

- Sum: $e = 0$
- Product: $e = 1$
- Max: $e = -\infty$

The identity element initializes buffers in reduction operations.

### Closure Under Composition

If we compose two collectives, do we get another valid operation?

$$\text{AllReduce} = \text{AllGather} \circ \text{ReduceScatter}$$

This decomposition is fundamental to understanding collective implementations.

## The Decomposition Theorem

**Theorem**: For an associative, commutative reduction operation $\oplus$:

$$\text{AllReduce}(x_0, x_1, ..., x_{P-1}) = \text{AllGather}(\text{ReduceScatter}(x_0, x_1, ..., x_{P-1}))$$

**Proof**:

Let each process $i$ have vector $x_i$ of size $n$.

**Step 1**: ReduceScatter partitions each vector into $P$ chunks and reduces chunk $j$ to process $j$:

$$y_j = \bigoplus_{i=0}^{P-1} x_i^{(j)}$$

where $x_i^{(j)}$ is chunk $j$ of process $i$'s data.

After ReduceScatter, process $j$ holds $y_j$ of size $n/P$.

**Step 2**: AllGather collects all $y_j$ to all processes:

Each process receives $[y_0, y_1, ..., y_{P-1}]$.

**Result**:
$$[y_0, y_1, ..., y_{P-1}] = \left[\bigoplus_{i} x_i^{(0)}, \bigoplus_{i} x_i^{(1)}, ..., \bigoplus_{i} x_i^{(P-1)}\right]$$

By commutativity and associativity:
$$= \bigoplus_{i} x_i$$

This equals the result of AllReduce. $\square$

### Why This Matters

1. **Implementation flexibility**: Can implement AllReduce as RS + AG
2. **Bandwidth optimality**: Both RS and AG achieve bandwidth lower bound
3. **Fusion opportunities**: Can fuse computation between RS and AG
4. **Understanding ZeRO**: ZeRO exploits this decomposition directly

## Inverse Operations

Some collectives have natural inverses:

| Operation | Inverse |
|-----------|---------|
| Broadcast | Reduce |
| Scatter | Gather |
| AllGather | ReduceScatter (with reshape) |

**Gather ∘ Scatter = Identity** (on root):
```
Scatter: P0:[ABCD] → P0:[A], P1:[B], P2:[C], P3:[D]
Gather:  P0:[A], P1:[B], P2:[C], P3:[D] → P0:[ABCD]
```

## Collective Hierarchies

Collectives can be implemented using simpler primitives:

```
                AllReduce
                /       \
        ReduceScatter   AllGather
           /    \        /    \
       Reduce  Scatter  Gather  Broadcast
          \      |        |      /
           \     |        |     /
            Send/Recv (point-to-point)
```

Each level adds abstraction. Lower levels provide building blocks for higher levels.

## Implementation: NCCL

NVIDIA Collective Communications Library (NCCL) provides optimized implementations:

```cpp
// AllReduce
ncclAllReduce(sendbuff, recvbuff, count, datatype,
              ncclSum, comm, stream);

// ReduceScatter
ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype,
                   ncclSum, comm, stream);

// AllGather
ncclAllGather(sendbuff, recvbuff, sendcount, datatype,
              comm, stream);
```

**NCCL automatically selects algorithms** based on:

- Message size (ring vs tree vs direct)
- Topology (NVLink vs PCIe vs network)
- Number of GPUs

### Algorithm Selection

| Size | Algorithm | Reason |
|------|-----------|--------|
| < 256 KB | Tree | Latency-bound, minimize hops |
| 256 KB - 4 MB | Hybrid | Balance latency and bandwidth |
| > 4 MB | Ring | Bandwidth-bound, maximize throughput |

## Consistency Models

### Bulk Synchronous Parallel (BSP)

All processes complete collective before any proceeds.

```python
# Implicit barrier in collective
gradients = allreduce(local_gradients)
# All processes have same gradients here
weights -= lr * gradients
```

**Pros**: Deterministic, easy to reason about
**Cons**: Slowest process determines speed (stragglers)

### Asynchronous

Processes don't wait; use stale values.

```python
# Non-blocking
handle = allreduce_async(local_gradients)
# ... do other work ...
gradients = handle.wait()  # Eventually complete
```

**Pros**: No straggler problem
**Cons**: Convergence may be affected by staleness

## Exercises

1. **Decomposition verification**: Implement AllReduce using ReduceScatter followed by AllGather. Verify they produce the same result.

2. **Associativity test**: Write code that demonstrates floating-point non-associativity. Find inputs where tree-reduction gives different results than sequential reduction.

3. **Communication volume**: For AllReduce of an $n$-byte tensor across $P$ processes:

   - Calculate total bytes sent using naive reduce + broadcast
   - Calculate total bytes sent using ring AllReduce
   - What's the improvement factor?

4. **Inverse operations**: Prove that for any vector $x$: $\text{Gather}(\text{Scatter}(x)) = x$ (on root).

5. **AlltoAll analysis**: Derive the communication volume for AlltoAll. Why is it the same as AllGather despite different semantics?

6. **Algorithm selection**: Given α = 1μs, β = 100 GB/s, at what message size does ring AllReduce become faster than tree AllReduce for P = 64?

??? success "Solution"
    **AllReduce using ReduceScatter + AllGather:**

    ```python
    import numpy as np

    def reduce_scatter(data, P, op=np.sum):
        """Reduce across processes, scatter result shards."""
        # Stack all data: shape (P, n)
        stacked = np.stack(data)
        # Reduce along process dimension
        reduced = op(stacked, axis=0)
        # Scatter: each process gets 1/P of result
        chunk_size = len(reduced) // P
        return [reduced[i*chunk_size:(i+1)*chunk_size] for i in range(P)]

    def all_gather(shards, P):
        """Gather all shards to all processes."""
        gathered = np.concatenate(shards)
        return [gathered.copy() for _ in range(P)]

    def allreduce_via_decomposition(data, P):
        """AllReduce = AllGather ∘ ReduceScatter"""
        shards = reduce_scatter(data, P)
        return all_gather(shards, P)

    def allreduce_direct(data, P, op=np.sum):
        """Direct AllReduce for comparison."""
        stacked = np.stack(data)
        reduced = op(stacked, axis=0)
        return [reduced.copy() for _ in range(P)]

    # Test
    P = 4
    data = [np.array([1, 2, 3, 4]),
            np.array([2, 4, 6, 8]),
            np.array([3, 6, 9, 12]),
            np.array([4, 8, 12, 16])]

    decomposed = allreduce_via_decomposition(data, P)
    direct = allreduce_direct(data, P)

    # Verify: both should give [10, 20, 30, 40] on all processes
    assert all(np.array_equal(d, [10, 20, 30, 40]) for d in decomposed)
    assert all(np.array_equal(d, [10, 20, 30, 40]) for d in direct)
    print("✓ Decomposition verified!")
    ```

    **Result:** Both produce `[10, 20, 30, 40]` on all 4 processes.

??? success "Solution"
    **Demonstrating floating-point non-associativity:**

    ```python
    import numpy as np

    def sequential_sum(values):
        """Left-to-right sequential reduction."""
        result = values[0]
        for v in values[1:]:
            result = result + v
        return result

    def tree_sum(values):
        """Pairwise tree reduction."""
        if len(values) == 1:
            return values[0]
        pairs = []
        for i in range(0, len(values), 2):
            if i + 1 < len(values):
                pairs.append(values[i] + values[i+1])
            else:
                pairs.append(values[i])
        return tree_sum(pairs)

    # Example 1: Catastrophic cancellation
    a = np.float32(1e20)
    b = np.float32(-1e20)
    c = np.float32(1.0)

    seq_result = (a + b) + c  # = 0 + 1 = 1
    alt_result = a + (b + c)  # = 1e20 + (-1e20) = 0 (c absorbed)

    print(f"Sequential: (a + b) + c = {seq_result}")
    print(f"Alternate:  a + (b + c) = {alt_result}")
    print(f"Difference: {abs(seq_result - alt_result)}")

    # Example 2: Many small values
    np.random.seed(42)
    values = [np.float32(1e-8) for _ in range(1000000)]
    values.append(np.float32(1.0))

    seq = sequential_sum(values)
    tree = tree_sum(values)
    exact = 1.0 + 1000000 * 1e-8  # = 1.01

    print(f"\nSequential sum: {seq}")
    print(f"Tree sum: {tree}")
    print(f"Exact value: {exact}")
    ```

    **Output:**
    ```
    Sequential: (a + b) + c = 1.0
    Alternate:  a + (b + c) = 0.0
    Difference: 1.0

    Sequential sum: 1.0100002
    Tree sum: 1.0099999
    Exact value: 1.01
    ```

    **Key insight:** Tree reduction can give different results than sequential reduction due to different grouping of operations. Neither matches the exact mathematical result.

??? success "Solution"
    **Communication volume comparison:**

    **Naive Reduce + Broadcast:**

    - Reduce: Each of $P-1$ processes sends $n$ bytes to root
      - Total sent: $(P-1) \times n$
    - Broadcast: Root sends $n$ bytes to each of $P-1$ processes
      - Total sent: $(P-1) \times n$ (or $n \log P$ with tree)

    **Total naive:** $2(P-1) \times n$ bytes (or $(P-1 + \log P) \times n$ with tree broadcast)

    **Ring AllReduce:**

    - Each process sends: $2 \times \frac{P-1}{P} \times n$ bytes
    - Total across all processes: $2 \times \frac{P-1}{P} \times n \times P = 2(P-1)n$

    **Wait—same total?** Yes, but the difference is in **per-process** and **per-link** load:

    | Metric | Naive | Ring |
    |--------|-------|------|
    | Root sends | $(P-1)n$ | $\frac{2(P-1)}{P}n$ |
    | Root receives | $(P-1)n$ | $\frac{2(P-1)}{P}n$ |
    | Max link load | $(P-1)n$ | $\frac{n}{P}$ per step |

    **Improvement factor for root bottleneck:**
    $$\frac{(P-1)n}{\frac{2(P-1)}{P}n} = \frac{P}{2}$$

    For $P = 64$: Ring reduces root load by **32×**.

    **Key insight:** Total bytes are similar, but ring distributes load evenly across processes and time, eliminating the root bottleneck.

??? success "Solution"
    **Proof that Gather(Scatter(x)) = x on root:**

    **Scatter operation:** Root sends chunk $i$ to process $i$:
    $$\text{Scatter}([x_0, x_1, ..., x_{P-1}]) \to (P_0: x_0, P_1: x_1, ..., P_{P-1}: x_{P-1})$$

    **Gather operation:** Root receives chunk from each process and concatenates:
    $$\text{Gather}(P_0: y_0, P_1: y_1, ..., P_{P-1}: y_{P-1}) \to [y_0, y_1, ..., y_{P-1}]$$

    **Composition:**
    $$\text{Gather}(\text{Scatter}([x_0, x_1, ..., x_{P-1}]))$$
    $$= \text{Gather}(P_0: x_0, P_1: x_1, ..., P_{P-1}: x_{P-1})$$
    $$= [x_0, x_1, ..., x_{P-1}]$$
    $$= x$$

    **Therefore:** $\text{Gather} \circ \text{Scatter} = \text{Identity}$ (on root) $\square$

    **Note:** This only holds on the root process. Other processes don't have the full result after Gather.

??? success "Solution"
    **AlltoAll communication volume:**

    **Setup:** Each process $i$ has data $[D_{i,0}, D_{i,1}, ..., D_{i,P-1}]$ where $D_{i,j}$ is destined for process $j$.

    **Data movement:** Process $i$ sends $D_{i,j}$ to process $j$ for all $j \neq i$.

    **Per process:**
    - Sends: $(P-1)$ chunks of size $n/P$ = $\frac{P-1}{P} \times n$ bytes
    - Receives: $(P-1)$ chunks of size $n/P$ = $\frac{P-1}{P} \times n$ bytes

    **Total per process:** $\frac{P-1}{P} \times n$ sent + $\frac{P-1}{P} \times n$ received

    **AllGather comparison:**

    - Each process contributes $n/P$, receives $n$ total
    - Sends: $\frac{P-1}{P} \times n$ bytes (sends its chunk to $P-1$ others)
    - Receives: $\frac{P-1}{P} \times n$ bytes (receives from $P-1$ others)

    **Same volume!** Both move $\frac{P-1}{P} \times n$ bytes per process.

    **Why same despite different semantics?**

    | Operation | Sends | Receives | Pattern |
    |-----------|-------|----------|---------|
    | AllGather | Same data to all | Different data from each | 1-to-many |
    | AlltoAll | Different data to each | Different data from each | Many-to-many |

    The total data moved is identical; only the **pattern** differs. AlltoAll is a "personalized" AllGather where each destination gets unique data.

??? success "Solution"
    **Ring vs Tree crossover for P = 64:**

    **Given:** $\alpha = 1\mu s = 10^{-6}$ s, $\beta = 100$ GB/s $= 10^{11}$ bytes/s, $P = 64$

    **Ring AllReduce time:**
    $$T_{\text{ring}} = 2(P-1)\alpha + 2 \cdot \frac{P-1}{P} \cdot \frac{n}{\beta}$$
    $$= 2(63)(10^{-6}) + 2 \cdot \frac{63}{64} \cdot \frac{n}{10^{11}}$$
    $$= 126 \mu s + 1.97 \cdot 10^{-11} \cdot n$$

    **Tree AllReduce time:**
    $$T_{\text{tree}} = 2\log_2 P \cdot \alpha + 2\log_2 P \cdot \frac{n}{\beta}$$
    $$= 2(6)(10^{-6}) + 2(6) \cdot \frac{n}{10^{11}}$$
    $$= 12 \mu s + 1.2 \cdot 10^{-10} \cdot n$$

    **Set equal to find crossover:**
    $$126 \mu s + 1.97 \times 10^{-11} n = 12 \mu s + 1.2 \times 10^{-10} n$$
    $$114 \mu s = (1.2 \times 10^{-10} - 1.97 \times 10^{-11}) n$$
    $$114 \times 10^{-6} = 1.003 \times 10^{-10} \cdot n$$
    $$n = \frac{114 \times 10^{-6}}{1.003 \times 10^{-10}} = \boxed{1.14 \text{ MB}}$$

    **Conclusion:**
    - Messages < 1.14 MB: Use **tree** (lower latency)
    - Messages > 1.14 MB: Use **ring** (better bandwidth)

    | Message Size | Ring Time | Tree Time | Winner |
    |-------------|-----------|-----------|--------|
    | 100 KB | 128 μs | 24 μs | Tree |
    | 1 MB | 146 μs | 132 μs | Tree |
    | 10 MB | 324 μs | 1.2 ms | Ring |
    | 100 MB | 2.1 ms | 12 ms | Ring |

## Key Takeaways

1. **Seven primitives suffice**: All distributed training communication uses these building blocks.

2. **AllReduce = ReduceScatter + AllGather**: This decomposition is fundamental.

3. **Algebraic properties matter**: Associativity enables tree reductions; commutativity enables order-independence.

4. **Floating-point breaks associativity**: This affects reproducibility.

5. **Algorithm selection depends on size**: Small messages → tree; large messages → ring.

6. **NCCL handles complexity**: Automatic algorithm selection based on topology and size.
