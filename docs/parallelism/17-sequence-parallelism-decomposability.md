---
title: "Sequence Parallelism from Decomposability"
subtitle: "Sharding Along the Sequence Dimension"
---

<div class="chapter-opener" markdown>
When sequences grow to millions of tokens, even a single attention computation won't fit in memory. Sequence parallelism exploits the decomposability of attention to split along the sequence dimension.
</div>

<div class="investigation-question" markdown>
**The Question**: Attention is O(S²) in sequence length. For S=1M tokens, that's 10^12 attention scores. How do we compute this when no single GPU can hold the attention matrix?
</div>

## Why Sequence Parallelism?

As models process longer contexts—documents, codebases, video—the sequence length $S$ becomes the bottleneck.

### Memory Scaling with Sequence Length

For a transformer layer:

**Activation memory per layer**:
$$M_{\text{act}} = 2 \cdot b \cdot S \cdot H + b \cdot S \cdot S \cdot n_h$$

Where:
- First term: Input and output activations ($2 \times b \times S \times H$)
- Second term: Attention matrix ($b \times n_h \times S \times S$)

The attention matrix scales as $O(S^2)$.

**Example**: $S = 128K$, $b = 1$, $n_h = 32$, fp16:
$$M_{\text{attn}} = 1 \times 32 \times 128K \times 128K \times 2 = 1 \text{ TB}$$

No single GPU can hold this.

### The Sequence Dimension

Unlike batch or hidden dimensions, the sequence dimension appears in:

1. **Attention**: $Q K^T$ has shape $(S \times S)$
2. **Layer normalization**: Statistics computed per-token (independent)
3. **Feedforward**: Applied per-token (independent)
4. **Positional encoding**: Position-dependent

Only attention creates cross-token dependencies.

## Two Flavors of Sequence Parallelism

### 1. Megatron Sequence Parallelism

Reduces memory for LayerNorm and Dropout activations by distributing across the sequence dimension within tensor parallelism.

**Target**: Activation memory outside attention.

**Communication**: AllGather before attention, ReduceScatter after.

### 2. Context Parallelism (Ring Attention / Ulysses)

Distributes the attention computation itself across the sequence dimension.

**Target**: The $O(S^2)$ attention matrix.

**Communication**: Ring of P2P (Ring Attention) or AlltoAll (Ulysses).

Let's examine each in detail.

## Megatron Sequence Parallelism

Korthikanti et al. (2022) introduced sequence parallelism for memory efficiency.

### The Memory Problem

In tensor parallelism, certain operations are replicated:

```
Before LayerNorm: activation shape (b, S, H) replicated on all TP ranks
After LayerNorm:  same shape, still replicated
Before Dropout:   replicated
After Dropout:    replicated
```

With TP degree $T$, this wastes $(T-1) \cdot b \cdot S \cdot H$ memory.

### The Solution: Distribute Sequence

Split the sequence dimension across TP ranks:

```
Rank 0: tokens 0 to S/T - 1
Rank 1: tokens S/T to 2S/T - 1
...
Rank T-1: tokens (T-1)S/T to S - 1
```

### Communication Pattern

**Before column-parallel layer** (needs full sequence for each rank):
$$\text{AllGather along sequence} \to (b, S, H)$$

**After row-parallel layer** (output is partial, sum across):
$$\text{ReduceScatter along sequence} \to (b, S/T, H)$$

```
                    ┌─────────────┐
                    │  LayerNorm  │  Sequence-parallel
                    │  (b, S/T, H)│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  AllGather  │
                    │  (b, S, H)  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Col-Parallel│  Tensor-parallel
                    │   Linear    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Row-Parallel│
                    │   Linear    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ReduceScatter│
                    │  (b, S/T, H)│
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Dropout    │  Sequence-parallel
                    │  (b, S/T, H)│
                    └─────────────┘
```

### Memory Savings

**Without sequence parallelism**:
$$M_{\text{LN+Dropout}} = 2 \cdot b \cdot S \cdot H \text{ (replicated)}$$

**With sequence parallelism**:
$$M_{\text{LN+Dropout}} = 2 \cdot b \cdot \frac{S}{T} \cdot H$$

Savings factor: $T$ (the tensor parallelism degree).

### Communication Analysis

Each transformer layer incurs:

- 1 AllGather before attention ($b \cdot S \cdot H$ bytes)
- 1 ReduceScatter after attention ($b \cdot S \cdot H$ bytes)
- 1 AllGather before FFN ($b \cdot S \cdot H$ bytes)
- 1 ReduceScatter after FFN ($b \cdot S \cdot H$ bytes)

**Total per layer**: $4 \cdot b \cdot S \cdot H \cdot \text{sizeof(dtype)}$

This is **the same volume** as tensor parallelism's AllReduce operations, just restructured.

### Implementation

```python
class SequenceParallelLayerNorm(nn.Module):
    """LayerNorm operating on sequence-parallel inputs."""

    def __init__(self, hidden_size: int, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_local, hidden)
        # LayerNorm is per-token, so no cross-rank communication needed
        return F.layer_norm(x, (x.size(-1),), self.weight, self.bias, self.eps)


def sequence_parallel_attention(q, k, v, tp_group):
    """Attention with sequence-parallel input/output."""
    # Input: (batch, seq_local, hidden)

    # AllGather to get full sequence
    q_full = all_gather_sequence(q, tp_group)  # (batch, seq, hidden)
    k_full = all_gather_sequence(k, tp_group)
    v_full = all_gather_sequence(v, tp_group)

    # Standard attention on full sequence
    output = attention(q_full, k_full, v_full)  # (batch, seq, hidden)

    # ReduceScatter back to local sequence chunk
    output_local = reduce_scatter_sequence(output, tp_group)

    return output_local  # (batch, seq_local, hidden)


def all_gather_sequence(x: torch.Tensor, group) -> torch.Tensor:
    """AllGather along sequence dimension."""
    world_size = dist.get_world_size(group)
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x, group=group)
    return torch.cat(gathered, dim=1)  # Concat along seq dimension


def reduce_scatter_sequence(x: torch.Tensor, group) -> torch.Tensor:
    """ReduceScatter along sequence dimension."""
    world_size = dist.get_world_size(group)
    seq_len = x.size(1)
    local_seq = seq_len // world_size

    # Split into chunks
    chunks = x.split(local_seq, dim=1)
    output = torch.empty_like(chunks[0])

    dist.reduce_scatter(output, list(chunks), op=dist.ReduceOp.SUM, group=group)
    return output
```

## The Decomposability of Attention

For true sequence parallelism—partitioning the attention computation itself—we need to understand how attention can be decomposed.

### Standard Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

The challenge: softmax normalizes across the **entire** key sequence:

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{S} e^{x_j}}$$

We can't compute the denominator without seeing all keys.

### Online Softmax

Milakov & Gimelshein (2018) showed softmax can be computed incrementally.

**The insight**: Track the running maximum and sum.

For numerical stability:
$$\text{softmax}(x)_i = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$$

where $m = \max_j x_j$.

**Incremental update**: Given chunks $A$ and $B$:

$$m_{AB} = \max(m_A, m_B)$$

$$\text{sum}_{AB} = \text{sum}_A \cdot e^{m_A - m_{AB}} + \text{sum}_B \cdot e^{m_B - m_{AB}}$$

**Output update**:
$$\text{out}_{AB} = \frac{\text{out}_A \cdot \text{sum}_A \cdot e^{m_A - m_{AB}} + \text{out}_B \cdot \text{sum}_B \cdot e^{m_B - m_{AB}}}{\text{sum}_{AB}}$$

### Associativity of Online Softmax

**Theorem**: Online softmax combination is associative.

Let $(m, s, o)$ represent the state (max, sum, output). The combination operation $\oplus$:

$$(m_1, s_1, o_1) \oplus (m_2, s_2, o_2) = (m_{12}, s_{12}, o_{12})$$

is associative:
$$((m_1, s_1, o_1) \oplus (m_2, s_2, o_2)) \oplus (m_3, s_3, o_3) = (m_1, s_1, o_1) \oplus ((m_2, s_2, o_2) \oplus (m_3, s_3, o_3))$$

**Proof sketch**: The max operation is associative. The sum and output updates are weighted averages with weights determined by exponentiated differences from the global max. The final result depends only on all inputs, not on combination order. $\square$

This associativity enables distributed computation.

## Ring Attention

Liu et al. (2023) introduced Ring Attention for extremely long sequences.

### The Core Idea

Each GPU holds:
- **Query chunk** $Q_i$: Local queries (never moves)
- **KV buffer**: Key-value pairs that rotate around the ring

```
Initial:
GPU 0: Q₀, K₀, V₀
GPU 1: Q₁, K₁, V₁
GPU 2: Q₂, K₂, V₂
GPU 3: Q₃, K₃, V₃

After step 1 (K, V rotate):
GPU 0: Q₀, K₃, V₃
GPU 1: Q₁, K₀, V₀
GPU 2: Q₂, K₁, V₁
GPU 3: Q₃, K₂, V₂
```

### The Algorithm

```python
def ring_attention(Q_local, K_local, V_local, ring_group):
    """
    Ring Attention: Compute attention over distributed sequence.

    Args:
        Q_local: Local query chunk (batch, seq_local, heads, dim)
        K_local: Local key chunk (batch, seq_local, heads, dim)
        V_local: Local value chunk (batch, seq_local, heads, dim)
        ring_group: Process group for ring communication

    Returns:
        Output: Attention output for local queries
    """
    world_size = dist.get_world_size(ring_group)
    rank = dist.get_rank(ring_group)

    # Initialize output accumulator with online softmax state
    batch, seq_local, heads, dim = Q_local.shape
    output = torch.zeros(batch, seq_local, heads, dim, device=Q_local.device)
    max_scores = torch.full((batch, seq_local, heads, 1), float('-inf'),
                            device=Q_local.device)
    sum_exp = torch.zeros(batch, seq_local, heads, 1, device=Q_local.device)

    # Current K, V buffers (will rotate)
    K_current = K_local.clone()
    V_current = V_local.clone()

    # Buffers for async communication
    K_recv = torch.empty_like(K_current)
    V_recv = torch.empty_like(V_current)

    for step in range(world_size):
        # Start async receive from previous rank
        if step < world_size - 1:
            src = (rank - 1) % world_size
            recv_k = dist.irecv(K_recv, src=src, group=ring_group)
            recv_v = dist.irecv(V_recv, src=src, group=ring_group)

        # Compute local attention scores
        # Q_local @ K_current^T -> (batch, seq_local, heads, seq_local)
        scores = torch.einsum('bqhd,bkhd->bqhk', Q_local, K_current)
        scores = scores / math.sqrt(dim)

        # Apply causal mask if needed
        kv_offset = ((rank - step) % world_size) * seq_local
        if kv_offset > 0:  # Keys are from "future" positions
            causal_mask = create_causal_mask(seq_local, kv_offset)
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Online softmax update
        chunk_max = scores.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(max_scores, chunk_max)

        # Rescale previous sum and output
        scale_old = torch.exp(max_scores - new_max)
        scale_new = torch.exp(chunk_max - new_max)

        exp_scores = torch.exp(scores - chunk_max)
        chunk_sum = exp_scores.sum(dim=-1, keepdim=True)

        # Update running sum
        sum_exp = sum_exp * scale_old + chunk_sum * scale_new

        # Update output
        chunk_output = torch.einsum('bqhk,bkhd->bqhd', exp_scores, V_current)
        output = output * scale_old + chunk_output * scale_new
        max_scores = new_max

        # Send K, V to next rank (async)
        if step < world_size - 1:
            dst = (rank + 1) % world_size
            send_k = dist.isend(K_current, dst=dst, group=ring_group)
            send_v = dist.isend(V_current, dst=dst, group=ring_group)

            # Wait for receive and swap buffers
            recv_k.wait()
            recv_v.wait()
            K_current, K_recv = K_recv, K_current
            V_current, V_recv = V_recv, V_current

            # Wait for send to complete
            send_k.wait()
            send_v.wait()

    # Final normalization
    output = output / sum_exp

    return output
```

### Communication Pattern

Each step:
- Send $K_i, V_i$ to next rank: $2 \cdot S/P \cdot H$ elements
- Receive from previous rank: same

**Total communication per attention layer**:
$$C = 2(P-1) \cdot \frac{S}{P} \cdot H \cdot \text{sizeof} = 2 \cdot \frac{P-1}{P} \cdot S \cdot H \cdot \text{sizeof}$$

**Critical feature**: Communication overlaps with computation.

### Compute-Communication Overlap

While computing attention with current K, V:
- Simultaneously send current K, V to next rank
- Simultaneously receive next K, V from previous rank

**Overlap efficiency**:

$$\text{Overlap} = \min\left(1, \frac{T_{\text{compute}}}{T_{\text{comm}}}\right)$$

Where:
$$T_{\text{compute}} = \frac{2 \cdot (S/P)^2 \cdot H}{F_{\text{FLOPS}}}$$
$$T_{\text{comm}} = \frac{2 \cdot (S/P) \cdot H \cdot \text{sizeof}}{\text{bandwidth}}$$

For large sequences, compute dominates and overlap is nearly perfect.

### Memory Analysis

**Peak memory per GPU**:

- Query: $b \cdot \frac{S}{P} \cdot H$
- Two KV buffers (double buffering): $2 \cdot 2 \cdot b \cdot \frac{S}{P} \cdot H$
- Attention scores (one chunk): $b \cdot n_h \cdot \frac{S}{P} \cdot \frac{S}{P}$
- Output accumulator: $b \cdot \frac{S}{P} \cdot H$

**Total**:
$$M \approx 5 \cdot b \cdot \frac{S}{P} \cdot H + b \cdot n_h \cdot \left(\frac{S}{P}\right)^2$$

The quadratic term is now $(S/P)^2$ instead of $S^2$ — a factor of $P^2$ reduction.

## Ulysses: AlltoAll Sequence Parallelism

Fang et al. (2024) proposed Ulysses as an alternative to Ring Attention.

### The Approach

Instead of rotating K, V through a ring, use AlltoAll to redistribute:

1. **Initial state**: Each GPU has local Q, K, V for sequence chunk
2. **AlltoAll on K**: Redistribute so each GPU has all K for some heads
3. **AlltoAll on V**: Same redistribution
4. **Local attention**: Compute attention with full sequence for subset of heads
5. **AlltoAll on output**: Redistribute back to sequence-parallel layout

```
Before AlltoAll (sequence-parallel):
GPU 0: Q[0:S/P], K[0:S/P], V[0:S/P] for all heads
GPU 1: Q[S/P:2S/P], K[S/P:2S/P], V[S/P:2S/P] for all heads

After AlltoAll (head-parallel):
GPU 0: Q[0:S], K[0:S], V[0:S] for heads 0:H/P
GPU 1: Q[0:S], K[0:S], V[0:S] for heads H/P:2H/P
```

### Implementation

```python
def ulysses_attention(Q, K, V, sp_group):
    """
    Ulysses sequence parallelism using AlltoAll.

    Args:
        Q, K, V: Local chunks (batch, seq_local, heads, dim)
        sp_group: Sequence parallel process group

    Returns:
        Output attention for local sequence chunk
    """
    world_size = dist.get_world_size(sp_group)
    batch, seq_local, heads, dim = Q.shape

    # Reshape for AlltoAll: split heads dimension
    # (batch, seq_local, heads, dim) -> (batch, seq_local, P, heads/P, dim)
    heads_per_rank = heads // world_size
    Q = Q.view(batch, seq_local, world_size, heads_per_rank, dim)
    K = K.view(batch, seq_local, world_size, heads_per_rank, dim)
    V = V.view(batch, seq_local, world_size, heads_per_rank, dim)

    # AlltoAll: exchange sequence chunks for head chunks
    # After: each rank has full sequence for subset of heads
    Q = all_to_all(Q, dim_scatter=2, dim_gather=1, group=sp_group)
    K = all_to_all(K, dim_scatter=2, dim_gather=1, group=sp_group)
    V = all_to_all(V, dim_scatter=2, dim_gather=1, group=sp_group)

    # Now shape is (batch, seq_full, heads_local, dim)
    seq_full = seq_local * world_size
    Q = Q.view(batch, seq_full, heads_per_rank, dim)
    K = K.view(batch, seq_full, heads_per_rank, dim)
    V = V.view(batch, seq_full, heads_per_rank, dim)

    # Standard attention on full sequence (for local heads)
    output = flash_attention(Q, K, V)  # (batch, seq_full, heads_local, dim)

    # Reshape for reverse AlltoAll
    output = output.view(batch, world_size, seq_local, heads_per_rank, dim)

    # AlltoAll: exchange head chunks for sequence chunks
    output = all_to_all(output, dim_scatter=1, dim_gather=2, group=sp_group)

    # Reshape back
    output = output.view(batch, seq_local, heads, dim)

    return output


def all_to_all(x, dim_scatter, dim_gather, group):
    """AlltoAll with specified scatter and gather dimensions."""
    world_size = dist.get_world_size(group)

    # Split along scatter dimension
    splits = x.chunk(world_size, dim=dim_scatter)
    splits = [s.contiguous() for s in splits]

    # AlltoAll
    output_splits = [torch.empty_like(splits[0]) for _ in range(world_size)]
    dist.all_to_all(output_splits, splits, group=group)

    # Concatenate along gather dimension
    return torch.cat(output_splits, dim=dim_gather)
```

### Communication Analysis

**AlltoAll volume** (each direction):
$$C = (P-1) \cdot \frac{S}{P} \cdot \frac{H}{P} \cdot b = \frac{(P-1) \cdot S \cdot H \cdot b}{P^2}$$

Per attention layer: 4 AlltoAll operations (Q, K, V in; output out).

**Total**:
$$C_{\text{total}} = 4 \cdot \frac{(P-1) \cdot S \cdot H \cdot b}{P^2}$$

### Ring vs Ulysses Comparison

| Aspect | Ring Attention | Ulysses |
|--------|---------------|---------|
| Communication | P2P in ring | AlltoAll |
| Volume | $2 \cdot \frac{P-1}{P} \cdot S \cdot H$ | $4 \cdot \frac{P-1}{P^2} \cdot S \cdot H$ |
| Overlap | Yes (compute + comm) | Limited |
| Memory | KV buffers | No extra buffers |
| Best for | Long seq, P2P fast | Few ranks, AlltoAll fast |

**When to use Ring**: Many sequence parallel ranks, can overlap.

**When to use Ulysses**: Few ranks (2-8), high AlltoAll bandwidth (NVLink).

## Flash Attention Integration

Both Ring and Ulysses benefit from Flash Attention.

### Flash Attention Recap

Dao et al. (2022) compute attention in tiles without materializing the full $S \times S$ matrix:

```python
def flash_attention_forward(Q, K, V, block_size=64):
    """
    Flash Attention: memory-efficient attention using tiling.
    """
    batch, seq_q, heads, dim = Q.shape
    seq_k = K.shape[1]

    output = torch.zeros_like(Q)
    max_scores = torch.full((batch, seq_q, heads, 1), float('-inf'))
    sum_exp = torch.zeros((batch, seq_q, heads, 1))

    # Tile over K, V
    for k_start in range(0, seq_k, block_size):
        k_end = min(k_start + block_size, seq_k)
        K_block = K[:, k_start:k_end]
        V_block = V[:, k_start:k_end]

        # Tile over Q
        for q_start in range(0, seq_q, block_size):
            q_end = min(q_start + block_size, seq_q)
            Q_block = Q[:, q_start:q_end]

            # Compute attention scores for this tile
            scores = torch.einsum('bqhd,bkhd->bqhk', Q_block, K_block)
            scores = scores / math.sqrt(dim)

            # Online softmax update (same as Ring Attention)
            block_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(max_scores[:, q_start:q_end], block_max)

            scale_old = torch.exp(max_scores[:, q_start:q_end] - new_max)
            scale_new = torch.exp(block_max - new_max)

            exp_scores = torch.exp(scores - block_max)
            block_sum = exp_scores.sum(dim=-1, keepdim=True)

            # Update accumulators
            sum_exp[:, q_start:q_end] = (
                sum_exp[:, q_start:q_end] * scale_old +
                block_sum * scale_new
            )

            block_out = torch.einsum('bqhk,bkhd->bqhd', exp_scores, V_block)
            output[:, q_start:q_end] = (
                output[:, q_start:q_end] * scale_old +
                block_out * scale_new
            )

            max_scores[:, q_start:q_end] = new_max

    # Final normalization
    output = output / sum_exp
    return output
```

### Ring Attention with Flash Attention

The inner loop of Ring Attention can use Flash Attention for the local computation:

```python
def ring_flash_attention(Q_local, K_local, V_local, ring_group):
    """Ring Attention using Flash Attention for each step."""
    world_size = dist.get_world_size(ring_group)

    # Initialize accumulators
    output, max_scores, sum_exp = init_accumulators(Q_local)

    K_current, V_current = K_local.clone(), V_local.clone()

    for step in range(world_size):
        # Start async communication (overlapped)
        comm_handle = start_ring_comm(K_current, V_current, ring_group)

        # Use Flash Attention kernel for this chunk
        # Returns (output_chunk, max_chunk, sum_chunk)
        chunk_out, chunk_max, chunk_sum = flash_attention_with_state(
            Q_local, K_current, V_current
        )

        # Online softmax merge
        output, max_scores, sum_exp = merge_attention_state(
            output, max_scores, sum_exp,
            chunk_out, chunk_max, chunk_sum
        )

        # Complete communication
        K_current, V_current = complete_ring_comm(comm_handle)

    return output / sum_exp
```

## Hybrid Context Parallelism

For very long sequences, combine techniques.

### Hierarchical Ring

Use multiple rings at different levels:

```
Inter-node ring (slow network):
  [Node 0] ←→ [Node 1] ←→ [Node 2] ←→ [Node 3]

Intra-node ring (NVLink):
  GPU0 ←→ GPU1 ←→ GPU2 ←→ GPU3 (within each node)
```

### Combined Strategies

```python
class HybridContextParallel:
    """Combine Ulysses (intra-node) with Ring (inter-node)."""

    def __init__(self, local_group, global_ring_group):
        self.local_group = local_group  # GPUs within node
        self.ring_group = global_ring_group  # Across nodes

    def forward(self, Q, K, V):
        # Step 1: Ulysses within node (fast AlltoAll via NVLink)
        Q, K, V = ulysses_qkv_exchange(Q, K, V, self.local_group)

        # Step 2: Ring across nodes (overlapped P2P)
        output = ring_attention(Q, K, V, self.ring_group)

        # Step 3: Ulysses to restore layout
        output = ulysses_output_exchange(output, self.local_group)

        return output
```

## Memory Comparison

For sequence length $S$, hidden size $H$, heads $n_h$, parallel degree $P$:

| Method | Attention Matrix Memory | KV Memory |
|--------|------------------------|-----------|
| Standard | $O(n_h \cdot S^2)$ | $O(S \cdot H)$ |
| Flash Attention | $O(n_h \cdot B^2)$ (block size $B$) | $O(S \cdot H)$ |
| Ring Attention | $O(n_h \cdot (S/P)^2)$ | $O(S \cdot H / P)$ |
| Ulysses | $O((n_h/P) \cdot S^2)$ | $O(S \cdot H)$ |

**Ring Attention** reduces both dimensions by $P$; best for very long sequences.

**Ulysses** reduces head dimension by $P$; keeps full sequence per GPU.

## Causal Masking Considerations

Causal attention masks out future tokens: $\text{Mask}[i, j] = 1$ if $j \leq i$.

### Ring Attention Causality

When rotating K, V, we must track which positions they represent:

```python
def get_causal_mask(q_offset, k_offset, seq_local):
    """Create causal mask for Ring Attention step."""
    q_positions = torch.arange(q_offset, q_offset + seq_local)
    k_positions = torch.arange(k_offset, k_offset + seq_local)

    # Mask: True where k_pos > q_pos (future positions)
    mask = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
    return mask
```

### Optimization: Skip Future Chunks

If all keys in a chunk are "future" relative to all queries, skip computation entirely:

```python
def should_compute_chunk(q_start, q_end, k_start, k_end):
    """Check if KV chunk has any valid (non-future) positions."""
    # All keys are future if k_start > q_end
    return k_start <= q_end
```

This can reduce computation by up to 50% for causal attention.

## Implementation: Complete Sequence Parallel Layer

```python
class SequenceParallelTransformerLayer(nn.Module):
    """Complete transformer layer with sequence parallelism."""

    def __init__(self, config, sp_group, tp_group):
        super().__init__()
        self.sp_group = sp_group
        self.tp_group = tp_group
        self.sp_size = dist.get_world_size(sp_group)

        # Layer components
        self.ln1 = SequenceParallelLayerNorm(config.hidden_size, sp_group)
        self.attention = SequenceParallelAttention(config, sp_group, tp_group)
        self.ln2 = SequenceParallelLayerNorm(config.hidden_size, sp_group)
        self.ffn = SequenceParallelFFN(config, sp_group, tp_group)

    def forward(self, x):
        # x: (batch, seq_local, hidden) - sequence-parallel input

        # Pre-norm attention
        residual = x
        x = self.ln1(x)
        x = self.attention(x)  # Handles AllGather/ReduceScatter internally
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)  # Handles AllGather/ReduceScatter internally
        x = residual + x

        return x


class SequenceParallelAttention(nn.Module):
    """Attention with context parallelism (Ring or Ulysses)."""

    def __init__(self, config, sp_group, tp_group, method='ring'):
        super().__init__()
        self.sp_group = sp_group
        self.tp_group = tp_group
        self.method = method
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        # Projections (tensor-parallel)
        self.qkv_proj = ColumnParallelLinear(
            config.hidden_size,
            3 * config.hidden_size,
            tp_group
        )
        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            tp_group
        )

    def forward(self, x):
        batch, seq_local, hidden = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for attention
        q = q.view(batch, seq_local, self.num_heads, self.head_dim)
        k = k.view(batch, seq_local, self.num_heads, self.head_dim)
        v = v.view(batch, seq_local, self.num_heads, self.head_dim)

        # Context-parallel attention
        if self.method == 'ring':
            output = ring_attention(q, k, v, self.sp_group)
        else:  # ulysses
            output = ulysses_attention(q, k, v, self.sp_group)

        # Reshape and project
        output = output.view(batch, seq_local, hidden)
        output = self.out_proj(output)

        return output
```

## Exercises

1. **Memory calculation**: For a model with $H = 4096$, $n_h = 32$, $S = 256K$, batch 1, bf16:
   - Calculate attention matrix memory without sequence parallelism
   - Calculate with $P = 8$ Ring Attention
   - What's the reduction factor?

2. **Communication volume**: Compare Ring Attention and Ulysses communication volume for $S = 1M$, $H = 8192$, $P = 16$. Which uses less bandwidth?

3. **Overlap efficiency**: If attention compute takes 10ms and KV transfer takes 2ms per Ring step, what fraction of communication is hidden? With $P = 8$ steps, what's the effective communication overhead?

4. **Causal optimization**: For Ring Attention with $P = 8$ and causal masking, how many of the 8 steps can be skipped (on average) due to all-future chunks?

5. **Hybrid design**: Design a sequence parallelism strategy for 64 GPUs (8 nodes × 8 GPUs) with $S = 2M$ tokens. Propose group configurations and estimate memory per GPU.

6. **Online softmax verification**: Implement and verify that combining three chunks $(A, B, C)$ as $(A \oplus B) \oplus C$ gives the same result as $A \oplus (B \oplus C)$ for the attention operation.

7. **Flash + Ring**: Modify the Ring Attention algorithm to use Flash Attention internally. What are the memory implications of the nested tiling?

## Key Takeaways

1. **Two types of sequence parallelism**:
   - Megatron-style: Reduces LayerNorm/Dropout activation memory
   - Context parallelism: Distributes attention computation itself

2. **Online softmax enables decomposition**: Attention can be computed incrementally with associative state updates.

3. **Ring Attention**: Rotate K, V in a ring; communication overlaps with compute.

4. **Ulysses**: AlltoAll to redistribute sequence vs heads; simpler but less overlap.

5. **Memory scales as $O((S/P)^2)$**: Ring Attention reduces attention matrix by factor $P^2$.

6. **Flash Attention integration**: Use Flash as the inner kernel for memory efficiency.

7. **Causal masking optimization**: Skip chunks where all keys are in the future.

8. **Choose based on topology**: Ring for many ranks with good P2P; Ulysses for few ranks with fast AlltoAll.
