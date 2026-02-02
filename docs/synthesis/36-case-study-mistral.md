---
title: "Case Study: Mistral and Mixtral"
subtitle: "Efficient Attention and Sparse Mixture-of-Experts"
---

<div class="chapter-opener" markdown>
Mistral AI demonstrated that architectural efficiency can rival scale. Through sliding window attention, grouped-query attention, and sparse MoE, they achieved competitive performance with dramatically fewer resources.
</div>

<div class="investigation-question" markdown>
**The Question**: Mistral 7B matches LLaMA 2 13B performance with half the parameters. Mixtral 8x7B matches LLaMA 2 70B while activating only 13B parameters. What architectural innovations enable this efficiency, and what are the distributed training implications?
</div>

!!! tip "New Concepts Introduced"
    This case study introduces architectural innovations for efficiency:

    - **Sliding Window Attention (SWA)**: Limits attention to a local window of tokens, reducing memory from O(n²) to O(n·w) where w is window size
    - **Grouped-Query Attention (GQA)**: Shares key-value heads across multiple query heads, reducing KV cache by 4-8× with minimal quality loss
    - **Sparse MoE at scale**: Mixtral's 8×7B and 8×22B demonstrate production sparse mixture-of-experts

    These techniques combine multiplicatively—SWA + GQA together can reduce KV cache by 32×.

## The Mistral Architecture Philosophy

Mistral AI's approach inverts the typical scaling narrative. Instead of asking "how do we train larger models?", they ask "how do we maximize quality per FLOP?" This leads to different architectural choices:

| Model | Total Params | Active Params | Performance Target |
|-------|-------------|---------------|-------------------|
| Mistral 7B | 7.3B | 7.3B | LLaMA 2 13B |
| Mixtral 8x7B | 46.7B | 12.9B | LLaMA 2 70B |
| Mixtral 8x22B | 141B | 39B | GPT-4 class |

The key insight: **sparse computation enables dense performance**.

## Sliding Window Attention

Standard attention has $O(n^2)$ complexity. For long contexts, this dominates compute and memory.

### The Locality Hypothesis

For many tasks, tokens primarily attend to nearby context. Sliding Window Attention (SWA) exploits this:

$$\text{Attention}(Q, K, V)_{i} = \text{softmax}\left(\frac{Q_i K_{[i-w:i]}^T}{\sqrt{d_k}}\right) V_{[i-w:i]}$$

where $w$ is the window size.

### Memory Analysis

**Standard Attention** (sequence length $n$, batch $B$, heads $h$, dimension $d$):

$$\text{KV cache} = 2 \cdot B \cdot n \cdot h \cdot d \cdot 2 \text{ bytes}$$

For $n = 32768$, $B = 1$, $h = 32$, $d = 128$:

$$= 2 \cdot 1 \cdot 32768 \cdot 32 \cdot 128 \cdot 2 = 512 \text{ MB}$$

**Sliding Window Attention** (window size $w$):

$$\text{KV cache} = 2 \cdot B \cdot w \cdot h \cdot d \cdot 2 \text{ bytes}$$

For $w = 4096$:

$$= 2 \cdot 1 \cdot 4096 \cdot 32 \cdot 128 \cdot 2 = 64 \text{ MB}$$

**8× memory reduction** for this configuration.

### The Rolling Buffer

Implementation uses a circular buffer:

```python
class SlidingWindowKVCache:
    """
    Rolling KV cache for sliding window attention.

    Memory is fixed at window_size regardless of sequence length.
    """
    def __init__(self, batch_size: int, window_size: int,
                 n_heads: int, head_dim: int, dtype=torch.bfloat16):
        self.window_size = window_size
        # Circular buffer: position i stored at index i % window_size
        self.k_cache = torch.zeros(
            batch_size, n_heads, window_size, head_dim, dtype=dtype
        )
        self.v_cache = torch.zeros(
            batch_size, n_heads, window_size, head_dim, dtype=dtype
        )
        self.position = 0

    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new KV and return valid window."""
        seq_len = k.size(2)

        for i in range(seq_len):
            idx = (self.position + i) % self.window_size
            self.k_cache[:, :, idx, :] = k[:, :, i, :]
            self.v_cache[:, :, idx, :] = v[:, :, i, :]

        self.position += seq_len

        # Return properly ordered window
        return self._get_ordered_cache()

    def _get_ordered_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cache in correct temporal order."""
        if self.position <= self.window_size:
            return self.k_cache[:, :, :self.position], self.v_cache[:, :, :self.position]

        # Reorder circular buffer
        start = self.position % self.window_size
        indices = [(start + i) % self.window_size for i in range(self.window_size)]
        return self.k_cache[:, :, indices], self.v_cache[:, :, indices]
```

### Effective Context Through Stacking

With $L$ layers and window size $w$, the effective receptive field grows (upper bound):

$$\text{Effective context} \leq L \cdot w$$

For Mistral 7B with $L = 32$ layers and $w = 4096$:

$$\text{Effective context} \leq 32 \times 4096 = 131072 \text{ tokens}$$

Information propagates through the network even though each layer only sees $w$ tokens.

```
Layer 32: [============================================]
                         ↑
Layer 31: [========================================]
                     ↑
...
Layer 2:  [========]
              ↑
Layer 1:  [====]
            ↑
Input:    [Token at position 0 influences position 131K through layer propagation]
```

### Training with Sliding Windows

During training, we must mask attention appropriately:

```python
def sliding_window_mask(seq_len: int, window_size: int) -> torch.Tensor:
    """
    Create attention mask for sliding window.

    Returns mask where mask[i,j] = True if j is in window for query i.
    """
    # Causal: can only attend to previous positions
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    # Window: can only attend to positions within window
    positions = torch.arange(seq_len)
    window = (positions.unsqueeze(1) - positions.unsqueeze(0)) < window_size

    return causal & window

def sliding_window_attention_forward(
    q: torch.Tensor,  # [batch, heads, seq, dim]
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int
) -> torch.Tensor:
    """Efficient sliding window attention with FlashAttention."""
    batch, heads, seq_len, dim = q.shape

    # For training: use sparse attention pattern
    # FlashAttention 2 supports this natively
    if seq_len <= window_size:
        # Standard attention for short sequences
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Block-sparse attention for long sequences
    # Implementation depends on backend (FlashAttention, xFormers, etc.)
    mask = sliding_window_mask(seq_len, window_size)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim)
    scores = scores.masked_fill(~mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)

    return torch.matmul(attn, v)
```

## Grouped-Query Attention

Grouped-Query Attention (GQA) reduces KV cache by sharing key-value heads across query heads.

### The GQA Design Space

| Variant | Query Heads | KV Heads | Memory Ratio |
|---------|-------------|----------|--------------|
| Multi-Head (MHA) | $h$ | $h$ | 1.0 |
| Multi-Query (MQA) | $h$ | 1 | $1/h$ |
| Grouped-Query (GQA) | $h$ | $g$ | $g/h$ |

Mistral 7B uses $h = 32$ query heads with $g = 8$ KV groups:

$$\text{Memory ratio} = \frac{8}{32} = 0.25$$

**4× KV cache reduction** from GQA alone.

### Combined Savings

SWA + GQA together:

$$\text{Total reduction} = \frac{w}{n} \cdot \frac{g}{h}$$

For $n = 32768$, $w = 4096$, $h = 32$, $g = 8$:

$$= \frac{4096}{32768} \cdot \frac{8}{32} = \frac{1}{8} \cdot \frac{1}{4} = \frac{1}{32}$$

**32× total memory reduction** for KV cache.

### Implementation

```python
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention with Sliding Window support.

    Combines GQA for memory efficiency with SWA for long contexts.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        window_size: int,
        head_dim: Optional[int] = None
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or d_model // n_heads
        self.window_size = window_size

        # Query heads: full count
        self.W_q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)

        # KV heads: reduced count
        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)

        self.W_o = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # Repeat factor for broadcasting KV heads to query heads
        self.n_rep = n_heads // n_kv_heads

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: Optional[SlidingWindowKVCache] = None
    ) -> Tuple[torch.Tensor, Optional[SlidingWindowKVCache]]:
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.W_q(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.W_k(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.W_v(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Transpose to [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        q, k = apply_rotary_embedding(q, k, positions)

        # Handle KV cache for inference
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Expand KV heads to match query heads
        # [batch, n_kv_heads, seq, dim] -> [batch, n_heads, seq, dim]
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Compute attention with sliding window
        output = sliding_window_attention_forward(q, k, v, self.window_size)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.W_o(output), kv_cache

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query heads."""
        batch, n_kv_heads, seq_len, head_dim = x.shape

        # [batch, n_kv, seq, dim] -> [batch, n_kv, n_rep, seq, dim]
        x = x.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1)

        # [batch, n_kv * n_rep, seq, dim] = [batch, n_heads, seq, dim]
        return x.reshape(batch, self.n_heads, seq_len, head_dim)
```

## Mixtral: Sparse Mixture-of-Experts

Mixtral extends Mistral with sparse MoE, activating only 2 of 8 experts per token.

### Architecture Comparison

| Component | Mistral 7B | Mixtral 8x7B |
|-----------|-----------|--------------|
| Layers | 32 | 32 |
| Hidden dim | 4096 | 4096 |
| Heads | 32 | 32 |
| KV heads | 8 | 8 |
| FFN | Dense 14336 | 8 experts × 14336 |
| Active FFN | 14336 | 2 × 14336 |
| Total params | 7.3B | 46.7B |
| Active params | 7.3B | 12.9B |

### The Routing Mechanism

Mixtral uses a learned router with top-2 selection:

```python
class MixtralRouter(nn.Module):
    """
    Expert router for Mixtral-style sparse MoE.

    Selects top-k experts per token with learned routing weights.
    """
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int = 2,
        routing_noise: float = 0.0
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.routing_noise = routing_noise

        # Linear router: d_model -> n_experts
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Returns:
            expert_weights: [batch, seq, top_k] - normalized routing weights
            expert_indices: [batch, seq, top_k] - selected expert indices
            router_logits: [batch, seq, n_experts] - raw router outputs
        """
        batch, seq_len, _ = x.shape

        # Compute router logits
        router_logits = self.gate(x)  # [batch, seq, n_experts]

        # Add noise during training for exploration
        if self.training and self.routing_noise > 0:
            noise = torch.randn_like(router_logits) * self.routing_noise
            router_logits = router_logits + noise

        # Select top-k experts
        routing_weights = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )

        # Renormalize weights to sum to 1
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        return expert_weights, expert_indices, router_logits
```

### Load Balancing

Without balancing, routing can collapse to always selecting the same experts. Mixtral uses auxiliary loss:

```python
def load_balancing_loss(
    router_logits: torch.Tensor,
    expert_indices: torch.Tensor,
    n_experts: int,
    top_k: int
) -> torch.Tensor:
    """
    Auxiliary loss to encourage balanced expert usage.

    Combines:
    1. Load balancing: all experts should receive similar token counts
    2. Router z-loss: prevent router logits from growing too large
    """
    batch, seq_len, _ = router_logits.shape
    n_tokens = batch * seq_len

    # Compute routing probabilities
    routing_probs = F.softmax(router_logits, dim=-1)  # [batch, seq, n_experts]
    routing_probs = routing_probs.view(-1, n_experts)  # [tokens, n_experts]

    # Expert selection frequency (fraction of tokens routed to each expert)
    expert_mask = F.one_hot(expert_indices, n_experts).float()  # [batch, seq, top_k, n_experts]
    expert_mask = expert_mask.sum(dim=2)  # [batch, seq, n_experts]
    expert_mask = expert_mask.view(-1, n_experts)  # [tokens, n_experts]

    tokens_per_expert = expert_mask.sum(dim=0)  # [n_experts]
    fraction_per_expert = tokens_per_expert / (n_tokens * top_k)

    # Mean routing probability to each expert
    mean_routing_prob = routing_probs.mean(dim=0)  # [n_experts]

    # Load balancing loss: minimize product of fraction and probability
    # Ideally both should be uniform (1/n_experts)
    load_balance_loss = n_experts * (fraction_per_expert * mean_routing_prob).sum()

    # Router z-loss: regularize router logits
    z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()

    return load_balance_loss + 0.001 * z_loss
```

### MoE Block Implementation

```python
class MixtralMoEBlock(nn.Module):
    """
    Mixtral Mixture-of-Experts block.

    Replaces the dense FFN with sparse expert routing.
    """
    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        n_experts: int = 8,
        top_k: int = 2
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        # Router
        self.router = MixtralRouter(d_model, n_experts, top_k)

        # Expert FFNs (each is a standard SwiGLU FFN)
        self.experts = nn.ModuleList([
            SwiGLUFFN(d_model, ffn_dim)
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with expert routing.

        Returns:
            output: [batch, seq, d_model]
            aux_loss: scalar load balancing loss
        """
        batch, seq_len, d_model = x.shape

        # Route tokens to experts
        expert_weights, expert_indices, router_logits = self.router(x)

        # Compute auxiliary loss
        aux_loss = load_balancing_loss(
            router_logits, expert_indices, self.n_experts, self.top_k
        )

        # Process through experts
        # Naive implementation: loop over experts
        output = torch.zeros_like(x)

        for expert_idx in range(self.n_experts):
            # Find tokens routed to this expert
            for k in range(self.top_k):
                mask = (expert_indices[:, :, k] == expert_idx)
                if not mask.any():
                    continue

                # Get tokens for this expert
                expert_input = x[mask]  # [n_tokens, d_model]

                # Process through expert
                expert_output = self.experts[expert_idx](expert_input)

                # Weight by routing weight and accumulate
                weights = expert_weights[:, :, k][mask].unsqueeze(-1)
                output[mask] += weights * expert_output

        return output, aux_loss

class SwiGLUFFN(nn.Module):
    """SwiGLU FFN as used in Mistral/Mixtral."""
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, ffn_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

## Efficient MoE with Megablocks

The naive loop-based implementation is inefficient. Megablocks provides efficient batched expert execution:

### Token-to-Expert Permutation

```python
class EfficientMoE(nn.Module):
    """
    Efficient MoE using token permutation and batched matrix multiply.

    Key insight: Group tokens by expert, process in batches, then unpermute.
    """
    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        n_experts: int,
        top_k: int
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = MixtralRouter(d_model, n_experts, top_k)

        # Experts as fused parameter tensors for efficient batching
        # w1 and w3 combined for gating
        self.w13 = nn.Parameter(torch.randn(n_experts, d_model, 2 * ffn_dim))
        self.w2 = nn.Parameter(torch.randn(n_experts, ffn_dim, d_model))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, d_model = x.shape

        # Route tokens
        expert_weights, expert_indices, router_logits = self.router(x)

        # Flatten batch and sequence dimensions
        x_flat = x.view(-1, d_model)  # [batch*seq, d_model]
        expert_weights_flat = expert_weights.view(-1, self.top_k)
        expert_indices_flat = expert_indices.view(-1, self.top_k)

        # Build permutation indices
        # Group tokens by their assigned experts
        sorted_token_ids, expert_counts = self._compute_permutation(expert_indices_flat)

        # Permute tokens by expert assignment
        x_permuted = x_flat[sorted_token_ids]  # Tokens grouped by expert

        # Compute expert outputs using batched matrix multiply
        expert_outputs = self._batched_expert_forward(x_permuted, expert_counts)

        # Unpermute back to original order
        output_flat = torch.zeros_like(x_flat)
        output_flat = self._weighted_accumulate(
            expert_outputs, sorted_token_ids, expert_weights_flat, expert_indices_flat
        )

        output = output_flat.view(batch, seq_len, d_model)
        aux_loss = load_balancing_loss(
            router_logits, expert_indices, self.n_experts, self.top_k
        )

        return output, aux_loss

    def _compute_permutation(
        self, expert_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute permutation to group tokens by expert."""
        n_tokens = expert_indices.size(0)

        # Flatten expert selections
        flat_indices = expert_indices.view(-1)  # [n_tokens * top_k]
        token_ids = torch.arange(n_tokens, device=expert_indices.device).repeat_interleave(self.top_k)

        # Sort by expert index
        _, perm = flat_indices.sort(stable=True)
        sorted_token_ids = token_ids[perm]

        # Count tokens per expert
        expert_counts = torch.bincount(flat_indices, minlength=self.n_experts)

        return sorted_token_ids, expert_counts

    def _batched_expert_forward(
        self, x: torch.Tensor, expert_counts: torch.Tensor
    ) -> torch.Tensor:
        """Process all tokens through their assigned experts efficiently."""
        outputs = []
        offset = 0

        for expert_idx, count in enumerate(expert_counts.tolist()):
            if count == 0:
                continue

            # Get tokens for this expert
            expert_tokens = x[offset:offset + count]  # [count, d_model]

            # SwiGLU forward: w2(silu(w1(x)) * w3(x))
            w13 = self.w13[expert_idx]  # [d_model, 2*ffn_dim]
            w2 = self.w2[expert_idx]    # [ffn_dim, d_model]

            # Combined w1/w3 projection
            gate_up = expert_tokens @ w13  # [count, 2*ffn_dim]
            gate, up = gate_up.chunk(2, dim=-1)

            # SwiGLU activation
            hidden = F.silu(gate) * up  # [count, ffn_dim]

            # Output projection
            output = hidden @ w2  # [count, d_model]
            outputs.append(output)

            offset += count

        return torch.cat(outputs, dim=0) if outputs else torch.empty(0, x.size(-1))
```

### Distributed MoE: Expert Parallelism

For Mixtral-scale models, experts can be distributed across GPUs:

```python
class DistributedMoE(nn.Module):
    """
    MoE with expert parallelism.

    Each GPU holds a subset of experts. Tokens are routed via all-to-all.
    """
    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        n_experts: int,
        top_k: int,
        expert_parallel_group: torch.distributed.ProcessGroup
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.ep_group = expert_parallel_group
        self.ep_size = dist.get_world_size(expert_parallel_group)
        self.ep_rank = dist.get_rank(expert_parallel_group)

        # Each GPU holds n_experts // ep_size experts
        self.local_n_experts = n_experts // self.ep_size
        self.expert_start = self.ep_rank * self.local_n_experts

        # Router (replicated on all GPUs)
        self.router = MixtralRouter(d_model, n_experts, top_k)

        # Local experts only
        self.local_experts = nn.ModuleList([
            SwiGLUFFN(d_model, ffn_dim)
            for _ in range(self.local_n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, d_model = x.shape

        # Route tokens (same routing on all GPUs)
        expert_weights, expert_indices, router_logits = self.router(x)

        # Determine which tokens go to which GPU
        # Each GPU processes experts [expert_start, expert_start + local_n_experts)
        tokens_to_send, send_counts = self._prepare_all_to_all(
            x, expert_indices, expert_weights
        )

        # All-to-all: exchange tokens between GPUs
        tokens_received, recv_counts = self._all_to_all_tokens(
            tokens_to_send, send_counts
        )

        # Process tokens through local experts
        local_outputs = self._process_local_experts(tokens_received, recv_counts)

        # All-to-all: send results back
        outputs = self._all_to_all_results(local_outputs, recv_counts, send_counts)

        # Reconstruct output tensor
        output = self._reconstruct_output(outputs, expert_weights, expert_indices)

        aux_loss = load_balancing_loss(
            router_logits, expert_indices, self.n_experts, self.top_k
        )

        return output, aux_loss

    def _all_to_all_tokens(
        self, tokens: torch.Tensor, send_counts: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        """Exchange tokens between GPUs via all-to-all."""
        # Gather receive counts from all GPUs
        all_counts = [torch.tensor(send_counts, device=tokens.device)
                      for _ in range(self.ep_size)]
        dist.all_gather(all_counts, torch.tensor(send_counts, device=tokens.device),
                       group=self.ep_group)

        recv_counts = [c[self.ep_rank].item() for c in all_counts]

        # Prepare send/recv buffers
        send_splits = send_counts
        recv_splits = recv_counts

        output = torch.empty(sum(recv_counts), tokens.size(-1),
                            device=tokens.device, dtype=tokens.dtype)

        # All-to-all
        dist.all_to_all_single(output, tokens,
                               output_split_sizes=recv_splits,
                               input_split_sizes=send_splits,
                               group=self.ep_group)

        return output, recv_counts
```

## Distributed Training Analysis

### Memory per GPU

For Mixtral 8x7B training with expert parallelism:

```python
class MixtralMemoryAnalyzer:
    """
    Memory analysis for Mixtral distributed training.
    """
    def __init__(self):
        # Architecture
        self.layers = 32
        self.d_model = 4096
        self.n_heads = 32
        self.n_kv_heads = 8
        self.ffn_dim = 14336
        self.n_experts = 8
        self.vocab_size = 32000

        # Parallelism config
        self.tp = 1  # Tensor parallel
        self.ep = 8  # Expert parallel
        self.dp = 8  # Data parallel (with FSDP)
        self.total_gpus = self.tp * self.ep * self.dp

    def parameter_count(self) -> Dict[str, int]:
        """Count parameters by component."""
        # Attention per layer
        head_dim = self.d_model // self.n_heads
        q_params = self.d_model * self.n_heads * head_dim
        kv_params = 2 * self.d_model * self.n_kv_heads * head_dim
        o_params = self.n_heads * head_dim * self.d_model
        attn_per_layer = q_params + kv_params + o_params

        # MoE FFN per layer
        expert_params = 3 * self.d_model * self.ffn_dim  # w1, w2, w3
        moe_per_layer = self.n_experts * expert_params
        router_per_layer = self.d_model * self.n_experts

        # Layer norms
        norm_per_layer = 2 * self.d_model

        # Embeddings
        embed_params = self.vocab_size * self.d_model

        return {
            'attention': self.layers * attn_per_layer,
            'moe_experts': self.layers * moe_per_layer,
            'router': self.layers * router_per_layer,
            'norms': self.layers * norm_per_layer,
            'embeddings': 2 * embed_params,
            'total': (self.layers * (attn_per_layer + moe_per_layer +
                     router_per_layer + norm_per_layer) + 2 * embed_params)
        }

    def memory_per_gpu(self, batch_size: int, seq_len: int) -> Dict[str, float]:
        """Compute memory usage per GPU in GB."""
        params = self.parameter_count()

        # Parameters sharded across DP (FSDP)
        # Experts sharded across EP
        expert_params_local = params['moe_experts'] / self.ep
        other_params_local = (params['attention'] + params['router'] +
                             params['norms'] + params['embeddings']) / self.dp

        total_params_local = expert_params_local + other_params_local

        # Model states (FP32 optimizer)
        param_bytes = total_params_local * 2  # BF16
        optimizer_bytes = total_params_local * 8  # FP32 moments
        grad_bytes = total_params_local * 2  # BF16

        # Activations (per layer, rough estimate)
        # Attention: 2 * batch * seq * d_model (Q, K/V)
        # MoE: batch * seq * d_model * 2 (input, output)
        act_per_layer = batch_size * seq_len * self.d_model * 4 * 2  # BF16
        activation_bytes = self.layers * act_per_layer

        return {
            'parameters': param_bytes / 1e9,
            'optimizer': optimizer_bytes / 1e9,
            'gradients': grad_bytes / 1e9,
            'activations': activation_bytes / 1e9,
            'total': (param_bytes + optimizer_bytes + grad_bytes +
                     activation_bytes) / 1e9
        }

    def communication_volume(self, batch_size: int, seq_len: int) -> Dict[str, float]:
        """Estimate communication volume per step in GB."""
        n_tokens = batch_size * seq_len

        # Expert routing: all-to-all per layer
        # Send/recv tokens to/from each expert partition
        tokens_moved = n_tokens * self.d_model * 2 * 2  # Input + output, bytes
        all_to_all_per_layer = tokens_moved / self.ep  # Per GPU

        # Gradient sync: FSDP AllGather and ReduceScatter
        params = self.parameter_count()
        non_expert_params = (params['attention'] + params['router'] +
                            params['norms'] + params['embeddings'])
        fsdp_volume = 2 * non_expert_params * 2  # AG + RS, BF16

        return {
            'all_to_all_per_layer_gb': all_to_all_per_layer / 1e9,
            'all_to_all_total_gb': self.layers * all_to_all_per_layer / 1e9,
            'fsdp_volume_gb': fsdp_volume / 1e9,
            'total_gb': (self.layers * all_to_all_per_layer + fsdp_volume) / 1e9
        }
```

### Parallelism Strategy

Mixtral uses a hybrid approach:

| Dimension | Strategy | Reason |
|-----------|----------|--------|
| Tensor Parallel | TP=1 or 2 | Attention is small, TP overhead not justified |
| Expert Parallel | EP=8 | One expert per GPU, no replication |
| Data Parallel | FSDP | Shard non-expert params across replicas |
| Sequence | Not used | Window attention limits sequence cost |

### Communication Pattern

```
Per forward pass (one micro-batch):
├── Layer 1
│   ├── Attention: local compute
│   └── MoE: all-to-all (tokens to experts)
├── Layer 2
│   ├── Attention: local compute
│   └── MoE: all-to-all
├── ...
└── Layer 32
    ├── Attention: local compute
    └── MoE: all-to-all

Backward pass: same pattern in reverse

After backward:
└── FSDP gradient sync (ReduceScatter + AllGather)
```

## Inference Optimization

Mistral/Mixtral excel at inference efficiency.

### KV Cache Budget

For serving with memory constraint $M$:

$$\text{Max batch size} = \frac{M - M_{\text{model}}}{\text{KV per sequence}}$$

With SWA and GQA:

$$\text{KV per sequence} = 2 \cdot L \cdot w \cdot g \cdot d \cdot 2$$

For Mistral 7B ($L=32$, $w=4096$, $g=8$, $d=128$):

$$= 2 \cdot 32 \cdot 4096 \cdot 8 \cdot 128 \cdot 2 = 512 \text{ MB}$$

Compare to LLaMA 2 7B without SWA/GQA at 32K context:

$$= 2 \cdot 32 \cdot 32768 \cdot 32 \cdot 128 \cdot 2 = 16 \text{ GB}$$

**32× more sequences can be batched** with Mistral's architecture.

### Speculative Decoding

Mistral's efficiency enables speculative decoding with draft models:

```python
class SpeculativeDecoder:
    """
    Speculative decoding using small draft model.

    Draft model proposes K tokens, target model verifies in parallel.
    """
    def __init__(
        self,
        target_model: nn.Module,  # e.g., Mixtral 8x7B
        draft_model: nn.Module,   # e.g., Mistral 7B
        k: int = 4
    ):
        self.target = target_model
        self.draft = draft_model
        self.k = k

    def generate_step(
        self,
        input_ids: torch.Tensor,
        target_cache: Optional[Any],
        draft_cache: Optional[Any]
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate tokens with speculative decoding.

        Returns:
            new_tokens: accepted tokens
            n_accepted: number of tokens accepted
        """
        # Draft model proposes K tokens autoregressively
        draft_tokens = []
        draft_probs = []

        x = input_ids
        for _ in range(self.k):
            logits = self.draft(x, kv_cache=draft_cache)
            probs = F.softmax(logits[:, -1], dim=-1)
            token = torch.multinomial(probs, 1)

            draft_tokens.append(token)
            draft_probs.append(probs.gather(-1, token))
            x = token

        draft_tokens = torch.cat(draft_tokens, dim=1)  # [batch, k]

        # Target model verifies all K tokens in parallel
        full_input = torch.cat([input_ids, draft_tokens], dim=1)
        target_logits = self.target(full_input, kv_cache=target_cache)

        # Compare probabilities and accept/reject
        n_accepted = 0
        for i in range(self.k):
            target_probs = F.softmax(target_logits[:, -(self.k - i)], dim=-1)
            draft_prob = draft_probs[i]
            target_prob = target_probs.gather(-1, draft_tokens[:, i:i+1])

            # Acceptance probability
            accept_prob = min(1.0, target_prob / draft_prob)

            if torch.rand(1) < accept_prob:
                n_accepted += 1
            else:
                # Reject this and all subsequent tokens
                break

        # Return accepted tokens
        return draft_tokens[:, :n_accepted], n_accepted
```

## Training Efficiency Analysis

### Compute Efficiency: Sparse vs Dense

For Mixtral with 8 experts, top-2:

$$\text{Active FLOPs ratio} = \frac{2}{8} = 0.25$$

But expert computation is only part of the model:

```python
def compute_active_ratio(
    n_experts: int,
    top_k: int,
    d_model: int,
    ffn_dim: int,
    n_heads: int,
    n_kv_heads: int
) -> float:
    """Compute ratio of active to total FLOPs."""
    head_dim = d_model // n_heads

    # Attention FLOPs per token (fully activated)
    qkv_flops = 2 * d_model * (n_heads + 2 * n_kv_heads) * head_dim
    attn_flops = 2 * d_model * d_model  # Roughly
    attn_total = qkv_flops + attn_flops

    # Expert FLOPs per token
    expert_flops = 3 * 2 * d_model * ffn_dim  # w1, w2, w3
    active_expert_flops = top_k * expert_flops
    total_expert_flops = n_experts * expert_flops

    # Active ratio
    active = attn_total + active_expert_flops
    total = attn_total + total_expert_flops

    return active / total

# Mixtral 8x7B
ratio = compute_active_ratio(
    n_experts=8, top_k=2,
    d_model=4096, ffn_dim=14336,
    n_heads=32, n_kv_heads=8
)
# Returns ~0.31 (31% of total FLOPs activated)
```

### Training Speed

Mixtral 8x7B processes tokens **slower** than Mistral 7B despite 6.5× more total parameters:

$$\text{Speed ratio} \approx \frac{\text{Active params}}{\text{Mistral params}} = \frac{12.9B}{7.3B} \approx 1.8\times$$

Training is ~1.8× slower per token than Mistral 7B, but achieves LLaMA 2 70B quality.

## Exercises

1. **Window size selection**: Derive the relationship between window size $w$, number of layers $L$, and effective context length. If you need 100K effective context with 32 layers, what window size is required?

??? success "Solution"
    **Exercise 1: Window Size Selection**

    **Sliding Window Attention (SWA) Properties:**

    With window size $w$ and $L$ layers, information propagates through layers:
    - Layer 1: Each token sees $w$ tokens
    - Layer 2: Each token sees tokens that saw $w$ tokens each → effective $2w$
    - Layer $L$: Effective context $\leq L \times w$

    **Effective context length:**
    $$C_{eff} = L \times w$$

    **For 100K effective context with L=32 layers:**
    $$w = \frac{C_{eff}}{L} = \frac{100,000}{32} = 3,125$$

    Round to power of 2 for efficiency:

    $$\boxed{w = 4096 \text{ tokens}}$$

    This gives effective context of $32 \times 4096 = 131K$ tokens.

    **Verification with Mistral's design:**
    - Mistral 7B: $w = 4096$, $L = 32$ → 131K effective context
    - Memory per position: $O(w)$ instead of $O(S)$

    **Trade-off table:**

    | Window Size | Layers | Effective Context | KV Cache (relative) |
    |-------------|--------|-------------------|---------------------|
    | 2048 | 32 | 65K | 1× |
    | 4096 | 32 | 131K | 2× |
    | 8192 | 32 | 262K | 4× |
    | 4096 | 64 | 262K | 2× (more layers) |

2. **GQA memory analysis**: Compare KV cache memory for (a) MHA with 32 heads, (b) MQA with 1 KV head, (c) GQA with 8 KV groups. Which provides the best quality/memory trade-off?

??? success "Solution"
    **Exercise 2: GQA Memory Analysis**

    **Setup:**
    - Hidden dimension: H = 4096
    - Head dimension: $d_h$ = 128
    - Query heads: $n_q$ = 32
    - Sequence length: S = 8192

    **KV cache per token (FP16):**
    $$M_{KV} = 2 \times n_{kv} \times d_h \times 2 \text{ bytes}$$

    **(a) MHA (Multi-Head Attention): $n_{kv}$ = 32**
    $$M_{MHA} = 2 \times 32 \times 128 \times 2 = 16,384 \text{ bytes/token}$$

    For S=8192: $16,384 \times 8192 = 134$ MB per layer

    **(b) MQA (Multi-Query Attention): $n_{kv}$ = 1**
    $$M_{MQA} = 2 \times 1 \times 128 \times 2 = 512 \text{ bytes/token}$$

    For S=8192: $512 \times 8192 = 4.2$ MB per layer

    Reduction: 32×

    **(c) GQA (Grouped-Query Attention): $n_{kv}$ = 8**
    $$M_{GQA} = 2 \times 8 \times 128 \times 2 = 4,096 \text{ bytes/token}$$

    For S=8192: $4,096 \times 8192 = 33.5$ MB per layer

    Reduction: 4×

    **Quality comparison (empirical):**

    | Method | Memory | Quality (vs MHA) | Sweet Spot |
    |--------|--------|------------------|------------|
    | MHA | 100% | 100% | Small models |
    | MQA | 3.1% | 95-97% | Extreme compression |
    | GQA-8 | 25% | 99%+ | ✓ Best trade-off |
    | GQA-4 | 12.5% | 98-99% | Good trade-off |

    $$\boxed{\text{GQA with 8 groups: 4× memory reduction with <1\% quality loss}}$$

3. **MoE compute efficiency**: For a model with 16 experts and top-2 routing, what fraction of expert FLOPs are activated? How does this change the optimal training data budget according to Chinchilla scaling?

??? success "Solution"
    **Exercise 3: MoE Compute Efficiency**

    **Mixtral configuration:**
    - Total experts: $E = 16$
    - Active experts per token: $k = 2$
    - Expert size: Same as dense FFN

    **Activated fraction:**
    $$\text{Active fraction} = \frac{k}{E} = \frac{2}{16} = 12.5\%$$

    **Effective FLOPs per token:**

    For dense model with FFN FLOPs $F_{FFN}$:

    $$F_{MoE} = \frac{k}{E} \times E \times F_{FFN} = k \times F_{FFN}$$

    Wait—each token uses $k$ full experts, so:

    $$F_{MoE} = k \times F_{FFN} = 2 \times F_{FFN}$$

    But total model has $E$ experts worth of parameters.

    **Parameter efficiency:**
    $$\frac{\text{Active params}}{\text{Total params}} = \frac{k}{E} = 12.5\%$$

    **Chinchilla scaling adjustment:**

    Chinchilla: $D_{opt} = 20N$ (optimal tokens = 20× parameters)

    For MoE, effective parameters for quality ≈ active parameters × scaling factor:

    $$N_{eff} \approx N_{active} \times \alpha$$

    Where $\alpha \approx 2-3$ (MoE quality multiplier from routing specialization).

    For Mixtral 8x7B:
    - Total params: $8 \times 7B + \text{shared} = 46.7B$
    - Active params: $\approx 12.9B$
    - Effective for quality: $\approx 25-40B$ equivalent

    **Optimal training budget:**
    $$D_{opt,MoE} = 20 \times N_{active} \times \alpha \approx 20 \times 12.9B \times 2.5 = 645B \text{ tokens}$$

    Mistral trained on more tokens (estimated 1-2T) because:
    1. Inference cost is low (12.9B active)
    2. Additional tokens continue improving quality

    $$\boxed{\text{12.5\% activation → can train on 8× more tokens at same FLOP budget}}$$

4. **Expert parallelism**: You have 64 GPUs and want to train Mixtral 8x22B (8 experts). Design a parallelism strategy considering TP, EP, and DP dimensions. What are the trade-offs?

??? success "Solution"
    **Exercise 4: Expert Parallelism Strategy**

    **Setup:**
    - 64 GPUs (8 nodes × 8 GPUs)
    - Mixtral 8x22B: 8 experts, ~22B params each
    - Total params: ~141B

    **Memory per expert:**
    - Parameters: $22B \times 2 = 44$ GB (FP16)
    - Optimizer (Adam): $22B \times 12 = 264$ GB (FP32)
    - Total static per expert: ~308 GB

    Can't fit one expert per GPU! Need sharding.

    **Strategy 1: TP=4, EP=8, DP=2**

    ```
    - Tensor Parallel: 4 GPUs share each expert (within node)
    - Expert Parallel: 8 expert groups (one per TP group)
    - Data Parallel: 2 replicas

    GPU Layout (8 nodes × 8 GPUs = 64):
    Node 0: Expert 0 (TP=4) + Expert 0 replica (TP=4)
    Node 1: Expert 1 (TP=4) + Expert 1 replica (TP=4)
    ...
    ```

    Memory per GPU: $\frac{44}{4} + \frac{264}{4 \times 2} = 11 + 33 = 44$ GB ✓

    **Strategy 2: TP=8, EP=8, DP=1**

    ```
    - Full node for each expert (TP=8)
    - Each node handles one expert
    - No DP (batch parallelism only)
    ```

    Memory per GPU: $\frac{44}{8} + \frac{264}{8} = 5.5 + 33 = 38.5$ GB ✓

    **Trade-off analysis:**

    | Strategy | TP | EP | DP | Comm Volume | Batch Size |
    |----------|----|----|----|--------------| -----------|
    | Strategy 1 | 4 | 8 | 2 | Lower TP comm | 2× larger |
    | Strategy 2 | 8 | 8 | 1 | Higher TP comm | Limited |

    **All-to-All communication for expert routing:**
    $$V_{A2A} = 2 \times B \times S \times H = 2 \times B \times 4096 \times 6144$$

    With DP=2, each replica handles half the batch → lower A2A volume.

    $$\boxed{\text{Strategy 1 (TP=4, EP=8, DP=2) balances memory and throughput}}$$

5. **Load balancing**: Implement a simulation of token routing with and without auxiliary loss. Measure the expert load imbalance (max/min ratio) after 1000 steps.

??? success "Solution"
    **Exercise 5: Load Balancing Simulation**

    ```python
    import numpy as np
    from dataclasses import dataclass
    from typing import List, Tuple

    @dataclass
    class RouterState:
        num_experts: int
        logit_bias: np.ndarray  # Auxiliary bias for load balancing

    class MoERouter:
        def __init__(self, num_experts: int, hidden_dim: int, top_k: int = 2):
            self.num_experts = num_experts
            self.top_k = top_k
            self.W_gate = np.random.randn(hidden_dim, num_experts) * 0.02
            self.aux_loss_weight = 0.01
            self.load_history = []

        def route(self, x: np.ndarray, use_aux_loss: bool = True) -> Tuple[np.ndarray, np.ndarray]:
            """
            Route tokens to experts.

            Args:
                x: [batch, hidden] input tokens
                use_aux_loss: whether to apply load balancing

            Returns:
                expert_indices: [batch, top_k]
                expert_weights: [batch, top_k]
            """
            # Compute routing logits
            logits = x @ self.W_gate  # [batch, num_experts]

            # Apply softmax
            probs = self._softmax(logits)

            # Select top-k experts
            top_k_indices = np.argsort(probs, axis=-1)[:, -self.top_k:]
            top_k_probs = np.take_along_axis(probs, top_k_indices, axis=-1)

            # Normalize weights
            top_k_weights = top_k_probs / top_k_probs.sum(axis=-1, keepdims=True)

            # Record load
            load = np.bincount(top_k_indices.flatten(), minlength=self.num_experts)
            self.load_history.append(load)

            # Compute and apply auxiliary loss gradient (simplified)
            if use_aux_loss:
                self._update_for_balance(load)

            return top_k_indices, top_k_weights

        def _softmax(self, x: np.ndarray) -> np.ndarray:
            exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
            return exp_x / exp_x.sum(axis=-1, keepdims=True)

        def _update_for_balance(self, load: np.ndarray):
            """Adjust gate weights to balance load."""
            # Penalize overloaded experts
            avg_load = load.mean()
            imbalance = (load - avg_load) / (avg_load + 1e-6)
            # Push routing away from overloaded experts
            self.W_gate -= self.aux_loss_weight * imbalance

        def get_imbalance_ratio(self) -> float:
            """Max/min load ratio from recent history."""
            if not self.load_history:
                return 1.0
            recent = np.array(self.load_history[-100:])
            avg_load = recent.mean(axis=0)
            return avg_load.max() / (avg_load.min() + 1e-6)

    def simulate_routing(num_steps: int = 1000, use_aux_loss: bool = True):
        """Simulate token routing over many steps."""
        router = MoERouter(num_experts=8, hidden_dim=512, top_k=2)

        for step in range(num_steps):
            # Random batch of tokens (in practice, comes from model)
            batch = np.random.randn(256, 512)
            router.route(batch, use_aux_loss=use_aux_loss)

            if step % 200 == 0:
                ratio = router.get_imbalance_ratio()
                print(f"Step {step}: imbalance ratio = {ratio:.2f}")

        return router.get_imbalance_ratio()

    # Run simulations
    print("With auxiliary loss:")
    ratio_with = simulate_routing(1000, use_aux_loss=True)
    print(f"Final imbalance: {ratio_with:.2f}x\n")

    print("Without auxiliary loss:")
    ratio_without = simulate_routing(1000, use_aux_loss=False)
    print(f"Final imbalance: {ratio_without:.2f}x")
    ```

    **Expected results:**

    | Condition | Max/Min Load Ratio | Efficiency |
    |-----------|-------------------|------------|
    | Without aux loss | 3-10× | 30-50% waste |
    | With aux loss | 1.1-1.5× | <5% waste |

    $$\boxed{\text{Auxiliary loss reduces imbalance from 5× to 1.3×}}$$

6. **Speculative decoding speedup**: If the draft model is 5× faster than the target and accepts 70% of proposed tokens on average with K=4 speculation depth, what is the expected speedup?

??? success "Solution"
    **Exercise 6: Speculative Decoding Speedup**

    **Given:**
    - Draft model: 5× faster than target
    - Acceptance rate: $p = 70\%$ per token
    - Speculation depth: $K = 4$

    **Speculative decoding process:**
    1. Draft model generates $K$ tokens
    2. Target model verifies all $K$ in parallel
    3. Accept prefix of tokens that match

    **Expected accepted tokens per iteration:**

    With acceptance probability $p$ per token:

    $$E[\text{accepted}] = \sum_{i=0}^{K} i \cdot p^i \cdot (1-p) + K \cdot p^K$$

    For $K=4$, $p=0.7$:

    $$E[\text{accepted}] = 0 \cdot 0.3 + 1 \cdot 0.7 \cdot 0.3 + 2 \cdot 0.49 \cdot 0.3 + 3 \cdot 0.343 \cdot 0.3 + 4 \cdot 0.2401$$

    $$= 0 + 0.21 + 0.294 + 0.309 + 0.960 = 1.77 \text{ tokens}$$

    Actually, simpler formula:

    $$E[\text{accepted}] = \frac{1 - p^{K+1}}{1-p} - 1 = \frac{1 - 0.7^5}{0.3} - 1 = \frac{1 - 0.168}{0.3} - 1 = 1.77$$

    **Time analysis:**

    Without speculation:
    - 1 target forward per token
    - Time per token: $T_{target}$

    With speculation:
    - $K$ draft forwards + 1 target forward (parallel verification)
    - Time: $K \cdot T_{draft} + T_{target} = K \cdot \frac{T_{target}}{5} + T_{target} = T_{target}(1 + K/5)$
    - Tokens generated: $E[\text{accepted}] + 1 = 2.77$ (including verified token)

    **Speedup calculation:**

    $$\text{Speedup} = \frac{E[\text{accepted}] + 1}{1 + K/5} = \frac{2.77}{1 + 4/5} = \frac{2.77}{1.8} = 1.54\times$$

    **More detailed model:**

    | Iteration | Draft Time | Target Time | Tokens | Amortized |
    |-----------|------------|-------------|--------|-----------|
    | 1 | 0.8T | 1.0T | 2.77 | 0.65T/token |
    | Baseline | 0 | 1.0T | 1 | 1.0T/token |

    $$\boxed{\text{Speedup} = \frac{1.0}{0.65} \approx 1.54\times}$$

    **Sensitivity analysis:**

    | Acceptance Rate | K | Expected Tokens | Speedup |
    |-----------------|---|-----------------|---------|
    | 60% | 4 | 2.13 | 1.18× |
    | 70% | 4 | 2.77 | 1.54× |
    | 80% | 4 | 3.59 | 2.00× |
    | 70% | 6 | 3.02 | 1.37× |

    Higher acceptance rates dramatically improve speedup. K=4 is near-optimal for p=70%.

## Invariant Summary

| Invariant | Primary Pressure | Response |
|---|---|---|
| Memory | Long context KV cache | GQA + sliding window attention |
| Compute | Efficient attention | Flash-style kernels |
| Communication | Moderate scaling | Simpler parallelism mix |

## Key Takeaways

1. **Sliding Window Attention enables long context**: Fixed memory regardless of sequence length, with effective context growing through layer stacking.

2. **GQA reduces KV cache**: Sharing key-value heads across query groups provides 4-8× memory reduction.

3. **Combined savings are multiplicative**: SWA + GQA together can reduce KV cache by 32× or more.

4. **Sparse MoE provides dense-quality at sparse cost**: Activating 2 of 8 experts achieves similar quality to a dense model 3× larger.

5. **Expert parallelism distributes MoE efficiently**: One expert per GPU with all-to-all token routing.

6. **Inference efficiency enables practical deployment**: Small KV cache means larger batch sizes and better hardware utilization.

7. **Architecture efficiency can rival scale**: Mistral's innovations show that clever architecture beats brute-force scaling.
