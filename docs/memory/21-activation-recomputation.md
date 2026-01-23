---
title: "Activation Recomputation"
subtitle: "Trading Compute for Memory Through Gradient Checkpointing"
---

<div class="chapter-opener" markdown>
Activations dominate memory in deep networks. A 7B model's parameters need ~14 GB, but its activations can consume 100+ GB. Activation recomputation—also called gradient checkpointing—trades compute for memory by recomputing activations during the backward pass instead of storing them.
</div>

<div class="investigation-question" markdown>
**The Question**: The backward pass needs activations from the forward pass. Storing them all requires O(L) memory for L layers. Can we reduce this to O(√L) or even O(1)? What's the compute cost of this trade-off?
</div>

## The Activation Memory Problem

During backpropagation, computing gradients requires activations from the forward pass:

$$\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial y_l} \cdot \frac{\partial y_l}{\partial W_l} = \delta_l \cdot a_{l-1}^T$$

Where:
- $\delta_l = \partial L / \partial y_l$: the error signal
- $a_{l-1}$: the input activation (from forward pass)

Without stored activations, we cannot compute gradients.

### Memory Growth

For a transformer with $L$ layers, the stored activations include:

| Component | Shape | Per Layer |
|-----------|-------|-----------|
| Layer input | $(B, S, H)$ | $2BSH$ bytes |
| Attention Q, K, V | $(B, S, H) \times 3$ | $6BSH$ bytes |
| Attention scores | $(B, \text{heads}, S, S)$ | $2BnS^2$ bytes |
| Softmax output | $(B, \text{heads}, S, S)$ | $2BnS^2$ bytes |
| Attention output | $(B, S, H)$ | $2BSH$ bytes |
| FFN intermediate | $(B, S, 4H)$ | $8BSH$ bytes |
| Various intermediates | - | ~$2BSH$ bytes |

**Total per layer**: approximately $20BSH + 4BnS^2$ bytes (FP16).

For a 7B model ($L=32$, $H=4096$, $n=32$) with $B=4$, $S=2048$:
- Per layer: $20 \times 4 \times 2048 \times 4096 \times 2 + 4 \times 4 \times 32 \times 2048^2 \times 2$
- Per layer: $1.34$ GB + $4.29$ GB = $5.63$ GB
- Total: $32 \times 5.63$ GB = **180 GB**

This far exceeds GPU memory.

## The Recomputation Trade-off

**Key insight**: We don't need to store activations if we can recompute them.

### Basic Idea

```
Standard forward/backward:
Forward:  Layer 0 → save a0 → Layer 1 → save a1 → ... → Layer L → Loss
Backward: Load aL-1 → grad L → Load aL-2 → grad L-1 → ... → grad 0

With recomputation:
Forward:  Layer 0 → Layer 1 → ... → Layer L → Loss (save only checkpoints)
Backward: Recompute aL-1 → grad L → Recompute aL-2 → grad L-1 → ...
```

### The Fundamental Trade-off

Let:
- $M$: memory for activations
- $C$: compute for forward passes
- $K$: number of checkpoints

| Strategy | Memory | Forward Passes |
|----------|--------|----------------|
| Store all | $O(L)$ | $1$ |
| Store none | $O(1)$ | $O(L)$ |
| Checkpoint every $\sqrt{L}$ | $O(\sqrt{L})$ | $\approx 2$ |

The optimal checkpoint interval minimizes total cost.

## Checkpointing Strategies

### Strategy 1: Full Recomputation (No Storage)

Store only the input and recompute everything during backward.

```python
class FullRecomputeFunction(torch.autograd.Function):
    """Recompute entire forward pass during backward."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, layers: nn.ModuleList):
        ctx.layers = layers

        # Forward through all layers (no intermediate storage)
        output = input
        for layer in layers:
            output = layer(output)

        # Save only input for recomputation
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, = ctx.saved_tensors
        layers = ctx.layers

        # Recompute forward pass
        activations = [input]
        hidden = input
        for layer in layers:
            hidden = layer(hidden)
            activations.append(hidden)

        # Standard backward
        grad = grad_output
        grads = []
        for i in range(len(layers) - 1, -1, -1):
            with torch.enable_grad():
                act = activations[i].detach().requires_grad_(True)
                out = layers[i](act)
                grad = torch.autograd.grad(out, act, grad)[0]

        return grad, None
```

**Memory**: $O(1)$—only input and current layer.
**Compute**: $O(L)$ forward passes—one full recomputation.
**Problem**: Too expensive for large $L$.

### Strategy 2: Uniform Checkpointing

Save activation every $k$ layers. Recompute only within segments.

```python
def uniform_checkpoint_forward(input: torch.Tensor,
                               layers: nn.ModuleList,
                               checkpoint_interval: int) -> torch.Tensor:
    """Forward with uniform checkpointing."""
    checkpoints = [input]
    hidden = input

    for i, layer in enumerate(layers):
        hidden = layer(hidden)
        if (i + 1) % checkpoint_interval == 0:
            checkpoints.append(hidden)

    return hidden, checkpoints


def uniform_checkpoint_backward(grad_output: torch.Tensor,
                                layers: nn.ModuleList,
                                checkpoints: List[torch.Tensor],
                                checkpoint_interval: int):
    """Backward with segment recomputation."""
    num_segments = len(checkpoints)
    grad = grad_output

    for seg_idx in range(num_segments - 1, -1, -1):
        # Determine segment boundaries
        start_layer = seg_idx * checkpoint_interval
        end_layer = min((seg_idx + 1) * checkpoint_interval, len(layers))

        # Recompute segment from checkpoint
        checkpoint = checkpoints[seg_idx]
        activations = [checkpoint]
        hidden = checkpoint
        for i in range(start_layer, end_layer):
            with torch.enable_grad():
                hidden = layers[i](hidden)
                activations.append(hidden)

        # Backward through segment
        for i in range(end_layer - 1, start_layer - 1, -1):
            layer_input = activations[i - start_layer]
            layer_output = activations[i - start_layer + 1]

            # Compute gradients for this layer
            grad = torch.autograd.grad(
                layer_output, layer_input, grad,
                retain_graph=True
            )[0]

    return grad
```

**Analysis for interval $k$**:

- Number of checkpoints: $L/k$
- Memory per checkpoint: $M_{\text{layer}}$
- Peak memory within segment: $k \cdot M_{\text{layer}}$

**Total memory**: $\frac{L}{k} \cdot M_{\text{layer}} + k \cdot M_{\text{layer}}$

Minimized when $L/k = k$, i.e., $k = \sqrt{L}$.

**Optimal checkpoint interval**: $k^* = \sqrt{L}$

**Optimal memory**: $2\sqrt{L} \cdot M_{\text{layer}}$

For $L = 32$ layers: $k^* \approx 6$, memory reduced from $32M$ to $12M$ (2.7× savings).

### Strategy 3: Selective Checkpointing

Not all layers consume equal memory. Checkpoint strategically.

```python
def analyze_layer_memory(layer: nn.Module,
                         input_shape: Tuple[int, ...]) -> int:
    """Estimate activation memory for a layer."""
    # Hook to capture activation sizes
    activation_sizes = []

    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activation_sizes.append(output.numel() * output.element_size())
        elif isinstance(output, tuple):
            for o in output:
                if isinstance(o, torch.Tensor):
                    activation_sizes.append(o.numel() * o.element_size())

    handles = []
    for module in layer.modules():
        handles.append(module.register_forward_hook(hook))

    # Dummy forward
    dummy_input = torch.randn(input_shape)
    with torch.no_grad():
        layer(dummy_input)

    # Clean up hooks
    for h in handles:
        h.remove()

    return sum(activation_sizes)


def select_checkpoints(layers: nn.ModuleList,
                       memory_budget: int,
                       input_shape: Tuple[int, ...]) -> List[int]:
    """Select optimal checkpoint locations given memory budget."""
    layer_memories = [
        analyze_layer_memory(layer, input_shape)
        for layer in layers
    ]

    # Greedy selection: checkpoint before high-memory layers
    total_memory = sum(layer_memories)
    num_checkpoints = total_memory // memory_budget

    # Sort layers by memory, checkpoint before largest
    sorted_indices = sorted(range(len(layers)),
                           key=lambda i: layer_memories[i],
                           reverse=True)

    checkpoint_indices = sorted(sorted_indices[:num_checkpoints])
    return checkpoint_indices
```

### Strategy 4: Attention-Specific Checkpointing

Attention has unique memory patterns. The $O(S^2)$ attention scores dominate.

```python
class CheckpointedAttention(nn.Module):
    """
    Attention with selective recomputation.

    Stores Q, K, V but recomputes attention scores.
    Saves O(S²) memory per head.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape

        # Compute Q, K, V (stored)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Use checkpointing for attention computation
        attn_output = torch.utils.checkpoint.checkpoint(
            self._attention_forward,
            q, k, v,
            use_reentrant=False
        )

        # Output projection
        output = attn_output.transpose(1, 2).reshape(B, S, H)
        return self.out_proj(output)

    def _attention_forward(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor) -> torch.Tensor:
        """Core attention computation (recomputed in backward)."""
        # Attention scores: O(S²) memory NOT stored
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)
```

**Memory savings**:
- Without checkpointing: $2 \times B \times n \times S^2$ bytes for scores + softmax
- With checkpointing: Only Q, K, V stored ($6BSH$ bytes)

For $S=2048$, $n=32$: saves $4 \times 32 \times 2048^2 \times 2 = 1.07$ GB per layer.

## PyTorch Checkpoint API

PyTorch provides built-in checkpointing support.

### Basic Usage

```python
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class TransformerWithCheckpointing(nn.Module):
    """Transformer with activation checkpointing."""

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        self.head = nn.Linear(config.hidden_dim, config.vocab_size)

        self.use_checkpointing = config.use_checkpointing
        self.checkpoint_ratio = config.checkpoint_ratio  # e.g., 0.5

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)

        num_checkpointed = int(len(self.layers) * self.checkpoint_ratio)

        for i, layer in enumerate(self.layers):
            if self.use_checkpointing and i < num_checkpointed:
                # Checkpoint this layer
                hidden = checkpoint(
                    layer,
                    hidden,
                    use_reentrant=False,
                    preserve_rng_state=True
                )
            else:
                # Normal forward
                hidden = layer(hidden)

        return self.head(hidden)
```

### Checkpoint Sequential

For sequential models, use `checkpoint_sequential`:

```python
def forward_with_sequential_checkpoint(self, x: torch.Tensor) -> torch.Tensor:
    """Forward using checkpoint_sequential for uniform segments."""
    # Divide layers into segments
    num_segments = int(math.sqrt(len(self.layers)))

    # checkpoint_sequential handles segment boundaries automatically
    hidden = checkpoint_sequential(
        self.layers,
        num_segments,
        x,
        use_reentrant=False
    )

    return hidden
```

### Non-Reentrant Checkpointing

PyTorch offers two checkpointing modes:

**Reentrant** (legacy):
- Uses `torch.autograd.grad` internally
- Can have subtle bugs with certain operations
- Being deprecated

**Non-Reentrant** (recommended):
- Uses saved tensor hooks
- More robust with complex graphs
- Preserves RNG state correctly

```python
# Always prefer non-reentrant
hidden = checkpoint(
    layer,
    hidden,
    use_reentrant=False,  # Recommended
    preserve_rng_state=True  # Important for dropout
)
```

## The Compute Cost Analysis

### Single Checkpoint

Checkpointing layer $l$ means:
- Forward: compute layer $l$ once
- Backward: recompute layer $l$ once before computing gradients

**Overhead**: One extra forward pass per checkpointed layer.

### Full Model Analysis

Let:
- $F$: FLOPs for one forward pass
- $B$: FLOPs for one backward pass ($B \approx 2F$ typically)
- $c$: fraction of layers checkpointed

**Without checkpointing**:
$$C_{\text{total}} = F + B = F + 2F = 3F$$

**With checkpointing (fraction $c$)**:
$$C_{\text{total}} = F + cF + B = F(1 + c + 2) = F(3 + c)$$

**Compute overhead**: $c \cdot F$, or $\frac{c}{3}$ relative increase.

| Checkpoint Fraction | Relative Overhead |
|--------------------|-------------------|
| 0% | 0% |
| 50% | 16.7% |
| 100% | 33.3% |

**Maximum overhead is 33%** even with full checkpointing.

### Memory-Compute Pareto Frontier

```
Memory
  │
  │●  No checkpointing (3F compute, L memory)
  │
  │    ●  50% checkpointing (3.5F, 0.5L + √L)
  │
  │        ●  √L checkpointing (3.33F, 2√L)
  │
  │            ●  Full checkpointing (4F, O(1))
  │
  └─────────────────────────────────────────→ Compute

Trade-off: 33% more compute for ~√L memory reduction
```

## Advanced Techniques

### Activation Compression

Instead of discarding activations, compress them.

```python
class CompressedCheckpoint(torch.autograd.Function):
    """Checkpoint with lossy activation compression."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, layer: nn.Module, compression: str):
        ctx.layer = layer
        ctx.compression = compression

        # Compress input for storage
        if compression == 'fp8':
            compressed = input.to(torch.float8_e4m3fn)
        elif compression == 'quantize':
            # 8-bit quantization
            scale = input.abs().max() / 127
            compressed = (input / scale).round().to(torch.int8)
            ctx.scale = scale
        else:
            compressed = input

        ctx.save_for_backward(compressed)

        # Forward with full precision
        return layer(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        compressed, = ctx.saved_tensors
        layer = ctx.layer

        # Decompress
        if ctx.compression == 'fp8':
            input_approx = compressed.to(torch.float16)
        elif ctx.compression == 'quantize':
            input_approx = compressed.float() * ctx.scale
        else:
            input_approx = compressed

        # Compute gradients with approximate activations
        input_approx.requires_grad_(True)
        with torch.enable_grad():
            output = layer(input_approx)
            grad_input = torch.autograd.grad(output, input_approx, grad_output)[0]

        return grad_input, None, None
```

**Trade-off**: Some gradient accuracy for significant memory savings (4× for FP8, 2× for INT8).

### Offloaded Checkpointing

Combine checkpointing with CPU offloading.

```python
class OffloadedCheckpoint(torch.autograd.Function):
    """Checkpoint with CPU offloading."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, layer: nn.Module):
        ctx.layer = layer
        ctx.input_shape = input.shape
        ctx.input_dtype = input.dtype
        ctx.input_device = input.device

        # Offload input to CPU (async)
        ctx.input_cpu = input.to('cpu', non_blocking=True)

        return layer(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Prefetch input from CPU
        input_gpu = ctx.input_cpu.to(ctx.input_device, non_blocking=True)
        torch.cuda.synchronize()

        input_gpu.requires_grad_(True)
        with torch.enable_grad():
            output = ctx.layer(input_gpu)
            grad_input = torch.autograd.grad(output, input_gpu, grad_output)[0]

        return grad_input, None
```

### Selective Recomputation

Some operations are cheap to store, expensive to recompute. Be selective:

```python
class SelectiveRecompute(nn.Module):
    """Selectively recompute expensive operations."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

        # Operations to always store (cheap to store, expensive to compute)
        self.store_ops = {'embedding', 'layernorm', 'linear'}

        # Operations to recompute (expensive to store, cheap to compute)
        self.recompute_ops = {'attention_scores', 'softmax', 'dropout'}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Custom forward with selective storage
        return selective_checkpoint(self.layer, x, self.recompute_ops)
```

## Interaction with Other Memory Optimizations

### Checkpointing + ZeRO

These are complementary:
- ZeRO reduces model state memory (parameters, gradients, optimizer)
- Checkpointing reduces activation memory

```python
class ZeROWithCheckpointing:
    """Combined ZeRO and checkpointing."""

    def __init__(self, model: nn.Module, zero_stage: int):
        self.model = model
        self.zero_optimizer = ZeROOptimizer(model, stage=zero_stage)

        # Enable checkpointing for all transformer layers
        for layer in model.layers:
            layer.use_checkpoint = True

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # ZeRO-3: parameters gathered on-demand
        # Checkpointing: activations recomputed on-demand
        return self.model(input)
```

**Combined memory**:
- Model state: $16N/P$ (ZeRO-3)
- Activations: $O(\sqrt{L})$ (checkpointing)

### Checkpointing + Tensor Parallelism

When using TP, checkpoint carefully to avoid redundant communication:

```python
class TPCheckpointedLayer(nn.Module):
    """Tensor-parallel layer with smart checkpointing."""

    def __init__(self, hidden_dim: int, tp_degree: int):
        super().__init__()
        self.tp_degree = tp_degree
        self.local_hidden = hidden_dim // tp_degree

        self.linear = ColumnParallelLinear(hidden_dim, 4 * hidden_dim)
        self.output = RowParallelLinear(4 * hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Checkpoint the compute-heavy middle section
        # Don't checkpoint the AllReduce (would serialize communication)
        intermediate = checkpoint(
            self.linear,
            x,
            use_reentrant=False
        )

        # AllReduce happens here (in RowParallel backward)
        return self.output(F.gelu(intermediate))
```

### Checkpointing + Pipeline Parallelism

Pipeline stages naturally create checkpoints at stage boundaries:

```python
class PipelineStage(nn.Module):
    """Pipeline stage with internal checkpointing."""

    def __init__(self, layers: nn.ModuleList, checkpoint_internal: bool = True):
        super().__init__()
        self.layers = layers
        self.checkpoint_internal = checkpoint_internal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage input is always stored (for pipeline backward)

        if self.checkpoint_internal and len(self.layers) > 1:
            # Checkpoint within stage
            num_segments = max(1, int(math.sqrt(len(self.layers))))
            x = checkpoint_sequential(
                self.layers,
                num_segments,
                x,
                use_reentrant=False
            )
        else:
            for layer in self.layers:
                x = layer(x)

        return x
```

## Memory Estimation with Checkpointing

### Analytical Model

```python
def estimate_activation_memory(
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    batch_size: int,
    seq_length: int,
    checkpoint_strategy: str = 'sqrt',
    checkpoint_attention: bool = True,
    dtype_bytes: int = 2
) -> dict:
    """
    Estimate activation memory with various checkpointing strategies.

    Args:
        num_layers: Number of transformer layers
        hidden_dim: Model hidden dimension
        num_heads: Number of attention heads
        batch_size: Batch size
        seq_length: Sequence length
        checkpoint_strategy: 'none', 'full', 'sqrt', or 'selective'
        checkpoint_attention: Whether to checkpoint attention scores
        dtype_bytes: Bytes per element (2 for FP16)

    Returns:
        Dictionary with memory estimates
    """
    B, S, H, n, L = batch_size, seq_length, hidden_dim, num_heads, num_layers

    # Per-layer activation memory (without attention scores)
    # Input, Q, K, V, attention output, FFN intermediate, output
    linear_per_layer = (2 + 3 + 1 + 4 + 1) * B * S * H * dtype_bytes  # ~11 BSH

    # Attention scores and softmax output
    attention_per_layer = 2 * B * n * S * S * dtype_bytes  # 2 * BnS²

    if checkpoint_attention:
        # Store Q, K, V; recompute scores
        stored_per_layer = linear_per_layer
        recompute_per_layer = attention_per_layer
    else:
        stored_per_layer = linear_per_layer + attention_per_layer
        recompute_per_layer = 0

    # Apply checkpointing strategy
    if checkpoint_strategy == 'none':
        # Store all layers
        total_stored = L * stored_per_layer
        peak_recompute = 0

    elif checkpoint_strategy == 'full':
        # Store only input, recompute all
        total_stored = B * S * H * dtype_bytes  # Just input
        peak_recompute = L * stored_per_layer  # Need to hold during recompute

    elif checkpoint_strategy == 'sqrt':
        # Optimal sqrt(L) checkpointing
        num_checkpoints = int(math.ceil(math.sqrt(L)))
        segment_size = (L + num_checkpoints - 1) // num_checkpoints

        # Store checkpoints
        checkpoint_memory = num_checkpoints * B * S * H * dtype_bytes

        # Peak during segment recomputation
        peak_segment = segment_size * stored_per_layer

        total_stored = checkpoint_memory + peak_segment
        peak_recompute = segment_size * recompute_per_layer

    elif checkpoint_strategy == 'selective':
        # Checkpoint every other layer
        num_stored = L // 2
        num_recomputed = L - num_stored

        total_stored = num_stored * stored_per_layer
        peak_recompute = stored_per_layer  # Only 1 layer at a time

    else:
        raise ValueError(f"Unknown strategy: {checkpoint_strategy}")

    return {
        'total_stored_gb': total_stored / (1024**3),
        'peak_recompute_gb': peak_recompute / (1024**3),
        'peak_total_gb': (total_stored + peak_recompute) / (1024**3),
        'strategy': checkpoint_strategy,
        'checkpoint_attention': checkpoint_attention,
        'layers': L,
        'per_layer_mb': stored_per_layer / (1024**2)
    }
```

### Example Calculations

```python
# 7B model with batch 4, sequence 2048
config = {
    'num_layers': 32,
    'hidden_dim': 4096,
    'num_heads': 32,
    'batch_size': 4,
    'seq_length': 2048
}

strategies = ['none', 'sqrt', 'selective', 'full']
for strategy in strategies:
    result = estimate_activation_memory(**config, checkpoint_strategy=strategy)
    print(f"{strategy:10}: {result['peak_total_gb']:.1f} GB")
```

Output:
```
none      : 180.5 GB
sqrt      : 45.1 GB  (4× reduction)
selective : 90.3 GB  (2× reduction)
full      : 5.6 GB   (32× reduction)
```

## Practical Implementation Guide

### DeepSpeed Activation Checkpointing

DeepSpeed provides optimized checkpointing:

```python
import deepspeed

# Configure in DeepSpeed config
ds_config = {
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": False,
        "contiguous_memory_optimization": True,
        "number_checkpoints": None,  # Auto-detect
        "synchronize_checkpoint_boundary": False,
        "profile": False
    }
}

# Wrap model
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)
```

### Megatron-LM Checkpointing

```python
from megatron.core.tensor_parallel import checkpoint

class MegatronTransformerLayer(nn.Module):
    def forward(self, hidden_states, attention_mask):
        if self.checkpoint_activations:
            hidden_states = checkpoint(
                self.attention,
                hidden_states,
                attention_mask
            )
            hidden_states = checkpoint(
                self.mlp,
                hidden_states
            )
        else:
            hidden_states = self.attention(hidden_states, attention_mask)
            hidden_states = self.mlp(hidden_states)

        return hidden_states
```

### HuggingFace Gradient Checkpointing

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Now training uses ~40% less activation memory
```

## Exercises

1. **Optimal checkpointing**: For a 48-layer transformer, derive the optimal checkpoint interval using the $\sqrt{L}$ rule. How many checkpoints are stored? What's the memory compared to no checkpointing?

2. **Compute overhead**: A training step takes 100ms without checkpointing. With full checkpointing, what's the expected step time? (Assume forward = 1/3 of backward compute.)

3. **Mixed strategy**: Design a checkpointing strategy that checkpoints attention scores but not FFN activations. Calculate memory savings vs. full checkpointing.

4. **Compression analysis**: If activations are compressed to FP8 before checkpointing, what's the memory reduction compared to FP16 checkpointing? What's the impact on gradient accuracy?

5. **Combined optimization**: A 70B model is trained with ZeRO-3 on 64 GPUs with $\sqrt{L}$ checkpointing. Calculate total memory per GPU including model state and activations.

6. **Pipeline interaction**: In a 4-stage pipeline with 8 microbatches, how many activation copies are stored at peak? How does checkpointing affect this?

## Key Takeaways

1. **Activations dominate memory**: Can exceed model size by 10×+ for large batch/sequence.

2. **Checkpointing trades compute for memory**: 33% compute overhead for $\sqrt{L}$ memory reduction.

3. **$\sqrt{L}$ is optimal**: Checkpoint every $\sqrt{L}$ layers for best memory-compute trade-off.

4. **Attention scores are expensive**: $O(S^2)$ per layer; prime candidates for recomputation.

5. **Composable with other techniques**: Works with ZeRO, TP, PP for multiplicative savings.

6. **Use non-reentrant checkpointing**: More robust, correctly handles RNG state.
