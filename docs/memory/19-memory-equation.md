---
title: "The Memory Equation"
subtitle: "Anatomy of GPU Memory in Training"
---

<div class="chapter-opener" markdown>
Training large models is fundamentally a memory problem. Before we can solve it, we must understand it precisely: what consumes memory, how much, and when? The memory equation gives us this understanding.
</div>

<div class="investigation-question" markdown>
**The Question**: A 7B parameter model in fp16 is 14GB. But training it requires 120GB+ of memory. Where does the other 106GB go?
</div>

## The Memory Budget

GPU memory during training holds four categories of data:

```
┌─────────────────────────────────────────────────────┐
│                  GPU Memory                          │
├─────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────┐    │
│  │           Model State Memory                │    │
│  │  ┌─────────────┬──────────┬──────────────┐  │    │
│  │  │ Parameters  │Gradients │  Optimizer   │  │    │
│  │  │     P       │    P     │    States    │  │    │
│  │  └─────────────┴──────────┴──────────────┘  │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │          Activation Memory                  │    │
│  │  ┌─────────────────────────────────────┐    │    │
│  │  │ Intermediate values saved for       │    │    │
│  │  │ backward pass                       │    │    │
│  │  └─────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │          Temporary Memory                   │    │
│  │  ┌─────────────────────────────────────┐    │    │
│  │  │ Workspace for operations (matmul,   │    │    │
│  │  │ flash attention, communication)     │    │    │
│  │  └─────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │          Fragmentation                      │    │
│  │  ┌─────────────────────────────────────┐    │    │
│  │  │ Unusable holes in allocated memory  │    │    │
│  │  └─────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

## Model State Memory

Model state memory is persistent throughout training. It holds everything needed to represent and update the model.

### Parameters

The model weights themselves:

$$M_{\text{params}} = N \times b_p$$

where:

- $N$ = number of parameters
- $b_p$ = bytes per parameter (2 for fp16/bf16, 4 for fp32)

**Example**: 7B parameters in fp16 = $7 \times 10^9 \times 2$ = 14 GB

### Gradients

During backward pass, gradients are computed and accumulated:

$$M_{\text{grads}} = N \times b_g$$

Gradients are typically stored in the same precision as parameters.

**Example**: 7B parameters = 14 GB for gradients

### Optimizer States

Optimizers maintain additional state per parameter.

**SGD with momentum**:

- Momentum buffer: $N \times b_m$ bytes

$$M_{\text{opt}}^{\text{SGD}} = N \times b_m$$

**Adam/AdamW**:

- First moment ($m$): $N \times b_m$ bytes
- Second moment ($v$): $N \times b_v$ bytes

Both moments are typically fp32 for numerical stability:

$$M_{\text{opt}}^{\text{Adam}} = 2 \times N \times 4 = 8N \text{ bytes}$$

### The Model State Equation

For mixed-precision training with AdamW:

| Component | Precision | Bytes per Parameter |
|-----------|-----------|---------------------|
| Parameters | fp16/bf16 | 2 |
| Gradients | fp16/bf16 | 2 |
| Master weights | fp32 | 4 |
| First moment | fp32 | 4 |
| Second moment | fp32 | 4 |
| **Total** | | **16** |

$$M_{\text{model}} = 16N \text{ bytes}$$

For 7B parameters: $16 \times 7 \times 10^9 = 112$ GB

**This is why training a 7B model requires ~8× the memory of just storing the weights.**

### Why Master Weights?

Mixed-precision training keeps fp32 "master" copies of weights:

```python
# Simplified mixed-precision training step
def training_step(model_fp16, optimizer, loss_fn, data):
    # Forward pass in fp16
    with autocast(dtype=torch.float16):
        output = model_fp16(data)
        loss = loss_fn(output)

    # Backward pass: gradients in fp16
    scaler.scale(loss).backward()

    # Unscale gradients
    scaler.unscale_(optimizer)

    # Optimizer step: updates fp32 master weights
    optimizer.step()  # optimizer holds fp32 weights

    # Copy back to fp16 model
    copy_fp32_to_fp16(optimizer.param_groups, model_fp16)
```

fp32 master weights are necessary because:
1. Small gradient updates may underflow in fp16
2. fp16 has only 3-4 decimal digits of precision
3. Accumulated rounding errors cause training instability

## Activation Memory

Activations are the intermediate values computed during forward pass and saved for backward computation.

### Why Store Activations?

The backward pass computes gradients using the chain rule:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W}$$

where $y = f(W, x)$.

Computing $\frac{\partial y}{\partial W}$ typically requires $x$ and/or $y$:

- Linear layer: $y = Wx$, $\frac{\partial y}{\partial W} = x^T$, need $x$
- ReLU: $y = \max(0, x)$, $\frac{\partial y}{\partial x} = \mathbf{1}[x > 0]$, need mask
- Softmax: need $y$ for Jacobian computation
- LayerNorm: need $\mu, \sigma, \hat{x}$

### Activation Memory in Transformers

For a transformer with:

- $L$ = number of layers
- $B$ = batch size
- $S$ = sequence length
- $H$ = hidden dimension
- $A$ = number of attention heads

Each transformer layer stores:

| Activation | Shape | Size |
|------------|-------|------|
| Input to LayerNorm 1 | $[B, S, H]$ | $BSH$ |
| Attention QKV | $[3, B, A, S, H/A]$ | $3BSH$ |
| Attention scores | $[B, A, S, S]$ | $BAS^2$ |
| Attention output | $[B, S, H]$ | $BSH$ |
| Input to LayerNorm 2 | $[B, S, H]$ | $BSH$ |
| FFN intermediate | $[B, S, 4H]$ | $4BSH$ |
| FFN output | $[B, S, H]$ | $BSH$ |

Total per layer:

$$M_{\text{act/layer}} \approx 11BSH + BAS^2$$

For the full model:

$$M_{\text{activations}} = L \cdot (11BSH + BAS^2) \cdot b_a$$

where $b_a$ is bytes per activation (typically 2 for fp16/bf16).

### The Sequence Length Problem

Attention scores scale quadratically with sequence length:

$$M_{\text{attention}} = L \cdot B \cdot A \cdot S^2 \cdot b_a$$

**Example**: L=32, B=1, A=32, S=8192, bf16:
$$32 \times 1 \times 32 \times 8192^2 \times 2 = 137.4 \text{ GB}$$

Just for attention scores! This is why long-context training requires Flash Attention and sequence parallelism.

### Activation Memory Formula

The complete activation memory for a transformer:

$$M_{\text{act}} = 2 \cdot L \cdot B \cdot S \cdot H \cdot \left(11 + \frac{A \cdot S}{H}\right)$$

For typical models where $H = 128A$ (head dimension 128):

$$M_{\text{act}} \approx 2 \cdot L \cdot B \cdot S \cdot H \cdot \left(11 + \frac{S}{128}\right)$$

When $S < 1408$: linear term dominates (11 > S/128)
When $S > 1408$: quadratic term dominates

## Temporary Memory

Operations require workspace memory that exists only during computation.

### Matrix Multiplication Workspace

cuBLAS and cuDNN use workspace for:

- Tiling intermediate results
- Efficient transpose operations
- Algorithm-specific storage

Workspace size depends on algorithm selection:

| Algorithm | Workspace | Speed |
|-----------|-----------|-------|
| GEMM (default) | Minimal | Baseline |
| GEMM + Tensor Cores | 0-100 MB | Faster |
| Strassen-like | Large | Asymptotically faster |

### Flash Attention Workspace

Flash Attention uses a different memory profile:

**Standard attention**:
```
Q, K, V → Scores (S²) → Softmax (S²) → Output
Memory: O(S²) for attention matrix
```

**Flash Attention**:
```
Q, K, V → Tiled computation → Output
Memory: O(S) - only current tile in SRAM
```

Flash Attention avoids materializing the full $S \times S$ attention matrix, but requires SRAM workspace for:

- Current query/key/value tiles
- Running softmax statistics (max, sum)
- Partial output accumulation

### Communication Buffers

Distributed training requires buffers for:

| Operation | Buffer Size |
|-----------|-------------|
| AllReduce (ring) | 2 × chunk size |
| AllGather | (P-1) × chunk size |
| ReduceScatter | (P-1) × chunk size |
| AlltoAll | Variable by routing |

These buffers are often allocated from a dedicated pool.

## The Complete Memory Equation

Total GPU memory requirement:

$$M_{\text{total}} = M_{\text{model}} + M_{\text{act}} + M_{\text{temp}} + M_{\text{frag}}$$

Expanding each term:

$$M_{\text{total}} = \underbrace{16N}_{\text{model state}} + \underbrace{2LBSH(11 + S/128)}_{\text{activations}} + \underbrace{M_{\text{workspace}}}_{\text{temporary}} + \underbrace{f \cdot M_{\text{peak}}}_{\text{fragmentation}}$$

where $f$ is the fragmentation factor (typically 5-20%).

### Worked Example: LLaMA-7B

Model specifications:

- Parameters: $N = 7 \times 10^9$
- Layers: $L = 32$
- Hidden: $H = 4096$
- Heads: $A = 32$
- Head dim: 128

Training configuration:

- Batch size: $B = 1$
- Sequence length: $S = 2048$

**Model state memory**:
$$M_{\text{model}} = 16 \times 7 \times 10^9 = 112 \text{ GB}$$

**Activation memory**:
$$M_{\text{act}} = 2 \times 32 \times 1 \times 2048 \times 4096 \times (11 + 2048/128)$$
$$= 2 \times 32 \times 2048 \times 4096 \times 27$$
$$= 2 \times 7.25 \times 10^9 \text{ bytes} = 14.5 \text{ GB}$$

**Temporary + fragmentation**: ~10-15 GB

**Total**: ~137-142 GB

This exceeds the 80GB of an A100! Hence the need for memory optimization techniques.

## Memory Scaling Analysis

### Scaling with Model Size

Model state scales linearly with parameters:
$$M_{\text{model}} \propto N$$

Activation memory scales with $N$ as well (since $H \propto \sqrt{N}$ and $L \propto \sqrt{N}$ typically):
$$M_{\text{act}} \propto L \cdot H \propto N$$

**Total memory scales linearly with model size** (for fixed batch and sequence).

### Scaling with Batch Size

Model state is independent of batch size.

Activation memory scales linearly with batch:
$$M_{\text{act}} \propto B$$

**Doubling batch size approximately doubles activation memory.**

### Scaling with Sequence Length

Model state is independent of sequence length.

Activation memory has a quadratic component:
$$M_{\text{act}} \propto S + S^2$$

**Long sequences are memory-expensive.** At S=8192, the quadratic term dominates.

### The Memory Wall

There's a fundamental tension:

```
Want: Larger batch → Better GPU utilization
Want: Longer sequence → Better context understanding
Want: Larger model → Better capability

Have: Fixed GPU memory

Reality: Can't have all three
```

This motivates all memory optimization techniques:

- **ZeRO**: Shard model state across GPUs
- **Activation checkpointing**: Trade compute for activation memory
- **Offloading**: Use CPU/NVMe as extended memory
- **Sequence parallelism**: Shard activation memory
- **Flash Attention**: Reduce attention memory

## Memory Profiling

### Understanding Peak Memory

Peak memory often occurs at specific points:

```
Memory
  │    ╭──╮
  │   ╱    ╲     Peak during
  │  ╱      ╲    backward pass
  │ ╱        ╲
  ├╱──────────╲────────
  │            ╲
  └───────────────────→ Time
    Forward  Backward  Optimizer
```

The peak typically occurs during backward pass because:
1. All activations are stored
2. Gradients are being computed
3. Some operations need workspace

### PyTorch Memory Tracking

```python
import torch

def memory_stats():
    """Get current GPU memory statistics."""
    return {
        'allocated': torch.cuda.memory_allocated() / 1e9,
        'reserved': torch.cuda.memory_reserved() / 1e9,
        'max_allocated': torch.cuda.max_memory_allocated() / 1e9,
    }

def profile_training_step(model, batch, optimizer, loss_fn):
    """Profile memory during a training step."""
    torch.cuda.reset_peak_memory_stats()

    # Before forward
    print(f"Before forward: {memory_stats()}")

    # Forward
    output = model(batch)
    print(f"After forward: {memory_stats()}")

    loss = loss_fn(output)

    # Backward
    loss.backward()
    print(f"After backward: {memory_stats()}")

    # Optimizer
    optimizer.step()
    optimizer.zero_grad()
    print(f"After optimizer: {memory_stats()}")

    return memory_stats()['max_allocated']
```

### Memory Timeline Visualization

```python
def create_memory_timeline(model, batch, optimizer, loss_fn):
    """Create a detailed memory timeline."""
    import torch.cuda.memory as mem

    # Start recording
    mem.reset_peak_memory_stats()

    snapshots = []

    def capture(label):
        snapshots.append({
            'label': label,
            'allocated': torch.cuda.memory_allocated(),
            'reserved': torch.cuda.memory_reserved(),
        })

    capture('start')

    # Hook into forward
    hooks = []
    for name, module in model.named_modules():
        def forward_hook(m, inp, out, name=name):
            capture(f'after_{name}')
        hooks.append(module.register_forward_hook(forward_hook))

    output = model(batch)
    capture('forward_complete')

    # Remove hooks
    for h in hooks:
        h.remove()

    loss = loss_fn(output)
    loss.backward()
    capture('backward_complete')

    optimizer.step()
    optimizer.zero_grad()
    capture('optimizer_complete')

    return snapshots
```

## Memory Breakdown by Component

### Empirical Analysis

For a LLaMA-7B style model, typical breakdown:

```
Memory Breakdown (% of total):
├── Model State: 75-80%
│   ├── Parameters: 10%
│   ├── Gradients: 10%
│   └── Optimizer States: 55-60%
├── Activations: 10-15%
├── Temporary: 5-10%
└── Fragmentation: 5-10%
```

**Key insight**: Optimizer states dominate. This is why ZeRO focuses on sharding optimizer states first.

### Layer-by-Layer Analysis

```python
def analyze_layer_memory(model, sample_input):
    """Analyze memory usage per layer."""
    layer_memory = {}

    def make_hook(name):
        def hook(module, input, output):
            # Compute output size
            if isinstance(output, torch.Tensor):
                size = output.numel() * output.element_size()
            elif isinstance(output, tuple):
                size = sum(o.numel() * o.element_size()
                          for o in output if isinstance(o, torch.Tensor))
            else:
                size = 0
            layer_memory[name] = size
        return hook

    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(sample_input)

    for h in hooks:
        h.remove()

    # Sort by memory usage
    sorted_layers = sorted(layer_memory.items(),
                          key=lambda x: x[1], reverse=True)

    return sorted_layers
```

## Memory Estimation Before Training

### Theoretical Estimation

```python
def estimate_memory_requirements(
    num_params: int,
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    batch_size: int,
    seq_length: int,
    precision: str = 'bf16',
    optimizer: str = 'adamw'
) -> dict:
    """
    Estimate GPU memory requirements for training.

    Returns:
        Dictionary with memory breakdown in bytes
    """
    # Bytes per element
    param_bytes = 2 if precision in ('fp16', 'bf16') else 4

    # Model state memory
    params_mem = num_params * param_bytes
    grads_mem = num_params * param_bytes

    if optimizer == 'adamw':
        # fp32 master weights + first moment + second moment
        optimizer_mem = num_params * (4 + 4 + 4)
    elif optimizer == 'sgd_momentum':
        optimizer_mem = num_params * 4  # momentum buffer
    else:
        optimizer_mem = 0

    model_state_mem = params_mem + grads_mem + optimizer_mem

    # Activation memory per layer
    head_dim = hidden_dim // num_heads

    # Linear activations: 11 * BSH (approximate)
    linear_act = 11 * batch_size * seq_length * hidden_dim * param_bytes

    # Attention scores: B * A * S * S
    attention_scores = (batch_size * num_heads * seq_length * seq_length
                       * param_bytes)

    activation_mem = num_layers * (linear_act + attention_scores)

    # Temporary memory (rough estimate: 10% of activations)
    temp_mem = int(activation_mem * 0.1)

    # Fragmentation (rough estimate: 10% of total)
    subtotal = model_state_mem + activation_mem + temp_mem
    fragmentation = int(subtotal * 0.1)

    total = subtotal + fragmentation

    return {
        'parameters': params_mem,
        'gradients': grads_mem,
        'optimizer_states': optimizer_mem,
        'model_state_total': model_state_mem,
        'activations': activation_mem,
        'temporary': temp_mem,
        'fragmentation': fragmentation,
        'total': total,
        'total_gb': total / (1024**3)
    }


# Example usage
memory = estimate_memory_requirements(
    num_params=7_000_000_000,
    num_layers=32,
    hidden_dim=4096,
    num_heads=32,
    batch_size=1,
    seq_length=2048
)

print(f"Estimated memory: {memory['total_gb']:.1f} GB")
print(f"  Model state: {memory['model_state_total']/1e9:.1f} GB")
print(f"  Activations: {memory['activations']/1e9:.1f} GB")
```

### Validation Against Actual Usage

Always validate estimates with actual profiling:

```python
def validate_memory_estimate(model, batch, estimate):
    """Compare estimated vs actual memory usage."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Measure actual
    optimizer = torch.optim.AdamW(model.parameters())

    output = model(batch)
    loss = output.mean()
    loss.backward()
    optimizer.step()

    actual = torch.cuda.max_memory_allocated()
    estimated = estimate['total']

    error = abs(actual - estimated) / actual * 100

    print(f"Estimated: {estimated/1e9:.2f} GB")
    print(f"Actual: {actual/1e9:.2f} GB")
    print(f"Error: {error:.1f}%")

    return actual, estimated, error
```

## The Memory Efficiency Ratio

Define memory efficiency as:

$$\eta_{\text{mem}} = \frac{M_{\text{params}}}{M_{\text{total}}} = \frac{N \cdot b_p}{M_{\text{total}}}$$

For standard training with AdamW:

$$\eta_{\text{mem}} = \frac{2N}{16N + M_{\text{act}}} \approx \frac{2}{16} = 12.5\%$$

Only 12.5% of memory holds actual model parameters!

**Memory optimization goal**: Increase $\eta_{\text{mem}}$ by reducing the denominator.

| Technique | $\eta_{\text{mem}}$ |
|-----------|---------------------|
| Standard AdamW | 12.5% |
| ZeRO-1 | ~15% |
| ZeRO-2 | ~20% |
| ZeRO-3 | ~40% |
| ZeRO-3 + CPU offload | ~60% |

## Exercises

1. **Memory calculation**: A 13B parameter model uses bf16 training with AdamW. Calculate the model state memory. If GPU has 80GB, how much is left for activations?

2. **Activation dominance**: At what sequence length do attention activations exceed linear activations for a model with H=4096, A=32?

3. **Batch size limit**: Given a 40GB GPU, 7B parameter model, and 2048 sequence length, what's the maximum batch size?

4. **Memory profiling**: Write code to identify which layer in a transformer consumes the most activation memory.

5. **Scaling analysis**: Plot expected memory usage vs. sequence length for S ∈ [512, 16384]. Identify the crossover point where quadratic term dominates.

6. **Efficiency calculation**: A training run uses 120GB for a 7B model. Calculate $\eta_{\text{mem}}$.

7. **Optimizer comparison**: Compare memory requirements for training with Adam vs. SGD with momentum vs. SGD without momentum for a 70B model.

8. **Fragmentation analysis**: Memory reports show 100GB allocated but only 85GB actually used. What's the fragmentation ratio? What could cause this?

## Key Takeaways

1. **The 16× rule**: Training with AdamW requires roughly 16 bytes per parameter of model state.

2. **Optimizer states dominate**: 75%+ of model state memory is optimizer states, not parameters.

3. **Activations scale with batch and sequence**: $M_{\text{act}} \propto B \cdot S \cdot (1 + S/128)$.

4. **Sequence length is quadratic**: Long sequences are memory-expensive due to attention.

5. **Peak is during backward**: Maximum memory usage typically occurs during backward pass.

6. **Memory efficiency is low**: Only ~12% of memory holds actual model weights in standard training.

7. **Profile, don't guess**: Theoretical estimates are useful but always validate with actual profiling.

8. **Memory optimization is essential**: No single GPU can train large models without optimization techniques.
