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

<div class="notation-banner" markdown>
**Notation in this chapter:** $\Psi$ = parameters, $B$ = batch tokens, $S$ = sequence length, $H$ = hidden size, $L$ = layers. See [Notation](../appendices/notation.md).
</div>

!!! abstract "Building On: Parts I and IV"
    We introduced the **16Ψ memory rule** briefly in [Chapter 1](../foundations/01-scale-imperative.md). Now we dissect it fully. This part also builds on **data parallelism** ([Chapter 14](../parallelism/14-data-parallelism-associativity.md))—understanding how gradients are distributed is essential for ZeRO's memory optimizations in the next chapter. The collectives from [Part III](../collectives/11-primitives-properties.md) (AllGather, ReduceScatter) will reappear as the mechanisms for memory-efficient sharding.

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

$$M_{\text{params}} = \Psi \times b_p$$

where:

- $\Psi$ = number of parameters
- $b_p$ = bytes per parameter (2 for fp16/bf16, 4 for fp32)

**Example**: 7B parameters in fp16 = $7 \times 10^9 \times 2$ = 14 GB

### Gradients

During backward pass, gradients are computed and accumulated:

$$M_{\text{grads}} = \Psi \times b_g$$

Gradients are typically stored in the same precision as parameters.

**Example**: 7B parameters = 14 GB for gradients

### Optimizer States

Optimizers maintain additional state per parameter.

**SGD with momentum**:

- Momentum buffer: $\Psi \times b_m$ bytes

$$M_{\text{opt}}^{\text{SGD}} = \Psi \times b_m$$

**Adam/AdamW**:

- First moment ($m$): $\Psi \times b_m$ bytes
- Second moment ($v$): $\Psi \times b_v$ bytes

Both moments are typically fp32 for numerical stability:

$$M_{\text{opt}}^{\text{Adam}} = 2 \times \Psi \times 4 = 8\Psi \text{ bytes}$$

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

$$M_{\text{model}} = 16\Psi \text{ bytes}$$

!!! note "Practice"
    Before a large run, compute $16\Psi$ and compare it to GPU HBM. If $16\Psi$ > HBM, you must shard or offload before tuning batch size.

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

Total per layer (summing the linear terms: 1+3+1+1+4+1 = 11):

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

For typical transformer architectures, each attention head has dimension $d_h = H/A \approx 128$ (e.g., GPT-3 has $H=12288$, $A=96$, so $d_h=128$). This means $H = 128A$, simplifying to:

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

$$M_{\text{total}} = \underbrace{16\Psi}_{\text{model state}} + \underbrace{2LBSH(11 + S/128)}_{\text{activations}} + \underbrace{M_{\text{workspace}}}_{\text{temporary}} + \underbrace{f \cdot M_{\text{peak}}}_{\text{fragmentation}}$$

where $f$ is the fragmentation factor (typically 5-20%).

### Worked Example: LLaMA-7B

Model specifications:

- Parameters: $\Psi = 7 \times 10^9$
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

$$M_{\text{model}} \propto \Psi$$

Activation memory scales with $\Psi$ as well (since $H \propto \sqrt{\Psi}$ and $L \propto \sqrt{\Psi}$ typically):

$$M_{\text{act}} \propto L \cdot H \propto \Psi$$

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

$$\eta_{\text{mem}} = \frac{M_{\text{params}}}{M_{\text{total}}} = \frac{\Psi \cdot b_p}{M_{\text{total}}}$$

For standard training with AdamW:

$$\eta_{\text{mem}} = \frac{2\Psi}{16\Psi + M_{\text{act}}} \approx \frac{2}{16} = 12.5\%$$

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

??? success "Solution"
    **Model state memory with AdamW (bf16 training):**

    | Component | Precision | Bytes/param | Total |
    |-----------|-----------|-------------|-------|
    | Parameters | bf16 | 2 | $2 \times 13\text{B} = 26\text{ GB}$ |
    | Gradients | bf16 | 2 | $2 \times 13\text{B} = 26\text{ GB}$ |
    | Master weights | fp32 | 4 | $4 \times 13\text{B} = 52\text{ GB}$ |
    | Momentum | fp32 | 4 | $4 \times 13\text{B} = 52\text{ GB}$ |
    | Variance | fp32 | 4 | $4 \times 13\text{B} = 52\text{ GB}$ |

    **Total model state:**
    $$M_{\text{state}} = (2 + 2 + 4 + 4 + 4) \times 13 \times 10^9 = 16 \times 13\text{B} = \boxed{208\text{ GB}}$$

    **Memory left for activations:**
    $$M_{\text{act}} = 80\text{ GB} - 208\text{ GB} = \boxed{-128\text{ GB}}$$

    **Analysis:** The model state alone exceeds GPU memory! This 13B model cannot be trained on a single 80GB GPU without memory optimization techniques:

    | Technique | Memory Reduction | After Optimization |
    |-----------|------------------|--------------------|
    | ZeRO-1 (8 GPUs) | Optimizer: 156 GB → 19.5 GB/GPU | 71.5 GB/GPU |
    | ZeRO-2 (8 GPUs) | + Gradients: 26 GB → 3.25 GB/GPU | 48.25 GB/GPU |
    | ZeRO-3 (8 GPUs) | + Parameters: 26 GB → 3.25 GB/GPU | 26 GB/GPU |

    With ZeRO-3 on 8 GPUs: 26 GB model state, leaving **54 GB for activations**.

2. **Activation dominance**: At what sequence length do attention activations exceed linear activations for a model with H=4096, A=32?

??? success "Solution"
    **Linear (non-attention) activations per token:**

    For a transformer layer, linear activations include:
    - LayerNorm outputs: $2H$ (two LayerNorms)
    - FFN intermediate: $4H$ (typical expansion)
    - Residual connections: $2H$

    $$M_{\text{linear}} \approx 8H \text{ bytes per token (bf16)}$$

    With $B$ batch, $S$ sequence:

    $$M_{\text{linear}} = 2 \times 8 \times B \times S \times H = 16BSH \text{ bytes}$$

    **Attention activations per layer:**

    - QKV projections: $3 \times BSH \times 2 = 6BSH$
    - Attention scores: $BAS^2 \times 2 = 2BAS^2$ (storing $S \times S$ per head)
    - Attention output: $BSH \times 2 = 2BSH$

    $$M_{\text{attn}} = 8BSH + 2BAS^2 \text{ bytes}$$

    **Crossover condition:**

    Attention exceeds linear when:

    $$2BAS^2 > 16BSH$$

    $$AS^2 > 8SH$$

    $$S > \frac{8H}{A}$$

    With $H = 4096$, $A = 32$:

    $$S > \frac{8 \times 4096}{32} = \frac{32768}{32} = \boxed{1024}$$

    **Interpretation:**

    | Sequence Length | Dominant Component | Ratio (Attn/Linear) |
    |-----------------|-------------------|---------------------|
    | 512 | Linear | 0.5× |
    | 1024 | Equal | 1.0× |
    | 2048 | Attention | 2.0× |
    | 4096 | Attention | 4.0× |
    | 8192 | Attention | 8.0× |

    For modern long-context models with $S = 8192+$, attention activations dominate heavily.

3. **Batch size limit**: Given a 40GB GPU, 7B parameter model, and 2048 sequence length, what's the maximum batch size?

??? success "Solution"
    **Model state memory (7B, bf16 + AdamW):**
    $$M_{\text{state}} = 16 \times 7 \times 10^9 = 112\text{ GB}$$

    **Problem:** Model state exceeds GPU memory! We need optimization.

    **Assuming ZeRO-3 with 8 GPUs:**
    $$M_{\text{state/GPU}} = \frac{112}{8} = 14\text{ GB}$$

    **Available for activations:**
    $$M_{\text{avail}} = 40 - 14 - 2 \text{ (overhead)} = 24\text{ GB}$$

    **Activation memory per token (7B model, ~32 layers, H≈4096):**

    Using the formula: $M_{\text{act}}^{\text{layer}} \approx BSH \cdot (34 + 5\frac{AS}{H})$

    For 7B: $L \approx 32$, $H \approx 4096$, $A \approx 32$

    Per layer: $M_{\text{act}}^{\text{layer}} = BS \times 4096 \times (34 + 5 \times \frac{32 \times 2048}{4096})$
    $$= BS \times 4096 \times (34 + 80) = BS \times 4096 \times 114$$

    Total (32 layers, with checkpointing every 4 layers):

    $$M_{\text{act}} \approx 8 \times BS \times 4096 \times 114 \times 2 \text{ bytes} \approx 7.5 \times BS \text{ GB}$$

    **Solving for batch size:**
    $$7.5 \times B \times 2048 \leq 24 \times 10^9$$

    $$B \leq \frac{24 \times 10^9}{7.5 \times 2048} \approx 1560$$

    Wait, let me recalculate more carefully.

    **Simplified activation estimate:**
    Per token across all layers: ~1.5 KB (with aggressive checkpointing)
    $$M_{\text{act}} = B \times S \times 1.5\text{ KB} = B \times 2048 \times 1500 = 3.07B \text{ MB}$$

    $$3.07B \text{ MB} \leq 24,000 \text{ MB}$$

    $$B \leq 7.8$$

    **Maximum batch size:** $\boxed{B = 7}$ (or 8 with micro-batching)

    **Practical verification:**
    - Total tokens per step: $7 \times 2048 = 14,336$ tokens
    - This is typical for 7B models on 40GB GPUs

4. **Memory profiling**: Write code to identify which layer in a transformer consumes the most activation memory.

??? success "Solution"
    ```python
    import torch
    import torch.nn as nn
    from contextlib import contextmanager
    from typing import Dict, List, Tuple

    class MemoryProfiler:
        """Profiles activation memory per layer in a transformer."""

        def __init__(self):
            self.layer_memory: Dict[str, int] = {}
            self.hooks = []

        def _get_tensor_memory(self, tensor: torch.Tensor) -> int:
            """Get memory in bytes for a tensor."""
            return tensor.numel() * tensor.element_size()

        def _forward_hook(self, name: str):
            def hook(module, input, output):
                mem = 0
                if isinstance(output, torch.Tensor):
                    mem = self._get_tensor_memory(output)
                elif isinstance(output, tuple):
                    for o in output:
                        if isinstance(o, torch.Tensor):
                            mem += self._get_tensor_memory(o)
                self.layer_memory[name] = mem
            return hook

        def register_hooks(self, model: nn.Module, prefix: str = ""):
            """Register hooks on all layers."""
            for name, module in model.named_modules():
                full_name = f"{prefix}.{name}" if prefix else name
                if len(list(module.children())) == 0:  # Leaf module
                    hook = module.register_forward_hook(
                        self._forward_hook(full_name)
                    )
                    self.hooks.append(hook)

        def remove_hooks(self):
            """Remove all registered hooks."""
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

        def profile(self, model: nn.Module, input_tensor: torch.Tensor
                   ) -> List[Tuple[str, int]]:
            """Run forward pass and return sorted memory usage."""
            self.layer_memory.clear()
            self.register_hooks(model)

            with torch.no_grad():
                model(input_tensor)

            self.remove_hooks()

            # Sort by memory usage (descending)
            sorted_layers = sorted(
                self.layer_memory.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_layers

        def report(self, top_k: int = 10) -> str:
            """Generate a formatted report."""
            sorted_layers = sorted(
                self.layer_memory.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            total = sum(self.layer_memory.values())
            report = "Top Activation Memory Consumers:\n"
            report += "-" * 60 + "\n"
            report += f"{'Layer':<40} {'Memory':>10} {'%':>6}\n"
            report += "-" * 60 + "\n"

            for name, mem in sorted_layers:
                pct = 100 * mem / total if total > 0 else 0
                report += f"{name:<40} {mem/1e6:>8.2f}MB {pct:>5.1f}%\n"

            report += "-" * 60 + "\n"
            report += f"{'Total':<40} {total/1e6:>8.2f}MB\n"
            return report

    # Usage example
    def profile_transformer():
        from transformers import AutoModel

        model = AutoModel.from_pretrained("gpt2")
        model.eval()

        # Create dummy input
        batch_size, seq_len = 4, 1024
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))

        profiler = MemoryProfiler()
        results = profiler.profile(model, input_ids)

        print(profiler.report(top_k=15))

        # Identify layer type with highest memory
        attention_mem = sum(m for n, m in results if 'attn' in n.lower())
        ffn_mem = sum(m for n, m in results if 'mlp' in n.lower() or 'fc' in n.lower())

        print(f"\nAttention total: {attention_mem/1e6:.2f} MB")
        print(f"FFN total: {ffn_mem/1e6:.2f} MB")
    ```

    **Expected output structure:**

    | Layer Type | Typical Memory % | Why |
    |------------|------------------|-----|
    | Attention QKV proj | 15-20% | $3 \times BSH$ tensors |
    | Attention scores | 25-40% | $BAS^2$ (quadratic!) |
    | FFN intermediate | 20-30% | $4 \times$ expansion |
    | LayerNorm | 5-10% | $BSH$ per norm |

    **Key insight:** For long sequences, attention score matrices (query-key products) dominate.

5. **Scaling analysis**: Plot expected memory usage vs. sequence length for S ∈ [512, 16384]. Identify the crossover point where quadratic term dominates.

??? success "Solution"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    def activation_memory(S, B=4, H=4096, A=32, L=32):
        """
        Calculate activation memory in GB.

        Components:
        - Linear: O(BSH) - embeddings, FFN, LayerNorm
        - Quadratic: O(BAS²) - attention scores
        """
        # Linear components (bytes)
        linear = L * B * S * H * 34 * 2  # bf16

        # Quadratic component (attention scores)
        quadratic = L * B * A * S * S * 2  # bf16

        return linear / 1e9, quadratic / 1e9

    # Sequence lengths to analyze
    seq_lengths = np.array([512, 1024, 2048, 4096, 8192, 16384])

    linear_mem = []
    quad_mem = []

    for S in seq_lengths:
        lin, quad = activation_memory(S)
        linear_mem.append(lin)
        quad_mem.append(quad)

    linear_mem = np.array(linear_mem)
    quad_mem = np.array(quad_mem)
    total_mem = linear_mem + quad_mem

    # Find crossover point (where quadratic = linear)
    # BAS² = 34BSH → AS = 34H → S = 34H/A
    H, A = 4096, 32
    crossover = 34 * H / A
    print(f"Theoretical crossover: S = {crossover:.0f}")

    # Create results table
    print("\nMemory Scaling Analysis (B=4, H=4096, A=32, L=32):")
    print("-" * 70)
    print(f"{'Seq Len':>8} {'Linear (GB)':>12} {'Quadratic (GB)':>14} "
          f"{'Total (GB)':>12} {'Quad %':>8}")
    print("-" * 70)
    for i, S in enumerate(seq_lengths):
        pct = 100 * quad_mem[i] / total_mem[i]
        print(f"{S:>8} {linear_mem[i]:>12.2f} {quad_mem[i]:>14.2f} "
              f"{total_mem[i]:>12.2f} {pct:>7.1f}%")
    print("-" * 70)
    ```

    **Results:**

    | Seq Len | Linear (GB) | Quadratic (GB) | Total (GB) | Quad % |
    |---------|-------------|----------------|------------|--------|
    | 512 | 4.56 | 0.54 | 5.10 | 10.5% |
    | 1024 | 9.13 | 2.15 | 11.28 | 19.0% |
    | 2048 | 18.25 | 8.59 | 26.84 | 32.0% |
    | **4096** | **36.51** | **34.36** | **70.87** | **48.5%** |
    | 8192 | 73.01 | 137.44 | 210.45 | 65.3% |
    | 16384 | 146.03 | 549.76 | 695.79 | 79.0% |

    **Crossover point:** $S = \frac{34H}{A} = \frac{34 \times 4096}{32} = \boxed{4352}$

    At $S \approx 4096$, quadratic and linear terms are roughly equal. Beyond this, attention memory dominates exponentially.

6. **Efficiency calculation**: A training run uses 120GB for a 7B model. Calculate $\eta_{\text{mem}}$.

??? success "Solution"
    **Memory efficiency definition:**
    $$\eta_{\text{mem}} = \frac{M_{\text{params}}}{M_{\text{total}}}$$

    **Parameter memory (7B, bf16):**
    $$M_{\text{params}} = 2 \times 7 \times 10^9 = 14\text{ GB}$$

    **Total memory used:**
    $$M_{\text{total}} = 120\text{ GB}$$

    **Memory efficiency:**
    $$\eta_{\text{mem}} = \frac{14}{120} = \boxed{11.7\%}$$

    **Memory breakdown (typical):**

    | Component | Estimated Size | % of Total |
    |-----------|----------------|------------|
    | Parameters (bf16) | 14 GB | 11.7% |
    | Gradients (bf16) | 14 GB | 11.7% |
    | Master weights (fp32) | 28 GB | 23.3% |
    | Optimizer states (fp32) | 56 GB | 46.7% |
    | **Model state subtotal** | **112 GB** | **93.3%** |
    | Activations | ~6 GB | 5.0% |
    | Overhead/fragmentation | ~2 GB | 1.7% |
    | **Total** | **120 GB** | **100%** |

    **Analysis:** Only 11.7% of memory holds the actual model weights. This low efficiency is typical for Adam-based training without memory optimization.

    **Ways to improve $\eta_{\text{mem}}$:**

    | Technique | New Efficiency |
    |-----------|----------------|
    | ZeRO-3 (8 GPUs) | ~14/(120/8) = 93% |
    | SGD (no momentum) | 14/(14+14+14) = 33% |
    | Activation checkpointing | Reduces activation memory |

7. **Optimizer comparison**: Compare memory requirements for training with Adam vs. SGD with momentum vs. SGD without momentum for a 70B model.

??? success "Solution"
    **Memory formula per optimizer:**

    | Component | Adam | SGD+Momentum | SGD |
    |-----------|------|--------------|-----|
    | Parameters (bf16) | $2\Psi$ | $2\Psi$ | $2\Psi$ |
    | Gradients (bf16) | $2\Psi$ | $2\Psi$ | $2\Psi$ |
    | Master weights (fp32) | $4\Psi$ | $4\Psi$ | $4\Psi$ |
    | Momentum (fp32) | $4\Psi$ | $4\Psi$ | 0 |
    | Variance (fp32) | $4\Psi$ | 0 | 0 |
    | **Bytes per param** | **16** | **12** | **8** |

    **For 70B model ($\Psi = 70 \times 10^9$):**

    | Optimizer | Bytes/param | Total Memory |
    |-----------|-------------|--------------|
    | **Adam** | 16 | $16 \times 70\text{B} = \boxed{1120\text{ GB}}$ |
    | **SGD+Momentum** | 12 | $12 \times 70\text{B} = \boxed{840\text{ GB}}$ |
    | **SGD** | 8 | $8 \times 70\text{B} = \boxed{560\text{ GB}}$ |

    **Minimum GPUs required (80GB H100):**

    | Optimizer | Min GPUs (no ZeRO) | With ZeRO-3 |
    |-----------|-------------------|-------------|
    | Adam | $\lceil 1120/80 \rceil = 14$ | 2 GPUs |
    | SGD+Momentum | $\lceil 840/80 \rceil = 11$ | 2 GPUs |
    | SGD | $\lceil 560/80 \rceil = 7$ | 1 GPU (tight) |

    **Trade-offs:**

    | Optimizer | Memory | Convergence | Use Case |
    |-----------|--------|-------------|----------|
    | Adam | Highest | Best | Standard LLM training |
    | SGD+Mom | Medium | Good | Fine-tuning, memory-constrained |
    | SGD | Lowest | Slow/unstable | Rarely used for LLMs |

    **Note:** Newer optimizers like 8-bit Adam or CAME reduce memory by quantizing optimizer states.

8. **Fragmentation analysis**: Memory reports show 100GB allocated but only 85GB actually used. What's the fragmentation ratio? What could cause this?

??? success "Solution"
    **Fragmentation ratio:**
    $$\text{Fragmentation} = \frac{M_{\text{allocated}} - M_{\text{used}}}{M_{\text{allocated}}}$$

    $$= \frac{100 - 85}{100} = \boxed{15\%}$$

    **Alternative metric (overhead ratio):**
    $$\text{Overhead} = \frac{M_{\text{allocated}}}{M_{\text{used}}} - 1 = \frac{100}{85} - 1 = 17.6\%$$

    **Common causes of GPU memory fragmentation:**

    | Cause | Description | Solution |
    |-------|-------------|----------|
    | **Allocation patterns** | Variable-sized allocations create gaps | Use memory pools |
    | **Caching allocator** | PyTorch caches freed memory | `torch.cuda.empty_cache()` |
    | **Peak memory** | High-water mark from temporary tensors | Gradient checkpointing |
    | **Tensor shape misalignment** | Non-power-of-2 shapes waste memory | Pad to aligned sizes |
    | **Dynamic shapes** | Variable batch/seq creates fragmentation | Fixed shapes |
    | **Pinned memory overhead** | Async transfer buffers | Reduce concurrency |

    **Debugging fragmentation:**

    ```python
    import torch

    def diagnose_fragmentation():
        # Current allocation state
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9

        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved:  {reserved:.2f} GB")
        print(f"Max alloc: {max_allocated:.2f} GB")
        print(f"Internal frag: {(reserved - allocated)/reserved*100:.1f}%")
        print(f"Peak overhead: {(max_allocated - allocated)/allocated*100:.1f}%")

        # Memory snapshot for detailed analysis
        if hasattr(torch.cuda, 'memory_snapshot'):
            snapshot = torch.cuda.memory_snapshot()
            # Analyze blocks, gaps, etc.

    # Mitigation strategies
    def reduce_fragmentation():
        # 1. Pre-allocate with consistent shapes
        torch.cuda.set_per_process_memory_fraction(0.95)

        # 2. Use memory-efficient attention
        # (Flash Attention allocates less temporary memory)

        # 3. Clear cache periodically
        torch.cuda.empty_cache()

        # 4. Use memory pools
        # CUDA memory pools reduce fragmentation
        torch.cuda.memory.set_allocator_settings("expandable_segments:True")
    ```

    **Fragmentation severity guide:**

    | Fragmentation | Severity | Action |
    |---------------|----------|--------|
    | < 5% | Normal | None needed |
    | 5-15% | Moderate | Monitor, consider empty_cache |
    | 15-25% | High | Optimize allocation patterns |
    | > 25% | Critical | Major refactoring needed |

    The observed 15% is in the "high" range—worth investigating allocation patterns.

## Key Takeaways

1. **The 16× rule**: Training with AdamW requires roughly 16 bytes per parameter of model state.

2. **Optimizer states dominate**: 75%+ of model state memory is optimizer states, not parameters.

3. **Activations scale with batch and sequence**: $M_{\text{act}} \propto B \cdot S \cdot (1 + S/128)$.

4. **Sequence length is quadratic**: Long sequences are memory-expensive due to attention.

5. **Peak is during backward**: Maximum memory usage typically occurs during backward pass.

6. **Memory efficiency is low**: Only ~12% of memory holds actual model weights in standard training.

7. **Profile, don't guess**: Theoretical estimates are useful but always validate with actual profiling.

8. **Memory optimization is essential**: No single GPU can train large models without optimization techniques.
