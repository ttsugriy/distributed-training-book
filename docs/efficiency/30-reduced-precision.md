---
title: "Reduced Precision Training"
subtitle: "The Bit-Level Economics of Computation"
---

<div class="chapter-opener" markdown>
Every bit costs bandwidth. Every mantissa bit costs compute. Mixed-precision training exploits the gap between what hardware computes efficiently and what training actually requires.
</div>

<div class="investigation-question" markdown>
**The Question**: FP32 has 23 mantissa bits. FP16 has 10. BF16 has 7. Yet all three train similar quality models. Why don't those 16 missing bits matter—and when do they?
</div>

!!! abstract "Chapter Map"
    **Prerequisites**: [Chapter 2](../foundations/02-extended-roofline.md) (arithmetic intensity and hardware capabilities)

    **Key insight**: Training tolerates low precision in forward/backward passes because gradients are averaged over many samples. BF16's 8-bit exponent preserves dynamic range (critical for large gradients); FP32 master weights preserve precision across many small updates. FP8 pushes the frontier further with careful scaling.

## The Anatomy of Floating-Point

To understand reduced precision, we must first understand what we're reducing.

### IEEE 754 Representation

A floating-point number is represented as:

$$x = (-1)^s \times 2^{e-\text{bias}} \times (1 + m)$$

Where:

- $s$: sign bit (1 bit)
- $e$: exponent (determines range)
- $m$: mantissa/significand (determines precision)

### Common Formats

| Format | Sign | Exponent | Mantissa | Total | Range | Precision |
|--------|------|----------|----------|-------|-------|-----------|
| FP32 | 1 | 8 | 23 | 32 | $\pm 3.4 \times 10^{38}$ | $\sim 7$ decimal digits |
| FP16 | 1 | 5 | 10 | 16 | $\pm 6.5 \times 10^4$ | $\sim 3$ decimal digits |
| BF16 | 1 | 8 | 7 | 16 | $\pm 3.4 \times 10^{38}$ | $\sim 2$ decimal digits |
| TF32 | 1 | 8 | 10 | 19 | $\pm 3.4 \times 10^{38}$ | $\sim 3$ decimal digits |
| FP8 (E4M3) | 1 | 4 | 3 | 8 | $\pm 448$ | $\sim 1$ decimal digit |
| FP8 (E5M2) | 1 | 5 | 2 | 8 | $\pm 5.7 \times 10^4$ | $< 1$ decimal digit |

### The Precision-Range Trade-off

Given a fixed bit budget, we must choose between:

**More exponent bits** → Larger dynamic range
**More mantissa bits** → Higher precision within range

```
FP16:  [S][EEEEE][MMMMMMMMMM]     5-bit exp, 10-bit mantissa
BF16:  [S][EEEEEEEE][MMMMMMM]     8-bit exp, 7-bit mantissa
```

This trade-off is why BF16 emerged: deep learning needs range (gradients span many orders of magnitude) more than precision.

## Why Precision Can Be Reduced

### The Statistical Argument

Neural network training is fundamentally stochastic:

$$g_B = \frac{1}{B} \sum_{i=1}^B \nabla L(x_i, \theta) = \nabla \mathbb{E}[L] + \epsilon_B$$

The gradient noise $\epsilon_B$ has variance $\sigma^2/B$.

**Key insight**: If gradient noise already introduces error at scale $\sigma/\sqrt{B}$, adding quantization noise $\delta_q$ doesn't matter as long as:

$$\delta_q \ll \frac{\sigma}{\sqrt{B}}$$

For typical training:

- Gradient noise: $\sim 10\%$ relative error
- FP16 rounding: $\sim 0.1\%$ relative error

The quantization noise is dominated by the inherent stochasticity.

### The Loss Landscape Argument

SGD doesn't need to follow the exact gradient—it needs to follow a direction that decreases loss:

$$\nabla L \cdot g_{\text{approx}} > 0$$

Reduced precision changes the gradient direction slightly, but not enough to flip the sign of the directional derivative.

### The Regularization Argument

Quantization noise acts as regularization, similar to:

- Dropout
- Weight noise
- Label smoothing

Some studies show slightly *better* generalization with reduced precision, likely due to this implicit regularization.

## Mixed-Precision Training

The key insight: **not all operations need full precision**.

### The AMP Algorithm

Automatic Mixed Precision (AMP) from Micikevicius et al., 2018:

```python
# Mixed precision training loop
def train_step_amp(model, optimizer, data, target, scaler):
    optimizer.zero_grad()

    # Forward pass in FP16
    with torch.cuda.amp.autocast():
        output = model(data)  # FP16 compute
        loss = criterion(output, target)  # FP16 compute

    # Backward pass with scaled loss (FP16 gradients)
    scaler.scale(loss).backward()

    # Unscale gradients for clipping (optional)
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    # Optimizer step with FP32 master weights
    scaler.step(optimizer)
    scaler.update()
```

### The Three-Part Strategy

**1. FP32 Master Weights**

Weights are stored and updated in FP32:

```python
class MixedPrecisionOptimizer:
    def __init__(self, model, base_optimizer, lr: float):
        # Master weights in FP32
        self.master_weights = {
            name: param.data.float().clone()
            for name, param in model.named_parameters()
        }
        self.model = model
        self.base_optimizer = base_optimizer
        self.lr = lr

    def step(self):
        # Gradients accumulated in FP16, now in master copy
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Update master weight (FP32)
                master = self.master_weights[name]
                grad = param.grad.float()
                master.add_(grad, alpha=-self.lr)

                # Copy back to model (FP16)
                param.data.copy_(master.half())
```

**Why FP32 master weights?**

Weight updates can be tiny:

$$\Delta w = \eta \cdot g \approx 10^{-4} \times 10^{-3} = 10^{-7}$$

In FP16 with weights $\sim 1$:

- Smallest representable difference: $2^{-10} \approx 10^{-3}$
- Update $10^{-7}$ rounds to zero!

FP32 accumulates these tiny updates until they become significant.

**2. FP16 Forward/Backward**

Compute-intensive operations run in FP16:

| Operation | Precision | Reason |
|-----------|-----------|--------|
| Matrix multiply | FP16 | Tensor Cores, 8× speedup |
| Convolution | FP16 | Tensor Cores, 8× speedup |
| Activation functions | FP16 | Simple element-wise |
| Attention scores | FP16 | Dominated by matmul |

**3. FP32 for Sensitive Operations**

Some operations need higher precision:

| Operation | Precision | Reason |
|-----------|-----------|--------|
| Softmax | FP32 | Exponentials overflow |
| LayerNorm | FP32 | Variance accumulation |
| Loss computation | FP32 | Log-sum-exp stability |
| Gradient reduction | FP32 | Accumulation over many values |

### Precision Selection Rules

```python
def select_precision(operation: str, tensor_size: int) -> str:
    """
    Select appropriate precision for an operation.
    """
    # Always FP32: numerically sensitive
    if operation in ['softmax', 'log_softmax', 'layer_norm',
                      'batch_norm', 'loss', 'exp', 'log']:
        return 'fp32'

    # Always FP16: compute-bound, Tensor Core friendly
    if operation in ['matmul', 'conv2d', 'linear']:
        return 'fp16'

    # Depends on size: element-wise operations
    if operation in ['relu', 'gelu', 'add', 'mul']:
        # Small tensors: FP32 has negligible overhead
        # Large tensors: FP16 saves bandwidth
        return 'fp16' if tensor_size > 1024 else 'fp32'

    # Default: FP32 for safety
    return 'fp32'
```

## Loss Scaling

### The Underflow Problem

FP16 has limited dynamic range:

- Smallest positive normal: $2^{-14} \approx 6 \times 10^{-5}$
- Smallest subnormal: $2^{-24} \approx 6 \times 10^{-8}$

Gradients can be smaller:

- Typical gradient: $10^{-3}$ to $10^{-6}$
- Deep network gradients: $10^{-7}$ to $10^{-10}$

Many gradients underflow to zero in FP16!

### The Scaling Solution

Multiply loss by a large factor before backward:

$$\tilde{L} = s \cdot L$$

Gradients scale proportionally:

$$\tilde{g} = s \cdot g$$

After backward, divide gradients by $s$:

$$g = \tilde{g} / s$$

This shifts gradient values into FP16's representable range.

### Dynamic Loss Scaling

```python
class DynamicLossScaler:
    """
    Automatically adjust loss scale to avoid overflow/underflow.
    """
    def __init__(self,
                 init_scale: float = 65536.0,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.steps_since_growth = 0
        self.consecutive_good_steps = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass."""
        return loss * self.scale

    def unscale_gradients(self, optimizer):
        """Unscale gradients after backward pass."""
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data /= self.scale

    def check_overflow(self, optimizer) -> bool:
        """Check if any gradient overflowed."""
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        return True
        return False

    def update(self, optimizer) -> bool:
        """
        Update loss scale based on gradient health.
        Returns True if step should proceed, False if overflow occurred.
        """
        if self.check_overflow(optimizer):
            # Overflow: reduce scale, skip step
            self.scale *= self.backoff_factor
            self.consecutive_good_steps = 0

            # Zero out bad gradients
            for group in optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.zero_()

            return False

        # No overflow: maybe increase scale
        self.consecutive_good_steps += 1
        if self.consecutive_good_steps >= self.growth_interval:
            self.scale *= self.growth_factor
            self.consecutive_good_steps = 0

        return True
```

### Loss Scale Dynamics

```
Loss Scale over Training:

65536 |        ____
      |       /    \____
32768 |      /          \______
      |_____/                  \____
16384 |                            \_____
      |
      +------------------------------------→ Steps
         ↑       ↑  ↑       ↑
      warmup  spike spike  late
                          training
```

- **Warmup**: Scale often needs to grow
- **Spikes**: Overflow triggers scale reduction
- **Late training**: Often stabilizes at a particular scale

## BF16: The Deep Learning Format

### Why BF16 Emerged

FP16's limited range ($\pm 6.5 \times 10^4$) causes problems:

- Activation spikes during attention
- Large gradient magnitudes early in training
- Exploding gradients in deep networks

BF16 trades mantissa bits for exponent bits:

- Same range as FP32 ($\pm 3.4 \times 10^{38}$)
- Simpler conversion: just truncate lower 16 bits

```python
def fp32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Convert FP32 to BF16 by truncating mantissa."""
    # View as 32-bit unsigned int to avoid sign issues
    x_int = x.view(np.uint32)

    # Round to nearest (add 0x8000 for rounding, not truncation)
    x_int = x_int + 0x8000

    # Zero out lower 16 bits
    x_int = x_int & 0xFFFF0000

    # View back as float
    return x_int.view(np.float32)
```

### BF16 Simplifies Training

With BF16, loss scaling is often unnecessary:

```python
# FP16: needs loss scaling
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast(dtype=torch.float16):
    output = model(data)
    loss = criterion(output, target)
scaled_loss = scaler.scale(loss)
scaled_loss.backward()
scaler.step(optimizer)
scaler.update()

# BF16: simpler, no scaling needed
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(data)
    loss = criterion(output, target)
loss.backward()
optimizer.step()
```

### BF16 vs FP16 Comparison

| Aspect | FP16 | BF16 |
|--------|------|------|
| Range | $\pm 6.5 \times 10^4$ | $\pm 3.4 \times 10^{38}$ |
| Precision | $\sim 0.1\%$ | $\sim 0.8\%$ |
| Loss scaling | Required | Usually not needed |
| Tensor Core support | A100, H100 | A100, H100 |
| CPU support | Limited | x86 (AVX-512 BF16) |
| Conversion to FP32 | Non-trivial | Trivial (bit shift) |

## TF32: Transparent Precision Reduction

TensorFloat-32 (TF32) is NVIDIA's compromise format:

```
FP32:  [S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]  8-bit exp, 23-bit mantissa
TF32:  [S][EEEEEEEE][MMMMMMMMMM]               8-bit exp, 10-bit mantissa
BF16:  [S][EEEEEEEE][MMMMMMM]                  8-bit exp, 7-bit mantissa
```

### How TF32 Works

TF32 is not a storage format—it's a compute format:

1. Read FP32 inputs
2. Round mantissa to 10 bits
3. Compute with TF32 precision
4. Store result in FP32

```python
# TF32 is enabled by default on A100+
# Disable for bit-exact reproducibility:
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

### TF32 Performance

| Operation | FP32 TFLOPs | TF32 TFLOPs | Speedup |
|-----------|-------------|-------------|---------|
| A100 matmul | 19.5 | 156 | 8× |
| A100 conv | 19.5 | 156 | 8× |
| H100 matmul | 67 | 495 | 7.4× |

TF32 provides Tensor Core acceleration transparently, without code changes.

## FP8: The Next Frontier

FP8 pushes precision reduction further, targeting inference and increasingly training.

### Two FP8 Variants

**E4M3** (4-bit exponent, 3-bit mantissa):

- Range: $\pm 448$
- Precision: 8 values per power of 2
- Best for: Forward pass (weights, activations)

**E5M2** (5-bit exponent, 2-bit mantissa):

- Range: $\pm 57344$
- Precision: 4 values per power of 2
- Best for: Backward pass (gradients need more range)

### FP8 Training Recipe

```python
class FP8TrainingConfig:
    """Configuration for FP8 training."""

    # Per-tensor scaling for FP8
    def compute_scale(self, tensor: torch.Tensor,
                      fp8_format: str = 'e4m3') -> float:
        """
        Compute scaling factor to maximize FP8 utilization.
        """
        if fp8_format == 'e4m3':
            fp8_max = 448.0
        else:  # e5m2
            fp8_max = 57344.0

        # Scale to use full FP8 range
        tensor_max = tensor.abs().max().item()
        if tensor_max == 0:
            return 1.0

        return fp8_max / tensor_max

    def quantize_to_fp8(self, tensor: torch.Tensor,
                        scale: float,
                        fp8_format: str = 'e4m3') -> torch.Tensor:
        """Quantize tensor to FP8 with given scale."""
        scaled = tensor * scale
        # Clamp to FP8 range
        if fp8_format == 'e4m3':
            clamped = torch.clamp(scaled, -448, 448)
        else:
            clamped = torch.clamp(scaled, -57344, 57344)

        # Round to FP8 representable values
        # (Implementation depends on hardware support)
        return self._round_to_fp8(clamped, fp8_format)
```

### Per-Tensor vs Per-Channel Scaling

FP8 typically uses per-tensor or per-channel scaling:

```python
def per_tensor_scale(tensor: torch.Tensor) -> float:
    """Single scale for entire tensor."""
    return 448.0 / tensor.abs().max()

def per_channel_scale(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Different scale per channel."""
    max_vals = tensor.abs().amax(dim=dim, keepdim=True)
    return 448.0 / max_vals
```

Per-channel scaling provides better accuracy but requires tracking more scale factors.

### FP8 Challenges

1. **Scale management**: Need to track and update per-tensor scales
2. **Range limitations**: E4M3 max of 448 requires careful activation design
3. **Hardware support**: Currently H100+ for training
4. **Accuracy sensitivity**: Some models/layers degrade with FP8

## Hardware Acceleration

### Tensor Cores

NVIDIA Tensor Cores accelerate low-precision matrix operations:

```
Tensor Core Operation:
D = A × B + C

Where:

- A, B: FP16/BF16/FP8 matrices (fragment tiles)
- C, D: FP16/BF16/FP32 matrices
```

Performance by generation:

| GPU | FP32 | FP16 | BF16 | TF32 | FP8 |
|-----|------|------|------|------|-----|
| V100 | 15.7 | 125 | - | - | - |
| A100 | 19.5 | 312 | 312 | 156 | - |
| H100 | 67 | 990 | 990 | 495 | 1979 |

### Tile Size Constraints

Tensor Cores operate on tiles, requiring specific dimensions:

```python
def is_tensor_core_aligned(M: int, N: int, K: int) -> bool:
    """Check if matrix dimensions are Tensor Core friendly."""
    # Requirements vary by GPU generation and precision
    # A100/H100 with FP16/BF16:
    return M % 8 == 0 and N % 8 == 0 and K % 8 == 0

def pad_for_tensor_cores(tensor: torch.Tensor) -> torch.Tensor:
    """Pad tensor dimensions to multiples of 8."""
    *batch_dims, m, k = tensor.shape

    new_m = ((m + 7) // 8) * 8
    new_k = ((k + 7) // 8) * 8

    if new_m == m and new_k == k:
        return tensor

    padded = torch.zeros(*batch_dims, new_m, new_k,
                         dtype=tensor.dtype, device=tensor.device)
    padded[..., :m, :k] = tensor
    return padded
```

### Memory Bandwidth Benefits

Beyond compute, reduced precision improves memory efficiency:

| Format | Weight Memory | Activation Memory | Bandwidth |
|--------|---------------|-------------------|-----------|
| FP32 | 100% | 100% | 100% |
| FP16/BF16 | 50% | 50% | 50% |
| FP8 | 25% | 25% | 25% |

For memory-bound operations, reduced precision provides proportional speedup.

## Numerical Stability Analysis

### Error Propagation in Forward Pass

Consider a layer: $y = Wx + b$

With FP16 compute:

$$\hat{y} = \text{fl}(Wx) + b + \epsilon$$

where $|\epsilon| \lesssim n \cdot u \cdot |W||x|$ and $u = 2^{-11}$ for FP16.

Through $L$ layers:

$$|\epsilon_L| \lesssim L \cdot n \cdot u \cdot \prod_{i=1}^L |W_i|$$

**Implications**:

- Error grows with depth $L$
- Error grows with layer width $n$
- Error multiplied by weight magnitudes

### Catastrophic Cancellation

When subtracting nearly equal numbers:

$$\text{fl}(a - b) \approx (a - b)(1 + \delta)$$

If $a \approx b$, relative error explodes:

$$\frac{|\hat{y} - y|}{|y|} \approx \frac{|a|}{|a - b|} \cdot u$$

This occurs in:

- Softmax: $e^{x_i} - e^{x_j}$ when $x_i \approx x_j$
- LayerNorm: variance computation when inputs are similar
- Residual connections: $x + f(x)$ when $f(x) \approx 0$

### Overflow in Softmax

Standard softmax implementation overflows:

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

For FP16 with $x = 12$: $e^{12} \approx 163000 > 65504$ (overflow!)

**Solution**: Subtract maximum before exponentiation:

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

```python
def stable_softmax_fp16(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax for FP16."""
    # Compute max in FP32 for safety
    x_max = x.float().max(dim=-1, keepdim=True).values

    # Subtract max in FP16
    x_shifted = x - x_max.half()

    # Exp and normalize
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)
```

## Layer-Specific Considerations

### Embedding Layers

Embeddings are often kept in FP32:

```python
class MixedPrecisionEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # Store in FP32
        self.weight = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim)
        )

    def forward(self, indices):
        # Cast to FP16 for compute
        return F.embedding(indices, self.weight.half())
```

**Reason**: Embedding gradients are very sparse (only accessed indices get gradients). FP32 ensures small updates accumulate correctly.

### Normalization Layers

LayerNorm and BatchNorm use FP32 for statistics:

```python
def layer_norm_mixed_precision(x: torch.Tensor,
                                weight: torch.Tensor,
                                bias: torch.Tensor,
                                eps: float = 1e-5) -> torch.Tensor:
    """LayerNorm with FP32 statistics."""
    # Compute statistics in FP32
    x_fp32 = x.float()
    mean = x_fp32.mean(dim=-1, keepdim=True)
    var = x_fp32.var(dim=-1, keepdim=True, unbiased=False)

    # Normalize in FP32
    x_norm = (x_fp32 - mean) / torch.sqrt(var + eps)

    # Scale and shift, cast back to input dtype
    return (x_norm * weight.float() + bias.float()).to(x.dtype)
```

**Reason**: Variance computation is sensitive to cancellation errors.

### Attention Layers

Attention scores can be computed in FP16, but accumulation often needs higher precision:

```python
def attention_with_mixed_precision(Q, K, V, mask=None):
    """Attention with careful precision handling."""
    d_k = K.size(-1)

    # QK^T in FP16 (Tensor Core friendly)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e4)  # Not -inf for FP16

    # Softmax in FP32 for stability
    attn = F.softmax(scores.float(), dim=-1).to(Q.dtype)

    # Attention * V in FP16
    return torch.matmul(attn, V)
```

### Loss Functions

Cross-entropy loss requires FP32:

```python
def cross_entropy_mixed_precision(logits: torch.Tensor,
                                   targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with FP32 log-softmax."""
    # Log-softmax in FP32 for numerical stability
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Gather and negate in FP32
    loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    return loss.mean()
```

## When Precision Reduction Fails

### Signs of Precision Problems

1. **Loss spikes**: Sudden increases in loss
2. **NaN/Inf gradients**: Overflow or invalid operations
3. **Training stalls**: Updates round to zero
4. **Accuracy degradation**: Final quality worse than FP32 baseline

### Debugging Precision Issues

```python
class PrecisionDebugger:
    """Tools for debugging mixed-precision issues."""

    @staticmethod
    def check_tensor_stats(tensor: torch.Tensor, name: str):
        """Print tensor statistics for debugging."""
        t = tensor.float()
        stats = {
            'min': t.min().item(),
            'max': t.max().item(),
            'mean': t.mean().item(),
            'std': t.std().item(),
            'num_zeros': (t == 0).sum().item(),
            'num_inf': torch.isinf(t).sum().item(),
            'num_nan': torch.isnan(t).sum().item(),
        }
        print(f"{name}: {stats}")

    @staticmethod
    def check_gradient_magnitudes(model: nn.Module):
        """Check gradient magnitudes across layers."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.float()
                print(f"{name}:")
                print(f"  grad mean: {grad.mean():.2e}")
                print(f"  grad std: {grad.std():.2e}")
                print(f"  grad max: {grad.abs().max():.2e}")
                print(f"  grad min nonzero: {grad[grad != 0].abs().min():.2e}")

    @staticmethod
    def compare_fp16_fp32(model_fp16, model_fp32, input_data):
        """Compare FP16 and FP32 model outputs."""
        with torch.no_grad():
            out_fp16 = model_fp16(input_data).float()
            out_fp32 = model_fp32(input_data.float())

            diff = (out_fp16 - out_fp32).abs()
            rel_diff = diff / (out_fp32.abs() + 1e-8)

            print(f"Absolute diff: mean={diff.mean():.2e}, max={diff.max():.2e}")
            print(f"Relative diff: mean={rel_diff.mean():.2e}, max={rel_diff.max():.2e}")
```

### Remediation Strategies

**1. Increase Loss Scale**

```python
# If gradients underflow, use larger initial scale
scaler = GradScaler(init_scale=2**20)  # 1M instead of 65536
```

**2. Keep Problem Layers in FP32**

```python
class HybridPrecisionModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Keep first and last layers in FP32
        self.model.embed.float()
        self.model.output_proj.float()

    def forward(self, x):
        # Embedding in FP32
        x = self.model.embed(x)

        # Main computation in FP16
        with autocast():
            x = self.model.transformer(x.half())

        # Output projection in FP32
        x = self.model.output_proj(x.float())
        return x
```

**3. Use BF16 Instead of FP16**

```python
# Switch from FP16 to BF16
with autocast(dtype=torch.bfloat16):  # Instead of float16
    output = model(data)
```

**4. Gradient Clipping Before Unscaling**

```python
# Clip in scaled space to prevent overflow during unscaling
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
```

## Communication and Precision

### AllReduce in Mixed Precision

Gradient reduction must be done carefully:

```python
def allreduce_gradients_mixed_precision(model: nn.Module,
                                         process_group):
    """AllReduce with precision-aware communication."""
    for param in model.parameters():
        if param.grad is None:
            continue

        # Option 1: Reduce in FP16 (fast, but accumulation error)
        # grad_fp16 = param.grad.half()
        # dist.all_reduce(grad_fp16, group=process_group)
        # param.grad.copy_(grad_fp16)

        # Option 2: Reduce in FP32 (accurate, more bandwidth)
        grad_fp32 = param.grad.float()
        dist.all_reduce(grad_fp32, group=process_group)
        param.grad.copy_(grad_fp32)
```

### Precision-Communication Trade-off

| Strategy | Bandwidth | Accuracy | Best For |
|----------|-----------|----------|----------|
| FP32 AllReduce | High | Best | Small models, research |
| FP16 AllReduce | Low | Good | Large models, production |
| BF16 AllReduce | Low | Good | Modern hardware (A100+) |
| FP16 + error feedback | Low | Better | Extreme scaling |

## Practical Recipe

### Recommended Mixed-Precision Setup

```python
def create_mixed_precision_trainer(model, optimizer, use_bf16=False):
    """
    Create a mixed-precision training configuration.
    """
    config = {
        'compute_dtype': torch.bfloat16 if use_bf16 else torch.float16,
        'master_weights': True,
        'loss_scaling': not use_bf16,  # BF16 often doesn't need scaling
    }

    # Wrap optimizer for master weights
    if config['master_weights']:
        optimizer = MasterWeightOptimizer(model, optimizer)

    # Create scaler if needed
    scaler = None
    if config['loss_scaling']:
        scaler = torch.cuda.amp.GradScaler()

    return MixedPrecisionTrainer(model, optimizer, scaler, config)

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, scaler, config):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config

    def train_step(self, data, target):
        self.optimizer.zero_grad()

        # Forward pass in reduced precision
        with torch.cuda.amp.autocast(dtype=self.config['compute_dtype']):
            output = self.model(data)
            loss = F.cross_entropy(output, target)

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return loss.item()
```

### Precision Selection Flowchart

```
                     Start
                       │
                       ▼
              ┌─────────────────┐
              │  H100 or newer? │
              └────────┬────────┘
                       │
           ┌───────────┴───────────┐
           │ Yes                   │ No
           ▼                       ▼
    ┌─────────────┐        ┌─────────────┐
    │  Use FP8    │        │ A100/A10G?  │
    │  (if model  │        └──────┬──────┘
    │   allows)   │               │
    └─────────────┘       ┌───────┴───────┐
                          │ Yes           │ No
                          ▼               ▼
                   ┌──────────┐    ┌──────────┐
                   │ Use BF16 │    │ V100?    │
                   └──────────┘    └────┬─────┘
                                        │
                                ┌───────┴───────┐
                                │ Yes           │ No
                                ▼               ▼
                         ┌──────────┐    ┌──────────┐
                         │ Use FP16 │    │ Use FP32 │
                         │ + scaling│    │(no accel)│
                         └──────────┘    └──────────┘
```

## Exercises

1. **Precision comparison**: Implement matrix multiplication in FP32, FP16, and BF16. For random 1000×1000 matrices, measure the maximum element-wise difference. How does error grow with matrix size?

??? success "Solution"
    **Implementation:**

    ```python
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    def compare_matmul_precision(n):
        """Compare matrix multiplication across precisions"""
        # Generate random matrices in FP32
        torch.manual_seed(42)
        A_fp32 = torch.randn(n, n, dtype=torch.float32)
        B_fp32 = torch.randn(n, n, dtype=torch.float32)

        # FP32 as ground truth
        C_fp32 = A_fp32 @ B_fp32

        # FP16
        A_fp16 = A_fp32.half()
        B_fp16 = B_fp32.half()
        C_fp16 = (A_fp16 @ B_fp16).float()

        # BF16
        A_bf16 = A_fp32.bfloat16()
        B_bf16 = B_fp32.bfloat16()
        C_bf16 = (A_bf16 @ B_bf16).float()

        # Compute errors
        error_fp16 = (C_fp32 - C_fp16).abs()
        error_bf16 = (C_fp32 - C_bf16).abs()

        return {
            'max_error_fp16': error_fp16.max().item(),
            'max_error_bf16': error_bf16.max().item(),
            'mean_error_fp16': error_fp16.mean().item(),
            'mean_error_bf16': error_bf16.mean().item(),
            'relative_error_fp16': (error_fp16 / C_fp32.abs().clamp(min=1e-8)).mean().item(),
            'relative_error_bf16': (error_bf16 / C_fp32.abs().clamp(min=1e-8)).mean().item(),
        }

    # Test across matrix sizes
    sizes = [100, 500, 1000, 2000, 4000]
    results = []

    for n in sizes:
        metrics = compare_matmul_precision(n)
        metrics['n'] = n
        results.append(metrics)
        print(f"n={n}: FP16 max_err={metrics['max_error_fp16']:.4f}, "
              f"BF16 max_err={metrics['max_error_bf16']:.4f}")
    ```

    **Expected results for n=1000:**

    | Metric | FP16 | BF16 |
    |--------|------|------|
    | Max absolute error | ~0.05-0.15 | ~0.5-1.5 |
    | Mean absolute error | ~0.005-0.01 | ~0.05-0.1 |
    | Relative error | ~0.01% | ~0.1% |

    **Error scaling with matrix size:**

    | Matrix Size (n) | FP16 Max Error | BF16 Max Error | Error Growth |
    |-----------------|----------------|----------------|--------------|
    | 100 | 0.015 | 0.15 | Baseline |
    | 500 | 0.08 | 0.8 | ~√5 |
    | 1000 | 0.12 | 1.2 | ~√10 |
    | 2000 | 0.18 | 1.8 | ~√20 |
    | 4000 | 0.25 | 2.5 | ~√40 |

    **Analysis:**

    Error grows approximately as $O(\sqrt{n})$ because:
    - Each output element is a sum of $n$ products
    - Rounding errors accumulate as random walk: $\sigma_{sum} = \sigma_{single} \cdot \sqrt{n}$

    **Key observations:**

    1. **BF16 has ~10× higher error than FP16**: Due to 3 fewer mantissa bits (7 vs 10)
    2. **Error scales as √n**: Consistent with random error accumulation
    3. **Relative error stays small**: ~0.01-0.1% for typical matrix sizes

    $$\boxed{\text{Max error } \propto \sqrt{n}, \text{ BF16 error} \approx 10 \times \text{FP16 error}}$$

2. **Loss scaling**: Artificially create a gradient that would underflow in FP16 (e.g., $10^{-6}$). Verify that without loss scaling the gradient becomes zero. Find the minimum loss scale that preserves the gradient.

??? success "Solution"
    **FP16 underflow threshold:**

    FP16 has:
    - Minimum positive normal: $2^{-14} \approx 6.1 \times 10^{-5}$
    - Minimum positive subnormal: $2^{-24} \approx 6.0 \times 10^{-8}$

    Values below the subnormal minimum become zero.

    **Demonstration:**

    ```python
    import torch

    def test_underflow():
        # Gradient values that progressively underflow
        test_values = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

        print("Without loss scaling:")
        for val in test_values:
            grad_fp32 = torch.tensor(val, dtype=torch.float32)
            grad_fp16 = grad_fp32.half()
            recovered = grad_fp16.float()
            print(f"  {val:.0e} -> FP16 -> {recovered.item():.2e} "
                  f"({'ZERO' if recovered == 0 else 'OK'})")

        print("\nWith loss scaling (scale=65536):")
        scale = 65536.0
        for val in test_values:
            grad_fp32 = torch.tensor(val * scale, dtype=torch.float32)
            grad_fp16 = grad_fp32.half()
            recovered = grad_fp16.float() / scale
            print(f"  {val:.0e} -> scaled -> FP16 -> unscaled -> {recovered.item():.2e}")

    test_underflow()
    ```

    **Expected output:**

    ```
    Without loss scaling:
      1e-04 -> FP16 -> 1.00e-04 (OK)
      1e-05 -> FP16 -> 1.00e-05 (OK)
      1e-06 -> FP16 -> 1.00e-06 (OK)
      1e-07 -> FP16 -> 5.96e-08 (OK - subnormal)
      1e-08 -> FP16 -> 0.00e+00 (ZERO)
      1e-09 -> FP16 -> 0.00e+00 (ZERO)

    With loss scaling (scale=65536):
      1e-04 -> scaled -> FP16 -> unscaled -> 1.00e-04
      1e-05 -> scaled -> FP16 -> unscaled -> 1.00e-05
      1e-06 -> scaled -> FP16 -> unscaled -> 1.00e-06
      1e-07 -> scaled -> FP16 -> unscaled -> 1.00e-07
      1e-08 -> scaled -> FP16 -> unscaled -> 1.00e-08
      1e-09 -> scaled -> FP16 -> unscaled -> 1.00e-09
    ```

    **Finding minimum loss scale:**

    ```python
    def find_minimum_scale(target_gradient):
        """Find minimum scale to preserve gradient in FP16"""
        min_fp16_normal = 2**-14  # ~6.1e-5

        # Need: target * scale >= min_fp16_normal
        min_scale = min_fp16_normal / target_gradient

        # Round up to power of 2 for numerical stability
        import math
        min_scale_pow2 = 2 ** math.ceil(math.log2(min_scale))

        return min_scale_pow2

    # For gradient = 1e-6
    target = 1e-6
    min_scale = find_minimum_scale(target)
    print(f"Minimum scale for {target}: {min_scale}")

    # Verify
    grad = torch.tensor(target * min_scale).half()
    recovered = grad.float() / min_scale
    print(f"Recovered: {recovered.item():.2e}")
    ```

    **Minimum scale calculation:**

    For gradient $g = 10^{-6}$:

    $$\text{scale}_{min} = \frac{2^{-14}}{10^{-6}} = \frac{6.1 \times 10^{-5}}{10^{-6}} = 61$$

    Rounded to power of 2: $\text{scale}_{min} = 64 = 2^6$

    **Summary table:**

    | Gradient Value | Min FP16 Normal | Minimum Scale | Rounded (2^n) |
    |----------------|-----------------|---------------|---------------|
    | $10^{-5}$ | $6.1 \times 10^{-5}$ | 6.1 | 8 |
    | $10^{-6}$ | $6.1 \times 10^{-5}$ | 61 | 64 |
    | $10^{-7}$ | $6.1 \times 10^{-5}$ | 610 | 1024 |
    | $10^{-8}$ | $6.1 \times 10^{-5}$ | 6100 | 8192 |

    $$\boxed{\text{For } g = 10^{-6}: \text{minimum scale} = 64}$$

3. **Dynamic range**: Create a tensor with values uniformly distributed in $[10^{-8}, 10^8]$. What fraction of values are lost when converting to FP16? To BF16?

??? success "Solution"
    **Format dynamic ranges:**

    | Format | Exponent Bits | Min Normal | Max Normal | Total Range |
    |--------|---------------|------------|------------|-------------|
    | FP16 | 5 | $\sim 6 \times 10^{-5}$ | $\sim 6.5 \times 10^4$ | $\sim 10^{10}$ |
    | BF16 | 8 | $\sim 10^{-38}$ | $\sim 10^{38}$ | $\sim 10^{76}$ |
    | FP32 | 8 | $\sim 10^{-38}$ | $\sim 10^{38}$ | $\sim 10^{76}$ |

    **Implementation:**

    ```python
    import torch
    import numpy as np

    def analyze_dynamic_range_loss():
        # Create log-uniform distribution from 10^-8 to 10^8
        n = 1_000_000
        log_min, log_max = -8, 8
        log_values = np.random.uniform(log_min, log_max, n)
        values_fp32 = torch.tensor(10 ** log_values, dtype=torch.float32)

        # Convert to FP16 and back
        values_fp16 = values_fp32.half().float()

        # Convert to BF16 and back
        values_bf16 = values_fp32.bfloat16().float()

        # Count zeros (underflow) and infs (overflow)
        fp16_underflow = (values_fp16 == 0) & (values_fp32 != 0)
        fp16_overflow = torch.isinf(values_fp16) & ~torch.isinf(values_fp32)
        fp16_lost = fp16_underflow | fp16_overflow

        bf16_underflow = (values_bf16 == 0) & (values_fp32 != 0)
        bf16_overflow = torch.isinf(values_bf16) & ~torch.isinf(values_fp32)
        bf16_lost = bf16_underflow | bf16_overflow

        return {
            'fp16_underflow_frac': fp16_underflow.float().mean().item(),
            'fp16_overflow_frac': fp16_overflow.float().mean().item(),
            'fp16_total_lost': fp16_lost.float().mean().item(),
            'bf16_underflow_frac': bf16_underflow.float().mean().item(),
            'bf16_overflow_frac': bf16_overflow.float().mean().item(),
            'bf16_total_lost': bf16_lost.float().mean().item(),
        }

    results = analyze_dynamic_range_loss()
    print(f"FP16 - Underflow: {results['fp16_underflow_frac']:.2%}, "
          f"Overflow: {results['fp16_overflow_frac']:.2%}")
    print(f"BF16 - Underflow: {results['bf16_underflow_frac']:.2%}, "
          f"Overflow: {results['bf16_overflow_frac']:.2%}")
    ```

    **Theoretical analysis:**

    For uniform distribution in $[\log_{10}(10^{-8}), \log_{10}(10^8)] = [-8, 8]$:

    **FP16:**
    - Underflow: Values below $6 \times 10^{-5}$ (log ≈ -4.2)
    - Overflow: Values above $6.5 \times 10^4$ (log ≈ 4.8)
    - Range covered: $[-4.2, 4.8]$ out of $[-8, 8]$

    $$\text{FP16 lost} = \frac{(-4.2 - (-8)) + (8 - 4.8)}{16} = \frac{3.8 + 3.2}{16} = \frac{7.0}{16} = 43.75\%$$

    **BF16:**
    - Underflow: Values below $\sim 10^{-38}$ → none in our range
    - Overflow: Values above $\sim 10^{38}$ → none in our range
    - Range covered: All of $[-8, 8]$

    $$\text{BF16 lost} = 0\%$$

    **Expected results:**

    | Format | Underflow | Overflow | Total Lost |
    |--------|-----------|----------|------------|
    | FP16 | ~24% | ~20% | **~44%** |
    | BF16 | ~0% | ~0% | **~0%** |

    **Visualization of representable ranges:**

    ```
    FP16:        [-----xxxxxxxxx|---------xxxxxxxxx]
                 -8    -4.2     0        4.8      8
                   underflow              overflow

    BF16:        [--------------------------------]
                 -8            0                 8
                        all representable
    ```

    $$\boxed{\text{FP16 loses } \sim44\% \text{ of values; BF16 loses } \sim0\%}$$

4. **Softmax stability**: Implement softmax without the max-subtraction trick. Find the smallest input value that causes overflow in FP16. Verify the stable version works.

??? success "Solution"
    **Unstable softmax implementation:**

    ```python
    import torch
    import numpy as np

    def softmax_unstable(x):
        """Naive softmax - numerically unstable."""
        exp_x = torch.exp(x)
        return exp_x / exp_x.sum(dim=-1, keepdim=True)

    def softmax_stable(x):
        """Stable softmax with max subtraction."""
        x_max = x.max(dim=-1, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        return exp_x / exp_x.sum(dim=-1, keepdim=True)

    # Find overflow threshold in FP16
    def find_overflow_threshold():
        """Binary search for smallest value causing overflow."""
        low, high = 0.0, 100.0

        while high - low > 0.01:
            mid = (low + high) / 2
            x = torch.tensor([mid], dtype=torch.float16)
            exp_x = torch.exp(x)

            if torch.isinf(exp_x).any():
                high = mid
            else:
                low = mid

        return low

    threshold = find_overflow_threshold()
    print(f"FP16 overflow threshold for exp(): {threshold:.2f}")
    # Expected: ~11.09 (since exp(11.09) ≈ 65504 = FP16 max)
    ```

    **Theoretical analysis:**

    FP16 max representable value: $65504$

    Overflow occurs when: $e^x > 65504$

    $$x > \ln(65504) \approx 11.09$$

    **Verification:**

    ```python
    # Test both implementations
    def compare_implementations():
        results = []

        for max_val in [5, 10, 11, 12, 15, 20, 50, 100]:
            # Create logits with specified maximum
            x_fp16 = torch.randn(1000, dtype=torch.float16)
            x_fp16 = x_fp16 * (max_val / x_fp16.abs().max())

            # Compute with both methods
            try:
                unstable = softmax_unstable(x_fp16)
                unstable_nan = torch.isnan(unstable).any().item()
                unstable_inf = torch.isinf(unstable).any().item()
            except:
                unstable_nan, unstable_inf = True, True

            stable = softmax_stable(x_fp16)
            stable_nan = torch.isnan(stable).any().item()
            stable_inf = torch.isinf(stable).any().item()

            # Ground truth in FP32
            x_fp32 = x_fp16.float()
            gt = torch.softmax(x_fp32, dim=-1)

            # Error (if stable succeeded)
            if not (stable_nan or stable_inf):
                error = (stable.float() - gt).abs().max().item()
            else:
                error = float('inf')

            results.append({
                'max_val': max_val,
                'unstable_ok': not (unstable_nan or unstable_inf),
                'stable_ok': not (stable_nan or stable_inf),
                'error': error
            })

        return results

    results = compare_implementations()
    for r in results:
        print(f"max={r['max_val']:3d}: unstable={'OK' if r['unstable_ok'] else 'FAIL':4s}, "
              f"stable={'OK' if r['stable_ok'] else 'FAIL':4s}, error={r['error']:.2e}")
    ```

    **Expected output:**

    ```
    FP16 overflow threshold for exp(): 11.09

    max=  5: unstable=OK  , stable=OK  , error=9.77e-04
    max= 10: unstable=OK  , stable=OK  , error=9.77e-04
    max= 11: unstable=OK  , stable=OK  , error=9.77e-04
    max= 12: unstable=FAIL, stable=OK  , error=9.77e-04
    max= 15: unstable=FAIL, stable=OK  , error=9.77e-04
    max= 20: unstable=FAIL, stable=OK  , error=9.77e-04
    max= 50: unstable=FAIL, stable=OK  , error=9.77e-04
    max=100: unstable=FAIL, stable=OK  , error=9.77e-04
    ```

    **Why max-subtraction works:**

    After subtracting max: $x_i - x_{max} \leq 0$ for all $i$

    Therefore: $e^{x_i - x_{max}} \leq 1$ — never overflows!

    The mathematical equivalence:

    $$\frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - x_{max}}}{\sum_j e^{x_j - x_{max}}}$$

    | Input range | Unstable | Stable |
    |-------------|----------|--------|
    | $[-5, 5]$ | ✓ Works | ✓ Works |
    | $[-10, 10]$ | ✓ Works | ✓ Works |
    | $[-15, 15]$ | ✗ Overflow/NaN | ✓ Works |
    | $[-100, 100]$ | ✗ Overflow/NaN | ✓ Works |

    $$\boxed{\text{FP16 overflows at } x \geq 11.09; \text{ max-subtraction fixes it}}$$

5. **Mixed-precision speedup**: Benchmark a transformer layer in FP32 vs FP16 on your GPU. What's the speedup? What fraction comes from compute vs memory bandwidth?

??? success "Solution"
    **Benchmarking implementation:**

    ```python
    import torch
    import torch.nn as nn
    import time

    class TransformerLayer(nn.Module):
        def __init__(self, hidden_dim, num_heads, dtype=torch.float32):
            super().__init__()
            self.dtype = dtype
            self.hidden_dim = hidden_dim

            # Self-attention
            self.q_proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
            self.o_proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)

            # MLP
            self.fc1 = nn.Linear(hidden_dim, 4 * hidden_dim, dtype=dtype)
            self.fc2 = nn.Linear(4 * hidden_dim, hidden_dim, dtype=dtype)

            # Norms
            self.norm1 = nn.LayerNorm(hidden_dim, dtype=dtype)
            self.norm2 = nn.LayerNorm(hidden_dim, dtype=dtype)

            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads

        def forward(self, x):
            # Self-attention
            residual = x
            x = self.norm1(x)

            B, S, H = x.shape
            q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

            attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)

            out = out.transpose(1, 2).contiguous().view(B, S, H)
            out = self.o_proj(out)
            x = residual + out

            # MLP
            residual = x
            x = self.norm2(x)
            x = self.fc1(x)
            x = torch.gelu(x)
            x = self.fc2(x)
            x = residual + x

            return x

    def benchmark_layer(hidden_dim, num_heads, batch_size, seq_len, dtype, num_iters=100):
        """Benchmark forward + backward pass."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        layer = TransformerLayer(hidden_dim, num_heads, dtype=dtype).to(device)
        x = torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype, device=device)

        # Warmup
        for _ in range(10):
            out = layer(x)
            loss = out.sum()
            loss.backward()

        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(num_iters):
            out = layer(x)
            loss = out.sum()
            loss.backward()
        torch.cuda.synchronize()
        elapsed = time.time() - start

        # Memory moved per iteration (approximate)
        params = sum(p.numel() for p in layer.parameters())
        bytes_per_elem = 4 if dtype == torch.float32 else 2
        param_bytes = params * bytes_per_elem * 3  # params + grads + opt read

        # Activations (rough estimate)
        act_bytes = batch_size * seq_len * hidden_dim * bytes_per_elem * 10

        total_bytes = (param_bytes + act_bytes) * num_iters

        return {
            'dtype': str(dtype).split('.')[-1],
            'time_per_iter_ms': (elapsed / num_iters) * 1000,
            'total_bytes_gb': total_bytes / 1e9,
            'bandwidth_gbps': total_bytes / elapsed / 1e9
        }

    # Run benchmarks
    configs = [
        {'hidden_dim': 4096, 'num_heads': 32, 'batch_size': 8, 'seq_len': 2048},
    ]

    for cfg in configs:
        print(f"\nConfig: hidden={cfg['hidden_dim']}, batch={cfg['batch_size']}, "
              f"seq={cfg['seq_len']}")

        fp32_result = benchmark_layer(**cfg, dtype=torch.float32)
        fp16_result = benchmark_layer(**cfg, dtype=torch.float16)

        speedup = fp32_result['time_per_iter_ms'] / fp16_result['time_per_iter_ms']

        print(f"  FP32: {fp32_result['time_per_iter_ms']:.2f} ms/iter")
        print(f"  FP16: {fp16_result['time_per_iter_ms']:.2f} ms/iter")
        print(f"  Speedup: {speedup:.2f}x")
    ```

    **Expected results on H100:**

    | Metric | FP32 | FP16 | Ratio |
    |--------|------|------|-------|
    | Time (ms) | ~12.5 | ~3.8 | 3.3× faster |
    | Memory BW used | ~1.8 TB/s | ~1.7 TB/s | ~same |
    | Compute TFLOP/s | ~150 | ~500 | 3.3× higher |

    **Decomposing the speedup:**

    ```python
    def decompose_speedup():
        """
        Speedup sources:
        1. Memory bandwidth: 2× (half the bytes)
        2. Tensor Core compute: 4× (FP16 vs FP32)
        3. Cache efficiency: 2× (more fits in cache)
        """

        # Estimate time breakdown for FP32
        # Typical transformer: 60% compute-bound, 40% memory-bound
        compute_frac_fp32 = 0.60
        memory_frac_fp32 = 0.40

        # FP16 speedups
        compute_speedup = 4.0  # Tensor Cores: FP8 1979 vs FP16 495 TFLOP/s
        memory_speedup = 2.0  # Half the bytes

        # Weighted speedup
        # T_fp16 = T_fp32 * (compute_frac / compute_speedup + memory_frac / memory_speedup)
        time_ratio = (compute_frac_fp32 / compute_speedup +
                      memory_frac_fp32 / memory_speedup)
        theoretical_speedup = 1 / time_ratio

        print(f"Compute contribution to speedup: {compute_speedup:.1f}x (weight: {compute_frac_fp32:.0%})")
        print(f"Memory contribution to speedup: {memory_speedup:.1f}x (weight: {memory_frac_fp32:.0%})")
        print(f"Theoretical combined speedup: {theoretical_speedup:.2f}x")

        # Reality check
        print(f"\nTypical observed speedup: 2.5-4.0x")
        print(f"Gap explained by: kernel overhead, non-matmul ops, memory access patterns")

        return theoretical_speedup

    decompose_speedup()
    ```

    **Analysis:**

    For a compute-bound workload (large batch, long sequence):
    - FP16 Tensor Core speedup: 4×
    - Actual observed: 3-3.5×

    For a memory-bound workload (small batch):
    - Memory bandwidth benefit: 2×
    - Actual observed: 1.8-2.2×

    **Speedup breakdown:**

    | Component | Speedup Factor | Contribution |
    |-----------|---------------|--------------|
    | Tensor Core (matmul) | 4× | ~60% of benefit |
    | Memory bandwidth | 2× | ~30% of benefit |
    | Cache efficiency | 1.5× | ~10% of benefit |

    $$\boxed{\text{Typical speedup: 2.5-4×; } \sim60\% \text{ from compute, } \sim40\% \text{ from memory}}$$

6. **FP8 quantization**: Implement per-tensor quantization to FP8 E4M3. For a pre-trained model, compare accuracy degradation vs FP16/BF16.

??? success "Solution"
    **FP8 E4M3 quantization implementation:**

    ```python
    import torch
    import torch.nn as nn
    import numpy as np

    class FP8Quantizer:
        """
        FP8 E4M3 format:
        - 1 sign bit
        - 4 exponent bits (bias = 7)
        - 3 mantissa bits
        - Range: ±448, precision: ~1/16
        """

        # FP8 E4M3 constants
        E4M3_MAX = 448.0
        E4M3_MIN = 2**-9  # Smallest normal number

        @staticmethod
        def quantize_to_fp8(tensor, scale=None):
            """Quantize FP32/FP16 tensor to FP8 E4M3 (simulated)."""
            if scale is None:
                # Compute per-tensor scale
                amax = tensor.abs().max()
                scale = FP8Quantizer.E4M3_MAX / amax if amax > 0 else 1.0

            # Scale to FP8 range
            scaled = tensor * scale

            # Clamp to FP8 range
            clamped = torch.clamp(scaled, -FP8Quantizer.E4M3_MAX, FP8Quantizer.E4M3_MAX)

            # Simulate FP8 precision (round to 3 mantissa bits)
            # This is an approximation - real FP8 has more complex rounding
            sign = torch.sign(clamped)
            abs_val = torch.abs(clamped)

            # Find exponent and mantissa
            log2_val = torch.log2(abs_val.clamp(min=FP8Quantizer.E4M3_MIN))
            exponent = torch.floor(log2_val)
            mantissa = abs_val / (2.0 ** exponent) - 1.0  # Normalized mantissa [0, 1)

            # Round mantissa to 3 bits (8 levels)
            mantissa_quantized = torch.round(mantissa * 8) / 8

            # Reconstruct
            quantized = sign * (1.0 + mantissa_quantized) * (2.0 ** exponent)

            # Handle zeros and denormals
            quantized = torch.where(abs_val < FP8Quantizer.E4M3_MIN,
                                   torch.zeros_like(quantized), quantized)

            return quantized, scale

        @staticmethod
        def dequantize_from_fp8(quantized, scale):
            """Dequantize from FP8 back to original range."""
            return quantized / scale

    def compare_precision_formats(model, test_loader, device='cuda'):
        """Compare accuracy across precision formats."""
        model = model.to(device)
        model.eval()

        results = {}

        # Test FP32 (baseline)
        correct_fp32 = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct_fp32 += predicted.eq(labels).sum().item()
                total += labels.size(0)
        results['fp32'] = 100 * correct_fp32 / total

        # Test FP16
        model_fp16 = model.half()
        correct_fp16 = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.half().to(device), labels.to(device)
                outputs = model_fp16(images)
                _, predicted = outputs.max(1)
                correct_fp16 += predicted.eq(labels).sum().item()
        results['fp16'] = 100 * correct_fp16 / total

        # Test BF16
        model_bf16 = model.to(torch.bfloat16)
        correct_bf16 = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(torch.bfloat16).to(device)
                labels = labels.to(device)
                outputs = model_bf16(images)
                _, predicted = outputs.max(1)
                correct_bf16 += predicted.eq(labels).sum().item()
        results['bf16'] = 100 * correct_bf16 / total

        # Test FP8 (simulated)
        model.float()  # Back to FP32
        correct_fp8 = 0

        # Quantize all weights to FP8
        original_weights = {}
        for name, param in model.named_parameters():
            original_weights[name] = param.data.clone()
            quantized, scale = FP8Quantizer.quantize_to_fp8(param.data)
            param.data = FP8Quantizer.dequantize_from_fp8(quantized, scale)

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct_fp8 += predicted.eq(labels).sum().item()
        results['fp8'] = 100 * correct_fp8 / total

        # Restore original weights
        for name, param in model.named_parameters():
            param.data = original_weights[name]

        return results

    # Example usage with a pre-trained model
    def run_comparison():
        import torchvision.models as models
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets
        from torch.utils.data import DataLoader

        # Load pre-trained ResNet-18
        model = models.resnet18(pretrained=True)

        # Prepare ImageNet validation set (or CIFAR-10 for quick test)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Use CIFAR-10 for demonstration
        test_dataset = datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

        results = compare_precision_formats(model, test_loader)

        print("\nAccuracy comparison:")
        print("-" * 40)
        for fmt, acc in results.items():
            degradation = results['fp32'] - acc
            print(f"{fmt.upper():6s}: {acc:.2f}% (degradation: {degradation:+.2f}%)")

        return results

    results = run_comparison()
    ```

    **Expected results (ResNet-18 on ImageNet):**

    | Format | Accuracy | Degradation |
    |--------|----------|-------------|
    | FP32 | 69.76% | baseline |
    | FP16 | 69.74% | -0.02% |
    | BF16 | 69.71% | -0.05% |
    | FP8 E4M3 | 69.12% | -0.64% |
    | FP8 E5M2 | 68.85% | -0.91% |

    **Per-tensor vs per-channel scaling:**

    ```python
    def compare_scaling_strategies(tensor):
        """Compare per-tensor vs per-channel FP8 quantization."""

        # Per-tensor scaling
        quantized_tensor, scale_tensor = FP8Quantizer.quantize_to_fp8(tensor)
        dequant_tensor = FP8Quantizer.dequantize_from_fp8(quantized_tensor, scale_tensor)
        error_tensor = (tensor - dequant_tensor).abs().mean().item()

        # Per-channel scaling (for conv weights: scale per output channel)
        if tensor.dim() >= 2:
            scales = []
            quantized_channels = []
            for i in range(tensor.shape[0]):
                q, s = FP8Quantizer.quantize_to_fp8(tensor[i])
                scales.append(s)
                quantized_channels.append(q)

            dequant_channel = torch.stack([
                FP8Quantizer.dequantize_from_fp8(q, s)
                for q, s in zip(quantized_channels, scales)
            ])
            error_channel = (tensor - dequant_channel).abs().mean().item()
        else:
            error_channel = error_tensor

        print(f"Per-tensor quantization error: {error_tensor:.6f}")
        print(f"Per-channel quantization error: {error_channel:.6f}")
        print(f"Improvement: {error_tensor / error_channel:.2f}x")

        return error_tensor, error_channel

    # Test with sample weight tensor
    sample_weights = torch.randn(64, 64, 3, 3)  # Conv layer weights
    compare_scaling_strategies(sample_weights)
    ```

    **Analysis:**

    | Scaling Strategy | Quantization Error | Storage Overhead |
    |-----------------|-------------------|------------------|
    | Per-tensor | Higher | 1 scale value |
    | Per-channel | 2-4× lower | O(channels) scales |
    | Per-token (activations) | Lowest | O(batch × seq) scales |

    **Key observations:**

    1. **FP16/BF16**: Near-lossless for inference (<0.1% degradation)
    2. **FP8 E4M3**: Small degradation (~0.5-1%) acceptable for most applications
    3. **Per-channel scaling**: Essential for FP8 to minimize error
    4. **Sensitive layers**: First/last layers often kept in higher precision

    $$\boxed{\text{FP8 degrades accuracy by } \sim0.5\text{-}1\%; \text{ per-channel scaling reduces this by } 2\text{-}4\times}$$

## Key Takeaways

1. **Three formats dominate**: FP32 for storage/accumulation, FP16/BF16 for compute, FP8 emerging.

2. **BF16 > FP16 for training**: Same range as FP32 eliminates most loss scaling needs.

3. **Master weights essential**: Small updates vanish in FP16; accumulate in FP32.

4. **Loss scaling prevents underflow**: Dynamic scaling adapts to gradient magnitudes.

5. **Not all operations are equal**: Softmax, LayerNorm, and loss need FP32; matmul can use FP16.

6. **2× memory, 8× compute**: Reduced precision helps both bandwidth and Tensor Core throughput.

7. **FP8 requires per-tensor scaling**: Limited range needs careful scale management.

8. **Debug methodically**: Check gradient magnitudes, compare to FP32 baseline, identify problem layers.
