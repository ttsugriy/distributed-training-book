---
title: "Data Parallelism from Associativity"
subtitle: "Why Gradient Accumulation Enables Distribution"
---

<div class="chapter-opener" markdown>
Data parallelism works because gradient accumulation is associative. This isn't an implementation detail—it's the mathematical foundation that makes the entire approach valid. Understanding this foundation reveals both the power and the limits of data parallelism.
</div>

<div class="investigation-question" markdown>
**The Question**: We compute gradients on different batches on different GPUs, then sum them. Why does this give us the same result as computing on the full batch? When does it fail?
</div>

!!! abstract "Building On: Part III Collectives"
    This part assumes mastery of **collective operations**—especially AllReduce (Chapter 11), the **ring and tree algorithms** (Chapter 12), and **cost modeling** (Chapter 13). Each parallelism strategy we derive here will rely on specific collectives. Data parallelism, our first strategy, uses AllReduce to synchronize gradients.

## The Mathematical Foundation

### The Loss Function Structure

Consider a loss function over a dataset $\mathcal{D}$:

$$L(\theta) = \frac{1}{|\mathcal{D}|} \sum_{x \in \mathcal{D}} \ell(x, \theta)$$

where $\ell(x, \theta)$ is the per-sample loss.

In minibatch SGD, we approximate this with a batch $B$:

$$L_B(\theta) = \frac{1}{|B|} \sum_{x \in B} \ell(x, \theta)$$

The gradient is:

$$\nabla_\theta L_B(\theta) = \frac{1}{|B|} \sum_{x \in B} \nabla_\theta \ell(x, \theta)$$

### The Partitioning Theorem

**Theorem**: For a batch $B$ partitioned into $P$ disjoint subsets $B = B_1 \cup B_2 \cup \cdots \cup B_P$ with $|B_i| = |B|/P$ for all $i$:

$$\nabla_\theta L_B(\theta) = \frac{1}{P} \sum_{i=1}^{P} \nabla_\theta L_{B_i}(\theta)$$

**Proof**:

Starting from the definition:

$$\nabla_\theta L_B(\theta) = \frac{1}{|B|} \sum_{x \in B} \nabla_\theta \ell(x, \theta)$$

Since $B = \bigcup_{i=1}^{P} B_i$ with disjoint $B_i$:

$$= \frac{1}{|B|} \sum_{i=1}^{P} \sum_{x \in B_i} \nabla_\theta \ell(x, \theta)$$

With $|B_i| = |B|/P$:

$$= \frac{1}{|B|} \sum_{i=1}^{P} |B_i| \cdot \frac{1}{|B_i|} \sum_{x \in B_i} \nabla_\theta \ell(x, \theta)$$

$$= \frac{1}{|B|} \sum_{i=1}^{P} |B_i| \cdot \nabla_\theta L_{B_i}(\theta)$$

$$= \frac{1}{|B|} \sum_{i=1}^{P} \frac{|B|}{P} \cdot \nabla_\theta L_{B_i}(\theta)$$

$$= \frac{1}{P} \sum_{i=1}^{P} \nabla_\theta L_{B_i}(\theta) \quad \square$$

This partitioning is valid because:

1. **Addition is associative**: $(a + b) + c = a + (b + c)$
2. **Addition is commutative**: $a + b = b + a$
3. **Gradients are elements of a vector space**: They inherit these properties from real numbers

### The Key Insight

The gradient of a sum equals the sum of gradients:

$$\nabla_\theta \left( \sum_i f_i(\theta) \right) = \sum_i \nabla_\theta f_i(\theta)$$

This **linearity of the gradient operator** combined with **associativity of addition** is what makes data parallelism mathematically sound.

## The Synchronous Data Parallel Algorithm

### Basic Algorithm

```
Input: Model θ, Dataset D, P GPUs, learning rate η
Output: Trained model θ

1. Broadcast θ to all P GPUs
2. For each training step:
   a. Sample batch B of size |B| = P × b
   b. Partition: Bi = samples [(i-1)b : ib] for GPU i
   c. On each GPU i in parallel:

      - Forward: yi = f(Bi; θ)
      - Loss: Li = L(yi, targets)
      - Backward: gi = ∇θLi
   d. AllReduce: g = (1/P) Σi gi
   e. Update: θ ← θ - η·g
   f. (Implicit) All GPUs now have identical θ
3. Return θ
```

### Correctness Invariant

**Invariant**: At the start of each step, all GPUs hold identical model parameters.

**Proof by induction**:

*Base case*: Step 0 broadcasts θ, so all GPUs start identical.

*Inductive step*: Assume all GPUs have identical θ at step $t$. After AllReduce, all GPUs have identical gradient $g$. Applying the same update rule with the same learning rate:

$$\theta^{(t+1)} = \theta^{(t)} - \eta \cdot g$$

produces identical $\theta^{(t+1)}$ on all GPUs. $\square$

This invariant is why we don't need to broadcast parameters every step—the identical update maintains synchronization.

### Pseudocode Implementation

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train_data_parallel(model, dataloader, optimizer, epochs):
    """Synchronous data parallel training."""

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Wrap model in DDP (handles gradient synchronization)
    model = DDP(model, device_ids=[rank])

    for epoch in range(epochs):
        # Distributed sampler ensures non-overlapping batches
        dataloader.sampler.set_epoch(epoch)

        for batch in dataloader:
            # Forward pass (identical on each GPU with different data)
            loss = model(batch)

            # Backward pass (computes local gradients)
            loss.backward()

            # DDP automatically performs AllReduce here
            # Gradients are averaged across all ranks

            # Update (identical update on all GPUs)
            optimizer.step()
            optimizer.zero_grad()
```

## Communication Analysis

### Per-Step Communication Volume

Let $\Psi$ be the number of model parameters (in elements, not bytes).

**AllReduce volume**: Each parameter's gradient must be synchronized.

Using ring AllReduce:

$$V = 2 \cdot \frac{P-1}{P} \cdot \Psi \cdot \text{sizeof}(\text{dtype})$$

For 32-bit floats:

$$V = 8 \cdot \frac{P-1}{P} \cdot \Psi \text{ bytes}$$

As $P \to \infty$:

$$V \to 8\Psi \text{ bytes}$$

### Communication Time

Using the α-β model with ring AllReduce:

$$T_{\text{comm}} = 2(P-1) \cdot \alpha + 2 \cdot \frac{P-1}{P} \cdot \frac{\Psi \cdot \text{sizeof}(\text{dtype})}{\beta}$$

For large $P$ and large $\Psi$ (bandwidth-dominated):

$$T_{\text{comm}} \approx \frac{2\Psi \cdot \text{sizeof}(\text{dtype})}{\beta}$$

### Compute Time

Per GPU, with local batch size $b$:

$$T_{\text{compute}} = \frac{6\Psi \cdot b}{F}$$

where $F$ is the GPU's FLOP/s.

### Compute-Communication Ratio

The critical ratio:

$$R = \frac{T_{\text{compute}}}{T_{\text{comm}}} = \frac{6\Psi \cdot b / F}{2\Psi \cdot \text{sizeof}(\text{dtype}) / \beta}$$

Simplifying:

$$R = \frac{3b \cdot \beta}{F \cdot \text{sizeof}(\text{dtype})}$$

For FP16 training:

$$R = \frac{3b \cdot \beta}{2F}$$

**Example**: H100 with $F = 2 \times 10^{15}$ FLOP/s, NVLink $\beta = 900$ GB/s:

$$R = \frac{3b \cdot 900 \times 10^9}{2 \times 2 \times 10^{15}} = \frac{3b \cdot 900}{4 \times 10^6} = \frac{675b}{10^6}$$

For $R > 1$ (compute-bound), need:

$$b > \frac{10^6}{675} \approx 1,500 \text{ samples}$$

### Scaling Efficiency

Define scaling efficiency:

$$E(P) = \frac{T_1}{P \cdot T_P}$$

where $T_1$ is single-GPU time and $T_P$ is $P$-GPU time per step.

With perfect overlap:

$$T_P = \max(T_{\text{compute}}, T_{\text{comm}})$$

**Compute-bound regime** ($T_{\text{compute}} > T_{\text{comm}}$):

$$E(P) = \frac{T_{\text{compute}}}{P \cdot T_{\text{compute}}} = \frac{1}{P} \cdot P = 1$$

Perfect scaling! But this assumes:
1. Perfect overlap of communication and computation
2. No additional overhead

**Communication-bound regime** ($T_{\text{comm}} > T_{\text{compute}}$):

$$E(P) = \frac{T_{\text{compute}}}{P \cdot T_{\text{comm}}} = \frac{R}{P}$$

Efficiency degrades as $P$ increases.

## Overlapping Communication and Computation

### The Overlap Opportunity

Backward pass computes gradients layer by layer, from output to input. Once a layer's gradient is computed, it can be communicated while computing earlier layers' gradients.

```
Time →
Layer 4: [backward][  AllReduce  ]
Layer 3:          [backward][  AllReduce  ]
Layer 2:                   [backward][  AllReduce  ]
Layer 1:                            [backward][  AllReduce  ]
                                             ↑
                                    Overlap region
```

### Bucket-Based Overlap

PyTorch DDP groups gradients into buckets:

```python
# DDP bucket configuration
model = DDP(
    model,
    device_ids=[rank],
    bucket_cap_mb=25,  # Bucket size in MB
)
```

**Bucketing algorithm**:
1. Register gradients in reverse computation order
2. When bucket reaches capacity, trigger AllReduce
3. Continue computing while AllReduce proceeds

**Optimal bucket size**:

- Too small: Many small AllReduces (latency overhead)
- Too large: Less overlap opportunity

Empirically: 10-50 MB buckets work well.

### Gradient Ready Order

DDP uses hooks to detect when gradients are ready:

```python
class DDPGradientHook:
    def __init__(self, bucket_manager):
        self.bucket_manager = bucket_manager

    def __call__(self, grad):
        # Called when gradient is computed
        self.bucket_manager.add_gradient(grad)
        if self.bucket_manager.bucket_ready():
            # Launch async AllReduce
            self.bucket_manager.flush_bucket()
        return grad
```

The communication for layer $L$ overlaps with backward computation of layers $1, ..., L-1$.

### Theoretical Overlap Efficiency

Let $f_i$ be the fraction of compute time for layer $i$'s backward pass.

Achievable overlap:

$$\text{Overlap} = \sum_{i=1}^{L-1} f_i \cdot \min\left(1, \frac{T_{\text{comm},i}}{T_{\text{compute},<i}}\right)$$

where $T_{\text{compute},<i}$ is compute time for layers before $i$.

In practice, 70-90% overlap is achievable.

## When Associativity Fails

### Floating-Point Non-Associativity

IEEE 754 floating-point addition is **not associative**:

$$(a + b) + c \neq a + (b + c)$$

**Example**:
```python
import numpy as np

a = np.float32(1e20)
b = np.float32(-1e20)
c = np.float32(1.0)

print((a + b) + c)  # = 1.0
print(a + (b + c))  # = 0.0  (c absorbed into b)
```

### Reduction Order Matters

Different AllReduce implementations use different reduction orders:

**Ring AllReduce**:

$$g = (\cdots((g_0 + g_1) + g_2) + \cdots + g_{P-1})$$

**Tree AllReduce** (binary tree):

$$g = ((g_0 + g_1) + (g_2 + g_3)) + ((g_4 + g_5) + (g_6 + g_7))$$

These give slightly different results for the same inputs!

### Sources of Non-Determinism

| Source | Cause | Mitigation |
|--------|-------|------------|
| AllReduce order | Ring direction varies | Fix ring order |
| Tree reduction | Different groupings | Use consistent tree |
| Async completion | Race conditions | Synchronous reduction |
| Fused operations | Different algorithms | Disable fusion |
| Hardware variations | Different GPUs | Homogeneous cluster |

### Achieving Determinism

```python
import torch

def enable_deterministic_training():
    """Enable deterministic training for reproducibility."""

    # Set seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # Disable auto-tuning (chooses different algorithms)
    torch.backends.cudnn.benchmark = False

    # Set deterministic NCCL reduction order
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ["NCCL_PROTO"] = "Simple"
```

**Cost of determinism**: Often 10-20% slower due to disabled optimizations.

### Higher Precision Accumulation

For gradient accumulation, use higher precision:

```python
class HighPrecisionAccumulator:
    """Accumulate gradients in FP32 even if model is FP16."""

    def __init__(self, model):
        self.fp32_grads = {
            name: torch.zeros_like(p, dtype=torch.float32)
            for name, p in model.named_parameters()
        }

    def accumulate(self, model):
        for name, p in model.named_parameters():
            if p.grad is not None:
                self.fp32_grads[name] += p.grad.float()

    def get_and_reset(self, model):
        for name, p in model.named_parameters():
            p.grad = self.fp32_grads[name].to(p.dtype)
            self.fp32_grads[name].zero_()
```

## Gradient Compression

### The Compression Opportunity

Gradients are high-dimensional but often compressible. We can trade computation for communication bandwidth.

### Top-K Sparsification

Keep only the $k$ largest magnitude gradients:

$$\text{TopK}(g, k) = \{g_i : |g_i| \geq |g|_{(k)}\}$$

where $|g|_{(k)}$ is the $k$-th largest magnitude.

**Compression ratio**: $k/d$ where $d$ is gradient dimension.

**Error feedback**: Accumulate dropped gradients for next iteration:

```python
class TopKCompressor:
    def __init__(self, model, k_ratio=0.01):
        self.k_ratio = k_ratio
        self.error_buffer = {
            name: torch.zeros_like(p)
            for name, p in model.named_parameters()
        }

    def compress(self, name, grad):
        # Add error from previous round
        grad = grad + self.error_buffer[name]

        # Select top-k
        k = max(1, int(self.k_ratio * grad.numel()))
        values, indices = torch.topk(grad.abs().flatten(), k)

        # Compute error (dropped values)
        mask = torch.zeros_like(grad.flatten())
        mask[indices] = 1
        self.error_buffer[name] = grad * (1 - mask.view_as(grad))

        # Return compressed gradient
        return indices, grad.flatten()[indices]

    def decompress(self, indices, values, shape):
        grad = torch.zeros(shape.numel())
        grad[indices] = values
        return grad.view(shape)
```

### Random Sparsification

Randomly sample gradients instead of top-k (faster, but higher variance):

$$\tilde{g}_i = \frac{1}{p} g_i \cdot \mathbb{1}[\text{sample}_i < p]$$

where $p$ is the sampling probability.

**Unbiased**: $\mathbb{E}[\tilde{g}] = g$

### Quantization

Reduce precision of gradients:

**1-bit SGD**: Sign of gradient only:

$$\tilde{g}_i = \text{sign}(g_i) \cdot \|g\|_1 / d$$

**TernGrad**: Three values $\{-1, 0, +1\}$

**QSGD**: Stochastic quantization to $s$ levels:

$$Q_s(g_i) = \|g\|_2 \cdot \text{sign}(g_i) \cdot \xi_i(s)$$

where $\xi_i(s)$ is a stochastic quantization function.

### PowerSGD

Low-rank approximation of gradient matrix:

$$G \approx PQ^T$$

where $P \in \mathbb{R}^{m \times r}$, $Q \in \mathbb{R}^{n \times r}$, and $r \ll \min(m, n)$.

**Algorithm**:
1. Project gradient onto low-rank subspace
2. Communicate low-rank factors (much smaller)
3. Reconstruct approximate gradient

**Compression ratio**: $(m + n) \cdot r / (m \cdot n)$

For $m = n = 1000$, $r = 4$: compression ratio = 0.8%

## Gradient Accumulation

### When Memory Limits Batch Size

If local batch $b$ doesn't fit in memory, accumulate over micro-batches:

```python
def train_with_accumulation(model, batch, accumulation_steps, optimizer):
    """Gradient accumulation for effective large batches."""

    optimizer.zero_grad()

    # Split batch into micro-batches
    micro_batches = batch.chunk(accumulation_steps)

    for micro_batch in micro_batches:
        # Forward and backward (gradients accumulate)
        loss = model(micro_batch) / accumulation_steps
        loss.backward()

    # Now perform AllReduce and update
    # (DDP would AllReduce here automatically)
    optimizer.step()
```

### Mathematical Equivalence

With accumulation over $A$ micro-batches of size $b/A$:

$$g_{\text{accum}} = \frac{1}{A} \sum_{j=1}^{A} \nabla L_{B_j}$$

This equals $\nabla L_B$ by the partitioning theorem.

### Communication Pattern

Without accumulation: AllReduce every micro-batch
With accumulation: AllReduce every $A$ micro-batches

**Speedup from reduced communication**:

$$\frac{T_{\text{no-accum}}}{T_{\text{accum}}} = \frac{A \cdot T_{\text{compute/A}} + A \cdot T_{\text{comm}}}{A \cdot T_{\text{compute/A}} + T_{\text{comm}}} = \frac{A \cdot T_c + A \cdot T_r}{A \cdot T_c + T_r}$$

For $T_r = T_c$: speedup = $(2A)/(A+1)$ → 2× as $A \to \infty$.

## Scaling Behavior

### Weak Scaling

Keep local batch size $b$ constant, add more GPUs:

- Total batch size: $B = P \cdot b$
- Compute per GPU: constant
- Communication: increases with $P$ (latency term)

**Efficiency**:

$$E_{\text{weak}}(P) = \frac{T_1}{T_P} = \frac{T_c}{T_c + T_r(P)}$$

where $T_r(P) = 2(P-1)\alpha + 2\frac{P-1}{P} \cdot \Psi/\beta$.

### Strong Scaling

Keep total batch size $B$ constant, add more GPUs:

- Local batch size: $b = B/P$
- Compute per GPU: decreases as $1/P$
- Communication: approximately constant (bandwidth term dominates)

**Efficiency**:

$$E_{\text{strong}}(P) = \frac{T_1}{P \cdot T_P} = \frac{T_c(1)}{P \cdot (T_c(1)/P + T_r)}$$

As $P$ increases, $T_c/P$ becomes small compared to $T_r$, and efficiency drops.

### Practical Limits

| Model Size | Typical P Limit (90% efficiency) |
|------------|----------------------------------|
| 1B params | ~256 GPUs |
| 10B params | ~128 GPUs |
| 100B params | ~64 GPUs |
| 1T params | ~16 GPUs |

Beyond these limits, data parallelism alone becomes communication-bound.

## Asynchronous Data Parallelism

### The Synchronization Bottleneck

Synchronous DP waits for the slowest worker ("straggler problem"):

$$T_{\text{step}} = \max_i T_i + T_{\text{AllReduce}}$$

Variance in $T_i$ reduces efficiency.

### Asynchronous SGD

Workers don't wait for each other:

```
Worker 1: [compute][push grad][pull params][compute]...
Worker 2: [compute][push][pull][compute][push]...
Worker 3: [compute][push][pull][compute]...
          ← Staleness →
```

**Staleness**: Worker applies update to params $\theta^{(t-\tau)}$ where $\tau$ is delay.

### Staleness-Adjusted Updates

To compensate for staleness:

$$\theta^{(t+1)} = \theta^{(t)} - \eta \cdot f(\tau) \cdot g^{(t-\tau)}$$

where $f(\tau)$ is a staleness penalty:

- $f(\tau) = 1$: ignore staleness (often diverges)
- $f(\tau) = 1/\tau$: inverse scaling
- $f(\tau) = e^{-\lambda\tau}$: exponential decay

### Trade-offs

| Aspect | Synchronous | Asynchronous |
|--------|-------------|--------------|
| Correctness | Exact | Approximate |
| Staleness | 0 | Variable |
| Straggler handling | Poor | Good |
| Convergence | Faster per step | More steps needed |
| Implementation | Simple | Complex |
| Debugging | Easy | Hard |

**Trend**: Synchronous is dominant in modern large-scale training due to simpler convergence guarantees.

## Implementation: PyTorch DDP

### Architecture

```
DDP Module
    ├── Forward Hook: Sync forward (no-op for most cases)
    ├── Backward Hooks: Register gradient ready callbacks
    ├── Bucket Manager: Group gradients for AllReduce
    ├── Reducer: Execute AllReduce operations
    └── Comm Hook: Customizable communication (compression, etc.)
```

### Custom Communication Hooks

```python
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default

def compression_hook(state, bucket):
    """Custom hook for gradient compression."""
    tensor = bucket.buffer()

    # Compress
    compressed, ctx = my_compressor.compress(tensor)

    # AllReduce compressed gradient
    fut = dist.all_reduce(compressed, async_op=True).get_future()

    def decompress(fut):
        synced = fut.value()[0]
        return my_compressor.decompress(synced, ctx)

    return fut.then(decompress)

# Register hook
model = DDP(model)
model.register_comm_hook(state=None, hook=compression_hook)
```

### Gradient Synchronization Points

DDP synchronizes at these points:
1. **First forward** (if `find_unused_parameters=True`)
2. **After backward** (main gradient sync)

```python
# For models with unused parameters
model = DDP(
    model,
    device_ids=[rank],
    find_unused_parameters=True,  # Extra overhead
)
```

## Exercises

1. **Partitioning proof**: Extend the partitioning theorem to unequal partition sizes. If $|B_i| = w_i \cdot |B|$ where $\sum_i w_i = 1$, what is the correct averaging formula?

??? success "Solution"
    **Extended Partitioning Theorem for Unequal Sizes**

    Given batch $B$ partitioned into $P$ disjoint subsets $B_1, B_2, \ldots, B_P$ where $|B_i| = w_i \cdot |B|$ and $\sum_{i=1}^P w_i = 1$.

    **Claim:**
    $$\nabla_\theta L_B(\theta) = \sum_{i=1}^{P} w_i \cdot \nabla_\theta L_{B_i}(\theta)$$

    **Proof:**

    Starting from the definition:

    $$\nabla_\theta L_B(\theta) = \frac{1}{|B|} \sum_{x \in B} \nabla_\theta \ell(x, \theta)$$

    Since $B = \bigcup_{i=1}^{P} B_i$ with disjoint $B_i$:

    $$= \frac{1}{|B|} \sum_{i=1}^{P} \sum_{x \in B_i} \nabla_\theta \ell(x, \theta)$$

    Multiplying and dividing by $|B_i|$:

    $$= \frac{1}{|B|} \sum_{i=1}^{P} |B_i| \cdot \frac{1}{|B_i|} \sum_{x \in B_i} \nabla_\theta \ell(x, \theta)$$

    $$= \frac{1}{|B|} \sum_{i=1}^{P} |B_i| \cdot \nabla_\theta L_{B_i}(\theta)$$

    Substituting $|B_i| = w_i \cdot |B|$:

    $$= \frac{1}{|B|} \sum_{i=1}^{P} w_i \cdot |B| \cdot \nabla_\theta L_{B_i}(\theta)$$

    $$= \boxed{\sum_{i=1}^{P} w_i \cdot \nabla_\theta L_{B_i}(\theta)} \quad \square$$

    **Practical implication:**

    When GPUs have unequal batch sizes (e.g., due to load balancing), the AllReduce should compute a **weighted average**, not a simple average:

    ```python
    # Weighted gradient averaging
    local_batch_size = len(local_batch)
    total_batch_size = dist.all_reduce(
        torch.tensor(local_batch_size), op=dist.ReduceOp.SUM
    )
    weight = local_batch_size / total_batch_size

    for param in model.parameters():
        param.grad *= weight
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    ```

    **Special case:** When all $w_i = 1/P$, we recover the original theorem with simple averaging.

2. **Compute-communication balance**: For a 7B parameter model on 8 H100s with NVLink (900 GB/s), what local batch size achieves $R = 2$ (compute 2× communication)?

??? success "Solution"
    **Given:**

    - $\Psi = 7 \times 10^9$ parameters
    - $P = 8$ GPUs
    - $\beta = 900$ GB/s (NVLink bandwidth)
    - $F = 989 \times 10^{12}$ FLOP/s (H100 dense FP16/BF16 peak)
    - FP16 training: sizeof(dtype) = 2 bytes
    - Target: $R = 2$

    **Compute-Communication Ratio Formula (from chapter):**

    $$R = \frac{3b \cdot \beta}{F \cdot \text{sizeof}(\text{dtype})}$$

    **Solving for batch size $b$:**

    $$b = \frac{R \cdot F \cdot \text{sizeof}(\text{dtype})}{3 \cdot \beta}$$

    $$b = \frac{2 \times 989 \times 10^{12} \times 2}{3 \times 900 \times 10^9}$$

    $$b = \frac{3.956 \times 10^{15}}{2.7 \times 10^{12}}$$

    $$b = \boxed{1,465 \text{ samples}}$$

    **Verification:**

    *Communication time:*
    $$T_{\text{comm}} = 2 \times \frac{P-1}{P} \times \frac{\Psi \times 2}{\beta}$$

    $$= 2 \times \frac{7}{8} \times \frac{7 \times 10^9 \times 2}{900 \times 10^9}$$

    $$= 1.75 \times 15.56 \text{ ms} = 27.2 \text{ ms}$$

    *Compute time:*
    $$T_{\text{compute}} = \frac{6\Psi \times b}{F}$$

    $$= \frac{6 \times 7 \times 10^9 \times 1465}{989 \times 10^{12}}$$

    $$= \frac{6.16 \times 10^{13}}{9.89 \times 10^{14}} = 62.2 \text{ ms}$$

    *Ratio check:*
    $$R = \frac{62.2}{27.2} = 2.29 \approx 2 \checkmark$$

    **Practical considerations:**

    | Sequence Length | Samples (b=2932) | Tokens per GPU |
    |-----------------|------------------|----------------|
    | 512 | 2,932 | 1.5M |
    | 2048 | 2,932 | 6.0M |
    | 4096 | 2,932 | 12.0M |

    With 4096 sequence length, this requires ~12M tokens per GPU per step—likely too large for memory. Gradient accumulation would be needed.

3. **Overlap analysis**: A model has 100 layers of equal size. If AllReduce for all gradients takes 100ms and backward for all layers takes 80ms, what fraction of communication can be overlapped? What is the effective step time?

??? success "Solution"
    **Given:**

    - $L = 100$ layers of equal size
    - $T_{\text{AR}}^{\text{total}} = 100$ ms (total AllReduce time)
    - $T_{\text{bwd}}^{\text{total}} = 80$ ms (total backward time)
    - All layers equal → per-layer times are uniform

    **Per-layer times:**

    $$T_{\text{AR}}^{\text{layer}} = \frac{100}{100} = 1 \text{ ms}$$

    $$T_{\text{bwd}}^{\text{layer}} = \frac{80}{100} = 0.8 \text{ ms}$$

    **Overlap analysis:**

    With bucket-based overlap, layer $i$'s AllReduce can overlap with layers $1, \ldots, i-1$'s backward computation.

    ```
    Layer 100: [bwd 0.8ms][  AR 1ms  ]
    Layer 99:            [bwd 0.8ms][  AR 1ms  ]
    Layer 98:                      [bwd 0.8ms][  AR 1ms  ]
    ...
    ```

    **For each layer except the first:**
    - AllReduce (1 ms) can potentially overlap with backward of earlier layers

    Since backward is faster than AllReduce (0.8 ms < 1 ms), AllReduce is the bottleneck.

    **Critical path analysis:**

    - Layer 100: starts at t=0, backward finishes at t=0.8ms, AR finishes at t=1.8ms
    - Layer 99: backward can start at t=0.8ms (after layer 100 backward), finishes at t=1.6ms
      - But AR for layer 100 is still running until t=1.8ms
      - Layer 99 AR starts at t=1.8ms, finishes at t=2.8ms
    - This continues with AR being the bottleneck

    **Actually, the overlap is limited by the relative speeds.**

    Let's compute more carefully:

    *Total backward time without overlap:* 80 ms

    *Total AR time without overlap:* 100 ms

    *With perfect pipelining:*
    - First layer backward: 0.8 ms
    - Then 99 AllReduces in sequence: 99 × 1 ms = 99 ms (partially overlapped)
    - Last AllReduce finishes after last backward

    The overlappable portion is the backward time of layers that can run concurrently with AllReduce:

    $$\text{Overlap time} = T_{\text{bwd}}^{\text{total}} - T_{\text{bwd}}^{\text{first layer}} = 80 - 0.8 = 79.2 \text{ ms}$$

    Since total AR time is 100 ms and backward (after first layer) is 79.2 ms:

    $$\text{Overlapped communication} = \min(79.2, 100) = 79.2 \text{ ms}$$

    **Fraction overlapped:**
    $$\text{Overlap fraction} = \frac{79.2}{100} = \boxed{79.2\%}$$

    **Effective step time:**

    $$T_{\text{step}} = T_{\text{bwd}}^{\text{first layer}} + \max(T_{\text{bwd}}^{\text{remaining}}, T_{\text{AR}}^{\text{total}})$$

    Since AR (100 ms) > remaining backward (79.2 ms):

    $$T_{\text{step}} = 0.8 + 100 = \boxed{100.8 \text{ ms}}$$

    **Comparison:**

    | Scenario | Step Time |
    |----------|-----------|
    | No overlap | 80 + 100 = 180 ms |
    | With overlap | 100.8 ms |
    | Speedup | 1.79× |

    **Key insight:** When AR is slower than backward, the step time approaches the AR time. The system is communication-bound.

4. **Compression analysis**: Top-1% sparsification compresses gradients 100×. If the original AllReduce takes 50ms, what is the new time? Account for compression/decompression compute (assume 5ms each).

??? success "Solution"
    **Given:**

    - Original AllReduce time: $T_{\text{AR}}^{\text{orig}} = 50$ ms
    - Compression ratio: 100× (Top-1% sparsification)
    - Compression compute: $T_{\text{compress}} = 5$ ms
    - Decompression compute: $T_{\text{decompress}} = 5$ ms

    **Analysis:**

    *Original communication is bandwidth-dominated (large gradients):*
    $$T_{\text{AR}}^{\text{orig}} \approx \frac{2(P-1)}{P} \cdot \frac{n}{\beta} \approx \frac{2n}{\beta} = 50 \text{ ms}$$

    *Compressed communication:*
    $$T_{\text{AR}}^{\text{compressed}} \approx \frac{2n/100}{\beta} = \frac{50}{100} = 0.5 \text{ ms}$$

    **However, sparse AllReduce is more complex:**

    Sparse tensors require AllGather of indices + values, not standard AllReduce:

    - Indices: $0.01n$ elements × 4 bytes (int32) = $0.04n$ bytes
    - Values: $0.01n$ elements × 2 bytes (FP16) = $0.02n$ bytes
    - Total: $0.06n$ bytes (vs original $2n$ bytes)

    Effective compression: $2n / 0.06n = 33×$ (not 100×!)

    *Revised compressed communication:*
    $$T_{\text{AR}}^{\text{compressed}} = \frac{50}{33} \approx 1.5 \text{ ms}$$

    **Total time with compression:**

    $$T_{\text{total}} = T_{\text{compress}} + T_{\text{AR}}^{\text{compressed}} + T_{\text{decompress}}$$

    $$= 5 + 1.5 + 5 = \boxed{11.5 \text{ ms}}$$

    **Speedup:**
    $$\text{Speedup} = \frac{T_{\text{orig}}}{T_{\text{total}}} = \frac{50}{11.5} = \boxed{4.3\times}$$

    **Summary:**

    | Component | Time |
    |-----------|------|
    | Compression | 5 ms |
    | Sparse AllReduce | 1.5 ms |
    | Decompression | 5 ms |
    | **Total** | **11.5 ms** |

    **Key insights:**

    1. **Compression overhead matters**: 10 ms of compute overhead is significant
    2. **Index overhead reduces effective compression**: 100× sparsity ≠ 100× bandwidth reduction
    3. **Still valuable**: 4.3× speedup is substantial for communication-bound training
    4. **Break-even analysis**: Compression only helps if $T_{\text{compress}} + T_{\text{decompress}} < T_{\text{AR}}^{\text{orig}} - T_{\text{AR}}^{\text{compressed}}$

    $$10 < 50 - 1.5 = 48.5 \text{ ms} \checkmark$$

5. **Accumulation trade-off**: Training with local batch 32 takes 100ms compute and 40ms AllReduce per micro-batch. If you accumulate over 4 micro-batches, what is the time per effective step? What if memory allowed batch 128 directly (single forward-backward)?

??? success "Solution"
    **Given:**

    - Micro-batch size: 32
    - $T_{\text{compute}}^{\text{micro}} = 100$ ms per micro-batch
    - $T_{\text{AR}} = 40$ ms per AllReduce
    - Accumulation steps: $A = 4$
    - Effective batch size: $32 \times 4 = 128$

    **Case 1: Gradient Accumulation (4 micro-batches)**

    With accumulation, we compute 4 micro-batches but AllReduce only once:

    $$T_{\text{accum}} = A \times T_{\text{compute}}^{\text{micro}} + T_{\text{AR}}$$

    $$= 4 \times 100 + 40 = \boxed{440 \text{ ms}}$$

    **Case 2: Direct Batch 128 (no accumulation)**

    Compute time scales linearly with batch size:

    $$T_{\text{compute}}^{\text{direct}} = \frac{128}{32} \times 100 = 400 \text{ ms}$$

    Total:

    $$T_{\text{direct}} = T_{\text{compute}}^{\text{direct}} + T_{\text{AR}} = 400 + 40 = \boxed{440 \text{ ms}}$$

    **Comparison:**

    | Method | Compute | AllReduce | Total |
    |--------|---------|-----------|-------|
    | Accumulation (4×32) | 4 × 100 = 400 ms | 40 ms | 440 ms |
    | Direct (128) | 400 ms | 40 ms | 440 ms |

    $$\boxed{\text{Both methods take the same time: 440 ms}}$$

    **Why are they equal?**

    Gradient accumulation doesn't add overhead—it just splits the compute into multiple passes. The AllReduce only happens once in both cases.

    **When does direct batching win?**

    If there's overlap between compute and communication:

    *With overlap (direct):*
    $$T_{\text{direct}}^{\text{overlap}} = \max(T_{\text{compute}}, T_{\text{AR}}) = \max(400, 40) = 400 \text{ ms}$$

    *With overlap (accumulation):*
    - Can't overlap micro-batch 1-3 AR (we don't do AR)
    - Can only overlap final AR with... nothing (compute is done)
    $$T_{\text{accum}}^{\text{overlap}} = 4 \times 100 + 40 = 440 \text{ ms}$$

    **Speedup from direct batching with overlap:**
    $$\frac{440}{400} = 1.1\times$$

    **Key insight:** Direct large batches enable better compute-communication overlap than gradient accumulation.

6. **Staleness bound**: In asynchronous SGD with $P = 16$ workers and exponential staleness penalty $f(\tau) = e^{-0.1\tau}$, if maximum staleness is $\tau = 15$, what is the effective learning rate scaling?

??? success "Solution"
    **Given:**

    - $P = 16$ workers
    - Staleness penalty: $f(\tau) = e^{-0.1\tau}$
    - Maximum staleness: $\tau_{\max} = 15$

    **Staleness-adjusted update rule:**

    $$\theta^{(t+1)} = \theta^{(t)} - \eta \cdot f(\tau) \cdot g^{(t-\tau)}$$

    The effective learning rate is: $\eta_{\text{eff}} = \eta \cdot f(\tau)$

    **Staleness distribution analysis:**

    In asynchronous SGD with $P$ workers, staleness $\tau$ ranges from 0 to $P-1$ in the worst case (when one worker is always ahead).

    With $\tau_{\max} = 15$ (close to $P - 1 = 15$), we have maximum asynchrony.

    **Effective learning rate at different staleness levels:**

    | Staleness $\tau$ | $f(\tau) = e^{-0.1\tau}$ | $\eta_{\text{eff}}/\eta$ |
    |------------------|--------------------------|--------------------------|
    | 0 | $e^0 = 1.000$ | 100% |
    | 5 | $e^{-0.5} = 0.607$ | 60.7% |
    | 10 | $e^{-1.0} = 0.368$ | 36.8% |
    | 15 | $e^{-1.5} = 0.223$ | 22.3% |

    **At maximum staleness ($\tau = 15$):**

    $$\eta_{\text{eff}} = \eta \cdot e^{-0.1 \times 15} = \eta \cdot e^{-1.5} = \boxed{0.223 \cdot \eta}$$

    The learning rate is scaled down to **22.3% of the base rate**.

    **Average effective learning rate:**

    Assuming uniform staleness distribution from 0 to $\tau_{\max}$:

    $$\mathbb{E}[f(\tau)] = \frac{1}{\tau_{\max} + 1} \sum_{\tau=0}^{\tau_{\max}} e^{-0.1\tau}$$

    This is a geometric series:

    $$= \frac{1}{16} \cdot \frac{1 - e^{-0.1 \times 16}}{1 - e^{-0.1}} = \frac{1}{16} \cdot \frac{1 - e^{-1.6}}{1 - e^{-0.1}}$$

    $$= \frac{1}{16} \cdot \frac{1 - 0.202}{1 - 0.905} = \frac{1}{16} \cdot \frac{0.798}{0.095} = \frac{8.4}{16} = 0.525$$

    **Average effective learning rate:** $\boxed{52.5\%}$ of base rate.

    **Practical implications:**

    | Aspect | Value |
    |--------|-------|
    | Minimum scaling (fresh gradients) | 100% |
    | Maximum scaling (stalest gradients) | 22.3% |
    | Average scaling | 52.5% |
    | Effective throughput loss | ~47.5% |

    To match synchronous SGD convergence speed, you would need to increase the base learning rate by approximately $1/0.525 \approx 1.9\times$, but this may cause instability with fresh gradients.

7. **Non-determinism quantification**: Two reduction orders $((g_0 + g_1) + g_2)$ and $(g_0 + (g_1 + g_2))$ differ by at most $\epsilon_{\text{machine}} \cdot |g_0 + g_1 + g_2|$. For a 1B parameter model with FP16 gradients ($\epsilon \approx 10^{-3}$), estimate the maximum gradient difference.

??? success "Solution"
    **Given:**

    - Model parameters: $\Psi = 1 \times 10^9$
    - Data type: FP16
    - Machine epsilon: $\epsilon \approx 10^{-3}$ (actually $\epsilon_{\text{FP16}} = 2^{-10} \approx 9.77 \times 10^{-4}$)
    - Error bound per reduction: $\epsilon \cdot |g_0 + g_1 + g_2|$

    **Single reduction error:**

    For a single floating-point addition of two numbers $a$ and $b$:

    $$\text{fl}(a + b) = (a + b)(1 + \delta), \quad |\delta| \leq \epsilon$$

    The absolute error is:

    $$|error| \leq \epsilon \cdot |a + b|$$

    **Error accumulation in AllReduce:**

    For $P$ GPUs using ring AllReduce, we perform $P-1$ additions per parameter:

    $$g = ((((g_0 + g_1) + g_2) + g_3) + \cdots + g_{P-1})$$

    Each step accumulates error. Using standard error analysis:

    $$|\text{accumulated error}| \leq (P-1) \cdot \epsilon \cdot |g_{\text{sum}}|$$

    **Per-parameter error estimate:**

    Assuming typical gradient magnitude $|g_i| \approx 10^{-4}$ (common for normalized training):

    - Sum of $P = 8$ gradients: $|g_{\text{sum}}| \approx 8 \times 10^{-4}$ (if aligned)
    - Per-parameter error: $\leq 7 \times 10^{-3} \times 8 \times 10^{-4} = 5.6 \times 10^{-6}$

    **Different reduction orders:**

    Two orderings can differ by at most:

    $$|\text{difference}| \leq 2 \cdot (P-1) \cdot \epsilon \cdot |g_{\text{sum}}|$$

    (Factor of 2 because each ordering can err in opposite directions)

    **Total model-wide error:**

    For 1B parameters, the aggregate gradient difference:

    *L2 norm of differences:*

    If errors are independent across parameters with variance $\sigma^2 = \epsilon^2 \cdot |g|^2$:

    $$\|g_{\text{order1}} - g_{\text{order2}}\|_2 \approx \sqrt{\Psi} \cdot \sigma$$

    $$= \sqrt{10^9} \cdot \epsilon \cdot |g_{\text{avg}}| \cdot (P-1)$$

    With $|g_{\text{avg}}| \approx 10^{-4}$, $P = 8$, $\epsilon = 10^{-3}$:

    $$= 3.16 \times 10^4 \cdot 10^{-3} \cdot 10^{-4} \cdot 7 = \boxed{2.2 \times 10^{-2}}$$

    **Per-parameter maximum difference:**

    $$|\Delta g_i|_{\max} \approx 2 \cdot (P-1) \cdot \epsilon \cdot |g_i| \approx 14 \cdot 10^{-3} \cdot 10^{-4} = \boxed{1.4 \times 10^{-6}}$$

    **Relative error:**

    $$\frac{|\Delta g|}{|g|} \approx 2(P-1) \cdot \epsilon = 14 \times 10^{-3} = \boxed{1.4\%}$$

    **Summary:**

    | Metric | Value |
    |--------|-------|
    | Per-parameter max difference | $1.4 \times 10^{-6}$ |
    | Relative error per parameter | ~1.4% |
    | L2 norm of difference (1B params) | ~0.022 |

    **Practical implications:**

    1. **Training stability**: 1.4% per-step variance accumulates over millions of steps
    2. **Reproducibility**: Same code, different GPU count → different results
    3. **Debugging**: Bitwise comparison across runs is impossible

    **Mitigation strategies:**

    | Strategy | Overhead | Determinism |
    |----------|----------|-------------|
    | Fixed reduction order | ~5% | Bit-exact per config |
    | FP32 accumulation | 2× memory | Reduces error ~1000× |
    | Kahan summation | 2× compute | Near FP64 accuracy |
    | Tree reduction (balanced) | Same | More stable than ring |

## Key Takeaways

1. **Associativity enables data parallelism**: The partitioning theorem is the mathematical foundation—gradient of sum equals sum of gradients.

2. **AllReduce is the communication primitive**: $2(P-1)/P \cdot \Psi$ bytes per step.

3. **Compute-communication ratio determines efficiency**: Need local batch large enough to amortize communication.

4. **Overlap is critical**: Bucket-based AllReduce during backward pass achieves 70-90% overlap.

5. **Floating-point breaks exact associativity**: Determinism requires fixed reduction order and disabled optimizations.

6. **Gradient compression trades compute for bandwidth**: Top-K, quantization, and low-rank methods can achieve 10-100× compression.

7. **Scaling has limits**: Beyond ~100 GPUs for large models, data parallelism alone becomes communication-bound.

8. **Synchronous dominates**: Simpler correctness and convergence make synchronous preferred despite straggler sensitivity.
