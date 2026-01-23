---
title: "Tensor Parallelism from Linearity"
subtitle: "Why Matrix Multiplication Shards and GELU Doesn't"
---

<div class="chapter-opener" markdown>
Matrix multiplication is linear: $f(aX) = af(X)$ and $f(X + Y) = f(X) + f(Y)$. This single property enables tensor parallelism. Non-linear operations like GELU break this property, forcing synchronization. Understanding linearity reveals which operations can be parallelized and which force communication.
</div>

<div class="investigation-question" markdown>
**The Question**: We want to shard a linear layer $Y = XW$ across 8 GPUs. Can we split $W$ column-wise? Row-wise? What about the bias? What about LayerNorm? What about GELU?
</div>

## The Linearity Property

### Definition

A function $f: V \to W$ between vector spaces is **linear** if for all vectors $X, Y \in V$ and scalars $a, b$:

$$f(aX + bY) = af(X) + bf(Y)$$

This is equivalent to two conditions:

1. **Additivity**: $f(X + Y) = f(X) + f(Y)$
2. **Homogeneity**: $f(aX) = af(X)$

### Why Linearity Enables Parallelism

If $f$ is linear and we partition the input $X = X_1 + X_2$:

$$f(X) = f(X_1 + X_2) = f(X_1) + f(X_2)$$

We can compute $f(X_1)$ and $f(X_2)$ independently, then sum the results.

For matrix multiplication $f(X) = XW$:

$$f(X_1 + X_2) = (X_1 + X_2)W = X_1W + X_2W = f(X_1) + f(X_2)$$

This **decomposition** is the foundation of tensor parallelism.

### The Classification of Operations

| Operation | Linear? | Parallelizable? |
|-----------|---------|-----------------|
| Matrix multiply | Yes | Yes (with structure) |
| Bias addition | Affine (linear + translation) | Special handling |
| ReLU | No | Element-wise only |
| GELU | No | Element-wise only |
| Softmax | No | Requires full tensor |
| LayerNorm | No | Requires statistics |
| Dropout | No (stochastic) | Element-wise with care |

## Column-Parallel Linear Layers

### The Idea

For a linear layer $Y = XW + b$ with $W \in \mathbb{R}^{d_{in} \times d_{out}}$:

Split $W$ along columns (output dimension):

$$W = [W_1 | W_2 | \cdots | W_P]$$

where each $W_i \in \mathbb{R}^{d_{in} \times (d_{out}/P)}$.

### The Computation

Each GPU $i$ holds:
- Full input $X$ (replicated)
- Shard $W_i$ of weights
- Shard $b_i$ of bias

Computes:
$$Y_i = XW_i + b_i$$

The results are column-partitioned:
$$Y = [Y_1 | Y_2 | \cdots | Y_P]$$

### Why It Works

Matrix multiplication distributes over column concatenation:

$$X[W_1 | W_2] = [XW_1 | XW_2]$$

**Proof**:

Let $X \in \mathbb{R}^{m \times n}$, $W_1 \in \mathbb{R}^{n \times k_1}$, $W_2 \in \mathbb{R}^{n \times k_2}$.

The $(i, j)$ entry of $XW_1$ for $j \leq k_1$:
$$(XW_1)_{ij} = \sum_{\ell=1}^{n} X_{i\ell} (W_1)_{\ell j}$$

The $(i, j)$ entry of $X[W_1 | W_2]$ for $j \leq k_1$:
$$(X[W_1 | W_2])_{ij} = \sum_{\ell=1}^{n} X_{i\ell} [W_1 | W_2]_{\ell j} = \sum_{\ell=1}^{n} X_{i\ell} (W_1)_{\ell j}$$

Identical. $\square$

### Communication

**Forward pass**: No communication needed. Each GPU computes independently.

**Backward pass**: Gradient w.r.t. $X$ requires AllReduce:
$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T = \sum_{i=1}^{P} \frac{\partial L}{\partial Y_i} W_i^T$$

### Diagram

```
        GPU 0           GPU 1           GPU 2           GPU 3

Input X ─────────────────────────────────────────────────────
   │        │               │               │               │
   │ Replicated             │               │               │
   ▼        ▼               ▼               ▼               ▼
 ┌────┐   ┌────┐         ┌────┐         ┌────┐         ┌────┐
 │ W₀ │   │ W₁ │         │ W₂ │         │ W₃ │         (weights)
 └────┘   └────┘         └────┘         └────┘
   │        │               │               │
   ▼        ▼               ▼               ▼
  Y₀       Y₁              Y₂              Y₃          (outputs)
   │        │               │               │
   └────────┴───────────────┴───────────────┘
            Column-partitioned output
```

## Row-Parallel Linear Layers

### The Idea

Split $W$ along rows (input dimension):

$$W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_P \end{bmatrix}$$

where each $W_i \in \mathbb{R}^{(d_{in}/P) \times d_{out}}$.

### The Computation

Requires input split along columns:
$$X = [X_1 | X_2 | \cdots | X_P]$$

Each GPU $i$ holds:
- Shard $X_i$ of input
- Shard $W_i$ of weights

Computes partial result:
$$Z_i = X_i W_i$$

Note: $X_i \in \mathbb{R}^{m \times (d_{in}/P)}$ and $W_i \in \mathbb{R}^{(d_{in}/P) \times d_{out}}$, so $Z_i \in \mathbb{R}^{m \times d_{out}}$.

### Why AllReduce Is Needed

The full result requires summing:

$$Y = XW = \sum_{i=1}^{P} X_i W_i = \sum_{i=1}^{P} Z_i$$

Each GPU has a partial sum $Z_i$. AllReduce computes $\sum_i Z_i$ on all GPUs.

**Proof**:

For partitioned multiplication:
$$XW = [X_1 | X_2 | \cdots | X_P] \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_P \end{bmatrix}$$

The $(i, j)$ entry:
$$(XW)_{ij} = \sum_{k=1}^{d_{in}} X_{ik} W_{kj} = \sum_{p=1}^{P} \sum_{k \in \text{shard } p} X_{ik} W_{kj}$$

$$= \sum_{p=1}^{P} (X_p W_p)_{ij} \quad \square$$

### Diagram

```
        GPU 0           GPU 1           GPU 2           GPU 3

Input X ─────────────────────────────────────────────────────
   │        │               │               │               │
   │ Column-partitioned     │               │               │
   ▼        ▼               ▼               ▼               ▼
  X₀       X₁              X₂              X₃          (input shards)
   │        │               │               │
   ▼        ▼               ▼               ▼
 ┌────┐   ┌────┐         ┌────┐         ┌────┐
 │ W₀ │   │ W₁ │         │ W₂ │         │ W₃ │         (weights)
 └────┘   └────┘         └────┘         └────┘
   │        │               │               │
   ▼        ▼               ▼               ▼
  Z₀       Z₁              Z₂              Z₃          (partial sums)
   │        │               │               │
   └────────┴───────────────┴───────────────┘
                      │
                      ▼ AllReduce (sum)
                      │
                      ▼
                 Y (replicated)
```

### Bias Handling

For row-parallel with bias $Y = XW + b$:
- Add bias **after** AllReduce (on the full result)
- Or: add $b/P$ on each GPU before AllReduce (works due to sum)

## The Megatron-LM Pattern

Shoeybi et al. (2019) introduced an elegant pattern combining column and row parallelism.

### MLP Block

Standard Transformer MLP:
$$\text{MLP}(X) = \text{GeLU}(X W_1) W_2$$

With Megatron parallelism:

```
Input X (replicated across TP group)
    │
    ▼ Column-parallel: Y = XW₁ (no communication)
    │
    ▼ GeLU(Y) - local on sharded Y
    │
    ▼ Row-parallel: Z = YW₂ (AllReduce)
    │
    ▼
Output Z (replicated)
```

**Key insight**: Column-parallel produces column-sharded output, which is exactly what row-parallel needs as input!

### Why GELU Doesn't Break This

GELU is applied element-wise. Each element of $Y$ is computed by one GPU.

Even though:
$$\text{GeLU}(Y_1 + Y_2) \neq \text{GeLU}(Y_1) + \text{GeLU}(Y_2)$$

We're not splitting elements—we're splitting the tensor along the hidden dimension. Each GPU computes GeLU on its complete slice:

$$\text{GeLU}(Y) = [\text{GeLU}(Y_0) | \text{GeLU}(Y_1) | \cdots | \text{GeLU}(Y_{P-1})]$$

This is valid because GeLU is applied independently to each element.

### Attention Block

Transformer attention:
$$\text{Attention}(X) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

Where $Q = XW^Q$, $K = XW^K$, $V = XW^V$.

**Multi-head attention is naturally parallelizable**:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

Each head is independent. With $h$ heads and $P$ GPUs (where $P$ divides $h$):

Each GPU computes $h/P$ heads.

```
Input X (replicated)
    │
    ├──▶ GPU 0: heads 0, 1, ..., h/P - 1
    ├──▶ GPU 1: heads h/P, ..., 2h/P - 1
    ├──▶ ...
    └──▶ GPU P-1: heads (P-1)h/P, ..., h - 1
    │
    ▼ Each GPU: Q, K, V projections (column-parallel)
    ▼ Each GPU: Attention computation (local)
    ▼ Each GPU: Output projection (row-parallel)
    │
    ▼ AllReduce
    │
Output (replicated)
```

### Communication Count

Per Transformer layer:

| Component | Communication |
|-----------|---------------|
| Attention Q, K, V projections | None (column-parallel) |
| Attention computation | None (head-parallel) |
| Attention output projection | 1 AllReduce |
| MLP up-projection | None (column-parallel) |
| GeLU | None (local) |
| MLP down-projection | 1 AllReduce |

**Total: 2 AllReduce operations per Transformer layer**

## Communication Analysis

### Volume per Layer

For a Transformer with:
- Hidden dimension $d$
- Tensor parallel degree $P$
- Sequence length $s$
- Batch size $b$

Each AllReduce synchronizes the activation tensor:
$$V_{\text{AR}} = 2 \cdot \frac{P-1}{P} \cdot b \cdot s \cdot d \cdot \text{sizeof}(\text{dtype})$$

Two AllReduces per layer:
$$V_{\text{layer}} = 4 \cdot \frac{P-1}{P} \cdot b \cdot s \cdot d \cdot \text{sizeof}(\text{dtype})$$

For FP16 and large $P$:
$$V_{\text{layer}} \approx 8bsd \text{ bytes}$$

### Time per Layer

Using α-β model:
$$T_{\text{comm}} = 2 \times \left( 2(P-1)\alpha + 2\frac{P-1}{P} \cdot \frac{bsd}{\beta} \right)$$

For large tensors (bandwidth-dominated):
$$T_{\text{comm}} \approx \frac{8bsd}{\beta}$$

### Compute-Communication Ratio

Compute per layer (forward only):
$$C_{\text{layer}} \approx 4 \cdot b \cdot s \cdot d^2 + 2 \cdot b \cdot s^2 \cdot d$$

For $s \ll d$ (typical for LLMs):
$$C_{\text{layer}} \approx 4bsd^2$$

Ratio:
$$R = \frac{C/P}{T_{\text{comm}}} = \frac{4bsd^2 / (P \cdot F)}{8bsd / \beta} = \frac{d \cdot \beta}{2PF}$$

For H100 ($F = 2 \times 10^{15}$ FLOP/s), NVLink ($\beta = 900$ GB/s), $d = 8192$:
$$R = \frac{8192 \times 900 \times 10^9}{2P \times 2 \times 10^{15}} = \frac{8192 \times 900}{4P \times 10^6} \approx \frac{1.8}{P}$$

For $P = 8$: $R \approx 0.23$ — communication-bound!

**This is why tensor parallelism is typically limited to within a node.**

## Layer Normalization

### The Challenge

LayerNorm:
$$\text{LN}(X) = \gamma \cdot \frac{X - \mu}{\sigma} + \beta$$

where:
$$\mu = \frac{1}{d} \sum_{i=1}^{d} X_i, \quad \sigma = \sqrt{\frac{1}{d} \sum_{i=1}^{d} (X_i - \mu)^2}$$

Computing $\mu$ and $\sigma$ requires the full hidden dimension—can't be done on sharded activations.

### Solutions

**Option 1: Pre-LayerNorm on Replicated Activations**

Apply LayerNorm before entering TP region:

```
X (replicated) → LayerNorm → X' (replicated) → Column-parallel → ...
```

This is the Megatron-LM approach.

**Option 2: Parallel LayerNorm**

Compute partial statistics on each shard, AllReduce to get global statistics.

GPU $i$ computes:
$$\mu_i = \frac{1}{d/P} \sum_{j \in \text{shard } i} X_j, \quad s_i = \sum_{j \in \text{shard } i} X_j^2$$

AllReduce to get:
$$\mu = \frac{1}{P} \sum_i \mu_i, \quad \sigma = \sqrt{\frac{1}{d} \sum_i s_i - \mu^2}$$

Then normalize locally with global statistics.

**Cost**: Additional AllReduce per LayerNorm. Usually avoided.

## Dropout in Tensor Parallelism

### The Challenge

Dropout applies a random mask:
$$\text{Dropout}(X) = \frac{X \odot M}{1-p}$$

where $M$ is a binary mask with $P(M_i = 1) = 1 - p$.

For reproducibility, the mask must be the same across GPUs for replicated activations.

### Solution: Synchronized RNG

```python
class TPDropout(nn.Module):
    def __init__(self, p, tp_group):
        self.p = p
        self.tp_group = tp_group

    def forward(self, x):
        if self.training:
            # Synchronize RNG state across TP group
            seed = torch.randint(0, 2**32, (1,))
            dist.broadcast(seed, src=0, group=self.tp_group)

            # Generate identical mask on all GPUs
            gen = torch.Generator().manual_seed(seed.item())
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p),
                                   generator=gen)

            return x * mask / (1 - self.p)
        return x
```

## Implementation

### Column-Parallel Linear

```python
import torch
import torch.nn as nn
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    """Linear layer with column-wise weight partitioning."""

    def __init__(self, in_features, out_features, tp_group, bias=True):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)

        # Each GPU holds out_features / tp_size columns
        self.out_features_per_gpu = out_features // self.tp_size

        # Local weight shard
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_gpu, in_features)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features_per_gpu)
            )
        else:
            self.bias = None

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: [batch, seq, in_features] - replicated
        # out: [batch, seq, out_features_per_gpu] - column-sharded
        out = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            out = out + self.bias
        return out
```

### Row-Parallel Linear

```python
class RowParallelLinear(nn.Module):
    """Linear layer with row-wise weight partitioning."""

    def __init__(self, in_features, out_features, tp_group, bias=True):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)

        # Each GPU holds in_features / tp_size rows
        self.in_features_per_gpu = in_features // self.tp_size

        # Local weight shard
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_gpu)
        )
        if bias:
            # Bias added after AllReduce (only on rank 0)
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: [batch, seq, in_features_per_gpu] - column-sharded
        # Compute partial result
        partial = torch.matmul(x, self.weight.t())

        # AllReduce to sum partial results
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=self.tp_group)

        # Add bias (same on all GPUs)
        if self.bias is not None:
            partial = partial + self.bias

        return partial
```

### Megatron MLP Block

```python
class TensorParallelMLP(nn.Module):
    """MLP block with Megatron-style tensor parallelism."""

    def __init__(self, hidden_size, ffn_hidden_size, tp_group):
        super().__init__()
        self.tp_group = tp_group

        # Up projection: column-parallel (output is sharded)
        self.up_proj = ColumnParallelLinear(
            hidden_size, ffn_hidden_size, tp_group, bias=True
        )

        # Down projection: row-parallel (input is sharded, output replicated)
        self.down_proj = RowParallelLinear(
            ffn_hidden_size, hidden_size, tp_group, bias=True
        )

        self.activation = nn.GELU()

    def forward(self, x):
        # x: [batch, seq, hidden] - replicated
        x = self.up_proj(x)  # [batch, seq, ffn/TP] - sharded
        x = self.activation(x)  # Local GELU
        x = self.down_proj(x)  # [batch, seq, hidden] - replicated
        return x
```

### Tensor Parallel Attention

```python
class TensorParallelAttention(nn.Module):
    """Multi-head attention with tensor parallelism."""

    def __init__(self, hidden_size, num_heads, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_heads_per_gpu = num_heads // self.tp_size
        self.head_dim = hidden_size // num_heads

        # Q, K, V projections: column-parallel
        self.qkv_proj = ColumnParallelLinear(
            hidden_size, 3 * hidden_size, tp_group, bias=True
        )

        # Output projection: row-parallel
        self.out_proj = RowParallelLinear(
            hidden_size, hidden_size, tp_group, bias=True
        )

    def forward(self, x, mask=None):
        batch, seq, _ = x.shape

        # QKV projection (column-parallel, no comm)
        qkv = self.qkv_proj(x)  # [batch, seq, 3 * hidden / TP]

        # Reshape to separate Q, K, V for local heads
        qkv = qkv.view(batch, seq, 3, self.num_heads_per_gpu, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Attention computation (local)
        q = q.transpose(1, 2)  # [batch, heads_per_gpu, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = attn_weights / (self.head_dim ** 0.5)
        if mask is not None:
            attn_weights = attn_weights + mask
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(
            batch, seq, self.num_heads_per_gpu * self.head_dim
        )

        # Output projection (row-parallel, AllReduce)
        output = self.out_proj(attn_output)

        return output
```

## Backward Pass Analysis

### Gradient Flow

For column-parallel $Y = XW$ where $W$ is column-sharded:

**Forward**: No communication
**Backward**:

Given $\frac{\partial L}{\partial Y}$ (sharded same as $Y$):

$$\frac{\partial L}{\partial W_i} = X^T \frac{\partial L}{\partial Y_i}$$

Local computation—$X$ is replicated, $\frac{\partial L}{\partial Y_i}$ is local.

$$\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T = \sum_i \frac{\partial L}{\partial Y_i} W_i^T$$

Requires AllReduce to sum contributions from all shards.

### Backward Communication

For one Megatron-style layer:

| Forward | Backward |
|---------|----------|
| Column-parallel MLP: 0 comm | AllReduce (for $\partial L/\partial X$) |
| Row-parallel MLP: AllReduce | 0 comm (input grad is local) |
| Column-parallel Attention: 0 comm | AllReduce |
| Row-parallel Attention output: AllReduce | 0 comm |

**Total per layer**: 4 AllReduce (2 forward + 2 backward)

## Scaling Limits

### Maximum Tensor Parallel Degree

The tensor parallel degree $P$ is limited by:

1. **Head divisibility**: $P$ must divide number of attention heads
2. **Hidden dimension divisibility**: $P$ must divide hidden dimension
3. **Communication overhead**: NVLink bandwidth limits

**Practical limits**:

| Node Type | Max TP | Reason |
|-----------|--------|--------|
| 8× A100 NVLink | 8 | Full node, high bandwidth |
| 8× H100 NVLink | 8 | Full node, higher bandwidth |
| 2 nodes (16 GPUs) | 16 | Inter-node communication expensive |

### When to Use Tensor Parallelism

**Use TP when**:
- Model doesn't fit in single GPU memory
- Within a single node (fast NVLink)
- Need to reduce per-GPU memory for activations

**Don't use TP when**:
- Model fits comfortably (DP is simpler)
- Crossing node boundaries (use PP instead)
- Very small batch sizes (latency-dominated)

## Exercises

1. **Bias partitioning**: In column-parallel linear $Y = XW + b$, show that partitioning $b$ along with $W$ columns gives correct results. In row-parallel, why must bias be added after AllReduce?

2. **Communication derivation**: For a Transformer with $d = 4096$, $s = 2048$, $b = 4$, TP = 8, calculate the exact bytes transferred per layer in forward pass. Use FP16.

3. **Compute-communication ratio**: For the same model, compute $R$ assuming H100 with 2 PFLOP/s and NVLink at 900 GB/s. Is the layer compute-bound or communication-bound?

4. **LayerNorm parallelism**: Derive the formulas for computing global mean and variance from partial statistics on sharded tensors. What are the AllReduce volumes needed?

5. **GeLU placement**: Why must GeLU come between column-parallel and row-parallel layers, not before or after both? What would go wrong if GeLU came after row-parallel?

6. **Attention head constraints**: With 32 attention heads and TP = 6, what goes wrong? How would you handle this case?

7. **Backward analysis**: For a column-parallel linear layer $Y = XW$, derive the gradient formulas and show that $\nabla_X L$ requires AllReduce while $\nabla_W L$ does not.

## Key Takeaways

1. **Linearity enables parallelism**: $f(X_1 + X_2) = f(X_1) + f(X_2)$ allows independent computation.

2. **Column-parallel is communication-free forward**: Split output dimension, no AllReduce needed.

3. **Row-parallel requires AllReduce**: Split input dimension, must sum partial results.

4. **Megatron pattern chains them**: Column-parallel → non-linearity → row-parallel → AllReduce.

5. **2 AllReduce per layer**: One for attention output, one for MLP output.

6. **Non-linear ops need care**: GeLU works on sharded tensors; LayerNorm doesn't.

7. **Tensor parallelism is communication-intensive**: Best within NVLink-connected nodes.

8. **Backward doubles communication**: 4 AllReduce per layer total (forward + backward).
