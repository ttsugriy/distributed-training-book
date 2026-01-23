---
title: "Tensor Parallelism from Linearity"
subtitle: "Why Matrix Multiplication Shards and GELU Doesn't"
---

::: {.chapter-opener}
Matrix multiplication is linear: $f(aX) = af(X)$ and $f(X + Y) = f(X) + f(Y)$. This single property enables tensor parallelism. Non-linear operations like GELU break this property, forcing synchronization.
:::

::: {.investigation-question}
**The Question**: We want to shard a linear layer $Y = XW$ across 8 GPUs. Can we split $W$ column-wise? Row-wise? What about the bias? What about LayerNorm? What about GELU?
:::

## The Linearity Property

A function $f$ is linear if:
1. **Additivity**: $f(X + Y) = f(X) + f(Y)$
2. **Homogeneity**: $f(aX) = a \cdot f(X)$

Equivalently: $f(aX + bY) = af(X) + bf(Y)$

## Column-Parallel Linear Layers

Split weight matrix column-wise: $W = [W_1 | W_2 | \cdots | W_P]$

$$XW = X[W_1 | W_2 | \cdots | W_P] = [XW_1 | XW_2 | \cdots | XW_P]$$

Each GPU computes $XW_i$ independently. No communication for the matmul itself.

**But**: The output is split across GPUs. What next?

## Row-Parallel Linear Layers

Split weight matrix row-wise: $W = \begin{bmatrix} W_1 \\ W_2 \\ \vdots \\ W_P \end{bmatrix}$

Need input split accordingly: $X = [X_1 | X_2 | \cdots | X_P]$

$$XW = \sum_{i=1}^P X_i W_i$$

Requires AllReduce to sum partial results.

## The Megatron Pattern

For MLP block with hidden dimension H:
```
Input X (replicated)
    ↓
Column-parallel: Y = XW₁ (no comm, output split)
    ↓
GELU(Y) (local, on split data)
    ↓  
Row-parallel: Z = YW₂ (AllReduce to sum)
    ↓
Output Z (replicated)
```

Communication: One AllReduce per MLP block.

## Why GELU Breaks Sharding

GELU is non-linear:
$$\text{GELU}(X_1 + X_2) \neq \text{GELU}(X_1) + \text{GELU}(X_2)$$

If we tried to apply GELU to split activations, we'd get the wrong answer.

**Solution**: Arrange computation so GELU operates on complete (though sharded across features) tensors.

## Attention Parallelism

Attention heads are embarrassingly parallel:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

Split heads across GPUs. Each GPU computes a subset of heads. AllReduce after output projection.

## Communication Pattern Summary

| Layer | Operation | Communication |
|-------|-----------|---------------|
| Attention Q,K,V | Column-parallel | None |
| Attention heads | Independent | None |
| Attention output | Row-parallel | AllReduce |
| MLP first linear | Column-parallel | None |
| GELU | Local | None |
| MLP second linear | Row-parallel | AllReduce |

**Total per layer**: 2 AllReduce operations (attention + MLP)

## Exercises

*[To be completed]*
