---
title: "Data Parallelism from Associativity"
subtitle: "Why Gradient Accumulation Enables Distribution"
---

::: {.chapter-opener}
Data parallelism works because gradient accumulation is associative. This isn't an implementation detail—it's the mathematical foundation that makes the entire approach valid.
:::

::: {.investigation-question}
**The Question**: We compute gradients on different batches on different GPUs, then sum them. Why does this give us the same result as computing on the full batch? When does it fail?
:::

## The Mathematical Foundation

Gradient of loss over batch $B$:

$$\nabla_\theta L(B) = \frac{1}{|B|} \sum_{x \in B} \nabla_\theta \ell(x, \theta)$$

Splitting batch $B = B_1 \cup B_2 \cup \cdots \cup B_P$:

$$\nabla_\theta L(B) = \frac{1}{|B|} \sum_{i=1}^{P} \sum_{x \in B_i} \nabla_\theta \ell(x, \theta) = \frac{1}{P} \sum_{i=1}^{P} \nabla_\theta L(B_i)$$

This works because:
1. Summation is **associative**: $(a + b) + c = a + (b + c)$
2. Summation is **commutative**: $a + b = b + a$

## The Algorithm

```
# On each GPU i:
1. Forward pass on local batch B_i
2. Backward pass to compute ∇L(B_i)
3. AllReduce gradients: g = (1/P) Σ ∇L(B_i)
4. Update: θ ← θ - η·g
```

## Communication Analysis

Per step:
- Compute: $6\Psi \cdot |B_i|$ FLOPs
- Communicate: $2\Psi$ bytes (AllReduce gradients)

Communication intensity:
$$I = \frac{6\Psi \cdot |B_i|}{2\Psi} = 3|B_i|$$

Larger local batch → higher intensity → less communication-bound.

## When Associativity Fails

Floating-point addition is *approximately* associative:
$$(a + b) + c \approx a + (b + c)$$

Different reduction orders → slightly different results → non-determinism.

For reproducibility: fix reduction order, use higher precision for accumulation.

## Exercises

*[To be completed]*
