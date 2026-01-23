---
title: "Pipeline Parallelism from Separability"
subtitle: "Decomposing Sequential Computation"
---

::: {.chapter-opener}
A neural network is a composition of layers: $f = f_L \circ f_{L-1} \circ \cdots \circ f_1$. This sequential structure enables pipeline parallelism, but creates "bubbles" of idle time that we must minimize.
:::

::: {.investigation-question}
**The Question**: If we split a 32-layer model into 4 stages of 8 layers each, how much time is wasted to "pipeline bubbles"? Can we reduce it to zero?
:::

## The Separability Property

The model is a composition:
$$f(x) = f_L(f_{L-1}(\cdots f_1(x)))$$

Each $f_i$ can be computed by a different device, as long as we pass activations between them.

## The Pipeline Bubble Problem

With $P$ pipeline stages and a single batch:

```
Time →
Stage 0: [F₀][B₀]
Stage 1:     [F₁][B₁]
Stage 2:         [F₂][B₂]
Stage 3:             [F₃][B₃]
              ↑ Idle time (bubble)
```

Bubble fraction with 1 micro-batch: $(P-1)/P$

With 4 stages: 75% idle time!

## Micro-batching (GPipe)

Split batch into $m$ micro-batches:

```
Stage 0: [F₀₀][F₀₁][F₀₂][F₀₃]      [B₀₃][B₀₂][B₀₁][B₀₀]
Stage 1:     [F₁₀][F₁₁][F₁₂][F₁₃]  [B₁₃][B₁₂][B₁₁][B₁₀]
Stage 2:         [F₂₀][F₂₁][F₂₂][F₂₃][B₂₃][B₂₂][B₂₁][B₂₀]
Stage 3:             [F₃₀][F₃₁][F₃₂][F₃₃][B₃₃][B₃₂][B₃₁][B₃₀]
```

Bubble fraction:
$$\text{Bubble} = \frac{P - 1}{m + P - 1}$$

With $P=4$, $m=32$: Bubble = 3/35 ≈ 8.6%

## 1F1B Scheduling

Interleave forward and backward passes:

```
Stage 0: [F₀][F₁][F₂][F₃][B₀][F₄][B₁][F₅][B₂]...
Stage 1:     [F₀][F₁][F₂][B₀][F₃][B₁][F₄][B₂]...
```

**Advantage**: Limits activation memory to O(P) instead of O(m).

## Zero-Bubble Pipeline Parallelism

Split backward pass into:
- $B$: gradient w.r.t. input (needed for previous stage)
- $W$: gradient w.r.t. weights (can be delayed)

Schedule $W$ passes to fill bubbles:

```
Stage 0: [F][F][F][B][F][B][W][B][W]...
              ↑ W fills what would be bubble
```

Achieves near-zero bubble overhead.

*[Detailed algorithm to be completed]*

## Communication Pattern

- Only point-to-point between adjacent stages
- Activation tensors: batch × sequence × hidden
- Much lower volume than AllReduce of full gradients

## Exercises

*[To be completed]*
