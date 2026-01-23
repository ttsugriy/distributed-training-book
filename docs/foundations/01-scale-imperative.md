---
title: "The Scale Imperative"
subtitle: "Why Distributed Training Is No Longer Optional"
---

<div class="chapter-opener" markdown>
In 2017, a single GPU could train the largest language models. By 2024, the largest models require thousands of GPUs running for months. This is not a temporary inconvenience—it is the new reality of machine learning.
</div>

<div class="investigation-question" markdown>
**The Question**: A model has 70 billion parameters. Each parameter is 2 bytes (FP16). That's 140GB just for weights—before gradients, optimizer states, or activations. A single H100 has 80GB of memory. How do we train this model at all?
</div>

## The Three Walls

Training large models hits three fundamental limits:

### The Memory Wall

A model with $\Psi$ parameters in mixed precision requires approximately:

$$\text{Memory} = 2\Psi + 2\Psi + 12\Psi = 16\Psi \text{ bytes}$$

Where:

- $2\Psi$: FP16 parameters
- $2\Psi$: FP16 gradients
- $12\Psi$: FP32 optimizer states (Adam: parameters, momentum, variance)

For a 70B model: $16 \times 70 \times 10^9 = 1.12\text{TB}$

No single accelerator holds this. We *must* distribute.

### The Time Wall

Even if memory weren't a constraint, training time would be:

$$T = \frac{6 \cdot N \cdot D}{F}$$

Where:

- $N$: parameters
- $D$: training tokens
- $F$: FLOP/s of the device
- Factor of 6: forward (2) + backward (4) FLOPs per parameter per token

For GPT-3 (175B parameters, 300B tokens) on a single H100 (1979 TFLOP/s at FP16):

$$T = \frac{6 \times 175 \times 10^9 \times 300 \times 10^9}{1979 \times 10^{12}} = 159 \times 10^6 \text{ seconds} \approx 5 \text{ years}$$

We need parallelism to finish in reasonable time.

### The Cost Wall

GPU-hours are expensive. Inefficiency is waste. If we achieve only 40% of peak FLOP/s due to poor parallelization, we're burning 60% of our compute budget.

The economic imperative is clear: **understand the mathematics of distributed training, or pay for ignorance**.

## The Thesis

This book's central claim:

> **Every parallelism strategy exploits a mathematical property of the computation. The optimal distribution follows from understanding which operations can be decomposed and which must be synchronized.**

We will derive—not just describe—each parallelism strategy from the mathematical property that enables it:

| Strategy | Mathematical Property | Key Operation |
|----------|----------------------|---------------|
| Data Parallelism | Associativity | Gradient accumulation |
| Tensor Parallelism | Linearity | Matrix multiplication |
| Pipeline Parallelism | Separability | Layer composition |
| Sequence Parallelism | Decomposability | Attention computation |
| Expert Parallelism | Sparsity | Conditional routing |

## The Extended Roofline

The original roofline model shows performance bounded by compute or memory bandwidth. For distributed training, we add a third ceiling: **network bandwidth**.

```
Performance (FLOP/s)
       ^
       |     _______________  Compute ceiling (peak FLOP/s)
       |    /
       |   /________________  Memory ceiling (bytes/s × arithmetic intensity)
       |  /
       | /_________________  Network ceiling (bytes/s × communication intensity)
       |/
       +-----------------------> Arithmetic Intensity (FLOPs/byte)
```

Most distributed training is **communication-bound**, not compute-bound. Understanding this changes everything about how we design systems.

## What We'll Build

By the end of this book, you'll have:

1. **Mental models** for reasoning about any distributed configuration
2. **Estimation skills** to predict throughput from first principles
3. **Derivation ability** to understand new techniques as they emerge
4. **Debugging intuition** to identify bottlenecks quickly

Let's begin with the foundations.
