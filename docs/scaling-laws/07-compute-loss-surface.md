---
title: "The Compute-Loss Surface"
subtitle: "Mapping the Relationship Between Resources and Performance"
---

::: {.chapter-opener}
Loss is a function of two investments: model size and training data. Understanding this surface is the first step to allocating compute efficiently.
:::

::: {.investigation-question}
**The Question**: You have a fixed compute budget C. Should you train a larger model on less data or a smaller model on more data? The loss surface tells us there's an optimal allocation.
:::

## The Loss Function

The empirical loss follows:

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty$$

Where:
- $N$: number of parameters
- $D$: number of training tokens
- $A, B, \alpha, \beta$: fitted constants
- $L_\infty$: irreducible loss (entropy of language)

## The Compute Constraint

Training compute:

$$C = 6ND$$

This creates an iso-compute curve on the $(N, D)$ plane.

## Minimizing Loss Under Constraint

Using Lagrange multipliers on $L(N, D)$ subject to $C = 6ND$:

$$\frac{\partial L}{\partial N} = \lambda \cdot 6D$$
$$\frac{\partial L}{\partial D} = \lambda \cdot 6N$$

*[Full derivation to be completed]*

## The Power Laws

From Kaplan et al. (2020):
- $\alpha \approx 0.076$
- $\beta \approx 0.095$

From Hoffmann et al. (2022, Chinchilla):
- Optimal: $N \propto C^{0.5}$, $D \propto C^{0.5}$

## Implications

*[Section to be completed]*

## Exercises

1. Given a compute budget of $10^{23}$ FLOPs, calculate the optimal model size and token count under Chinchilla scaling.

2. Derive the optimal $N/D$ ratio from the loss function.

3. GPT-3 used 175B parameters and 300B tokens. Calculate how undertrained it was relative to Chinchilla optimal.
