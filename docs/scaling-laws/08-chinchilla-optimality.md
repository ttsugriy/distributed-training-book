---
title: "Chinchilla Optimality"
subtitle: "The 20:1 Ratio and Compute-Optimal Training"
---

::: {.chapter-opener}
For years, the field scaled models without enough data. Chinchilla revealed the optimal balance: approximately 20 tokens per parameter.
:::

::: {.investigation-question}
**The Question**: Why were GPT-3 and similar models "undertrained"? What does it mean to be compute-optimal, and how do we achieve it?
:::

## The Chinchilla Finding

C = 6ND, and optimal allocation gives:

$$N^* = \left(\frac{C}{6 \cdot 20}\right)^{0.5}$$
$$D^* = 20 \cdot N^*$$

## The 20:1 Ratio

*[Derivation and evidence to be completed]*

## Practical Application

*[Worked examples to be completed]*

## When to Deviate

- Inference cost constraints (smaller model, more tokens)
- Data scarcity (larger model, fewer tokens)
- Downstream task requirements

## Exercises

*[To be completed]*
