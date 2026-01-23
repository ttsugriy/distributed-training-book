---
title: "Ring and Tree Algorithms"
subtitle: "Bandwidth-Optimal and Latency-Optimal Collectives"
---

::: {.chapter-opener}
The same logical operation—AllReduce—can be implemented many ways. Ring algorithms optimize for bandwidth; tree algorithms optimize for latency. Choosing correctly depends on message size.
:::

::: {.investigation-question}
**The Question**: Why is ring AllReduce bandwidth-optimal? Can we prove it achieves the theoretical minimum communication volume?
:::

## Ring AllReduce

### Algorithm

*[Step-by-step algorithm to be completed]*

### Bandwidth Optimality

**Theorem**: Ring AllReduce achieves bandwidth cost:
$$T_{\text{bw}} = 2 \cdot \frac{P-1}{P} \cdot \frac{n}{\beta} \approx 2 \cdot \frac{n}{\beta}$$

This is optimal: every byte must leave and enter each process at least once.

*[Proof to be completed]*

## Tree AllReduce

### Algorithm

*[Recursive halving/doubling to be completed]*

### Latency Optimality

**Theorem**: Tree AllReduce achieves latency cost:
$$T_{\text{lat}} = 2 \log_2(P) \cdot \alpha$$

*[Proof and comparison to be completed]*

## When to Use Which

| Message Size | Preferred Algorithm |
|-------------|---------------------|
| Small (< crossover) | Tree |
| Large (> crossover) | Ring |
| Very large, hierarchical | 2D Ring |

## Exercises

*[To be completed]*
