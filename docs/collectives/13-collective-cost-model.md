---
title: "The Collective Cost Model"
subtitle: "Predicting Communication Time for Any Operation"
---

::: {.chapter-opener}
Armed with α-β analysis and algorithm knowledge, we can predict the cost of any collective operation before running it.
:::

::: {.investigation-question}
**The Question**: Your AllReduce takes 500ms. Is that good or bad? How do you know if you're achieving near-optimal performance?
:::

## Cost Formulas

### Point-to-Point
$$T = \alpha + \frac{n}{\beta}$$

### AllReduce (Ring)
$$T = 2(P-1)\alpha + 2\frac{P-1}{P} \cdot \frac{n}{\beta}$$

### AllGather (Ring)
$$T = (P-1)\alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

### ReduceScatter (Ring)
$$T = (P-1)\alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

### AlltoAll
$$T = (P-1)\alpha + \frac{P-1}{P} \cdot \frac{n}{\beta}$$

## Hierarchical Collectives

For multi-level networks (intra-node + inter-node):

$$T = T_{\text{intra}} + T_{\text{inter}}$$

*[Detailed analysis to be completed]*

## Predicting vs Measuring

*[Methodology for validation to be completed]*

## Exercises

*[To be completed]*
