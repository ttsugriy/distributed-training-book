---
title: "Expert Parallelism from Sparsity"
subtitle: "Routing Tokens to Distributed Experts"
---

::: {.chapter-opener}
Mixture of Experts models activate only a subset of parameters per token. This sparsity enables massive parameter counts without proportional compute costs—but requires careful routing.
:::

::: {.investigation-question}
**The Question**: A model has 8 experts distributed across 8 GPUs. A token is routed to experts on GPUs 3 and 7. How does the token get there and back? What if all tokens want the same expert?
:::

## The Sparsity Property

Standard dense layer: all parameters active for all tokens.

MoE layer: each token activates top-k experts (typically k=1 or k=2).

$$y = \sum_{i \in \text{top-k}} g_i \cdot E_i(x)$$

where $g_i$ is the gating weight and $E_i$ is expert $i$.

## Communication Pattern: AlltoAll

```
Before AlltoAll:           After AlltoAll:
GPU 0: [tokens for E0-7]   GPU 0: [tokens for E0 only]
GPU 1: [tokens for E0-7]   GPU 1: [tokens for E1 only]
...                        ...
GPU 7: [tokens for E0-7]   GPU 7: [tokens for E7 only]
```

Two AlltoAll operations per MoE layer:
1. Dispatch: tokens → experts
2. Combine: expert outputs → original positions

## Load Balancing

**The problem**: If all tokens route to one expert, that GPU is overloaded while others idle.

**Solution 1: Auxiliary loss**
$$L_{\text{aux}} = \alpha \cdot P \cdot \sum_{i=1}^{P} f_i \cdot p_i$$

where $f_i$ = fraction of tokens to expert $i$, $p_i$ = average routing probability.

**Solution 2: Capacity factor**
Limit tokens per expert: $C = \lceil \text{capacity\_factor} \cdot N / P \rceil$

Tokens exceeding capacity are dropped or overflow to next expert.

## Expert Parallelism + Other Dimensions

```
            Data Parallel
               /   \
              /     \
         Expert Parallel
            /   \
           E0    E1  ...  E7
           |     |        |
          TP    TP       TP  (Tensor Parallel within expert)
```

*[Composition patterns to be completed]*

## Exercises

*[To be completed]*
