---
title: "Batch Size and Learning Dynamics"
subtitle: "Critical Batch Size, Learning Rate Scaling, and LARS/LAMB"
---

::: {.chapter-opener}
Batch size is not just a memory parameter—it fundamentally affects learning dynamics. There's a critical batch size beyond which returns diminish rapidly.
:::

::: {.investigation-question}
**The Question**: You want to scale to 10,000 GPUs. Can you just increase batch size linearly? What happens to convergence, and how do you adjust the learning rate?
:::

## The Critical Batch Size

Gradient noise scale:

$$B_{\text{crit}} = \frac{\text{tr}(H)}{\text{tr}(\Sigma)}$$

Below $B_{\text{crit}}$: training is noise-limited, more steps needed.
Above $B_{\text{crit}}$: diminishing returns, compute wasted.

## Learning Rate Scaling Rules

### Linear Scaling (Goyal et al.)
$$\eta(B) = \eta_0 \cdot \frac{B}{B_0}$$

Valid up to $B_{\text{crit}}$.

### Square Root Scaling
$$\eta(B) = \eta_0 \cdot \sqrt{\frac{B}{B_0}}$$

More conservative, works for larger batches.

## LARS and LAMB

Layer-wise Adaptive Rate Scaling:

$$\eta_l = \eta \cdot \frac{||w_l||}{||\nabla w_l|| + \lambda ||w_l||}$$

*[Full treatment to be completed]*

## Exercises

*[To be completed]*
