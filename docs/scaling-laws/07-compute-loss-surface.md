---
title: "The Compute-Loss Surface"
subtitle: "Mapping the Relationship Between Resources and Performance"
---

<div class="chapter-opener" markdown>
Loss is a function of two investments: model size and training data. Understanding this surface is the first step to allocating compute efficiently.
</div>

<div class="investigation-question" markdown>
**The Question**: You have a fixed compute budget C. Should you train a larger model on less data or a smaller model on more data? The loss surface tells us there's an optimal allocation—and most models before 2022 got it wrong.
</div>

## The Empirical Discovery

In 2020, researchers at OpenAI made a remarkable observation: language model loss follows smooth, predictable power laws. Plot log-loss against log-parameters or log-tokens, and you get straight lines.

This isn't obvious. Complex systems often exhibit chaotic behavior. But neural language models, across many orders of magnitude, follow:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}$$

Where $N_c$, $D_c$ are critical scales and $\alpha_N$, $\alpha_D$ are power law exponents.

## The Loss Surface

Combining both dependencies:

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N / \alpha} + \left(\frac{D_c}{D}\right)^{\alpha_D / \alpha}\right]^\alpha + L_\infty$$

Or in the simpler additive form often used:

$$L(N, D) = \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + L_\infty$$

Where:

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| $N$ | Number of parameters | $10^6$ to $10^{12}$ |
| $D$ | Training tokens | $10^9$ to $10^{13}$ |
| $A$ | Parameter scaling constant | ~400 |
| $B$ | Data scaling constant | ~400 |
| $\alpha$ | Parameter exponent | 0.34 (Kaplan) / 0.50 (Chinchilla) |
| $\beta$ | Data exponent | 0.28 (Kaplan) / 0.50 (Chinchilla) |
| $L_\infty$ | Irreducible loss | ~1.69 nats |

The irreducible loss $L_\infty$ represents the entropy of natural language—even a perfect model can't predict the unpredictable.

## The Compute Constraint

Training compute is dominated by matrix multiplications. For a transformer:

$$C \approx 6ND$$

**Derivation**: Each token passes through layers where:

- Forward pass: ~$2N$ FLOPs (matrix multiply is $2 \times$ parameters)
- Backward pass: ~$4N$ FLOPs (gradient computation + weight updates)
- Total per token: ~$6N$ FLOPs

For $D$ tokens:
$$C = 6ND \text{ FLOPs}$$

This creates a **constraint surface** in $(N, D, C)$ space. For fixed $C$, we get a hyperbola:

$$D = \frac{C}{6N}$$

## Minimizing Loss Under Compute Constraint

**Problem**: Minimize $L(N, D)$ subject to $C = 6ND$

**Method**: Lagrange multipliers

The Lagrangian:
$$\mathcal{L} = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + \lambda(C - 6ND)$$

Taking partial derivatives:

$$\frac{\partial \mathcal{L}}{\partial N} = -\frac{A\alpha}{N^{\alpha+1}} - 6\lambda D = 0$$

$$\frac{\partial \mathcal{L}}{\partial D} = -\frac{B\beta}{D^{\beta+1}} - 6\lambda N = 0$$

From the first equation:
$$\lambda = -\frac{A\alpha}{6DN^{\alpha+1}}$$

Substituting into the second:
$$\frac{B\beta}{D^{\beta+1}} = \frac{A\alpha N}{DN^{\alpha+1}} = \frac{A\alpha}{N^\alpha D}$$

Therefore:
$$\frac{B\beta}{D^\beta} = \frac{A\alpha}{N^\alpha}$$

This says: **at the optimum, the marginal contribution to loss reduction from parameters equals that from data**.

Rearranging:
$$\frac{N^\alpha}{D^\beta} = \frac{A\alpha}{B\beta}$$

## The Optimal Allocation

Using $C = 6ND$ to eliminate one variable:

$$N^* = \left(\frac{A\alpha}{B\beta}\right)^{\frac{\beta}{\alpha+\beta}} \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}$$

$$D^* = \left(\frac{B\beta}{A\alpha}\right)^{\frac{\alpha}{\alpha+\beta}} \left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha+\beta}}$$

**Key insight**: Both $N^*$ and $D^*$ are power laws in $C$.

If $\alpha = \beta$ (as Chinchilla suggests):

$$N^* \propto C^{0.5}, \quad D^* \propto C^{0.5}$$

The optimal ratio:
$$\frac{D^*}{N^*} = \frac{B\beta}{A\alpha}$$

For Chinchilla parameters: $D^*/N^* \approx 20$

**The 20:1 Rule**: Optimal training uses ~20 tokens per parameter.

## Kaplan vs Chinchilla

Two influential papers reached different conclusions:

| Paper | $\alpha$ | $\beta$ | Optimal $N^* \propto$ | Optimal $D^* \propto$ | Tokens/Param |
|-------|----------|---------|----------------------|----------------------|--------------|
| Kaplan (2020) | 0.076 | 0.095 | $C^{0.73}$ | $C^{0.27}$ | ~1.7 |
| Chinchilla (2022) | 0.34 | 0.28 | $C^{0.50}$ | $C^{0.50}$ | ~20 |

**Why the difference?**

Kaplan trained each model for a fixed number of steps, not to convergence. This systematically undertrained larger models, biasing the fit toward "make models bigger."

Chinchilla trained models to convergence at each size, revealing the true scaling relationship.

## Visualizing the Surface

The loss surface can be visualized as contour lines:

```
L (loss)
    │
2.5 ├─────────────────────────────
    │   ╲
2.3 ├────╲────────────────────────
    │     ╲   Iso-loss curves
2.1 ├──────╲──────────────────────
    │       ╲
1.9 ├────────●────────────────────
    │      ↗  ╲    Optimal path
1.7 ├─────●────╲──────────────────
    │    ↗      ╲
    └────┴───────┴────────────────→
        10⁹    10¹⁰    10¹¹      N
```

The optimal path traces the ridge where iso-compute lines are tangent to iso-loss curves.

## Implications for Distributed Training

### 1. Training Efficiency Matters More Than Model Size

A 7B model trained on 2T tokens often outperforms a 70B model trained on 200B tokens, despite 10× fewer parameters.

**For distributed systems**: Efficient data pipelines (high throughput) can be more valuable than scaling to more GPUs for larger models.

### 2. Compute-Optimal Models Are Memory-Hungry

Chinchilla-optimal training means more data passes through the model. This increases:
- Activation memory during forward pass
- Gradient accumulation requirements
- Data loading bandwidth needs

### 3. The Inference-Training Trade-off

Chinchilla-optimal models are expensive to serve: same quality, bigger model, more inference FLOPs.

Many practitioners deliberately **overtrain** smaller models:

$$L_{\text{overtrained}}(N_{\text{small}}, D_{\text{large}}) = L_{\text{optimal}}(N_{\text{large}}, D_{\text{optimal}})$$

LLaMA models train on 1-2T tokens, far exceeding Chinchilla ratios, to reduce serving costs.

## The Frontier Model Equation

Combining scaling laws with hardware:

$$\text{Time} = \frac{6ND}{\text{GPUs} \times \text{FLOPS/GPU} \times \text{MFU}}$$

Where MFU (Model FLOP Utilization) is typically 30-50%.

**Example**: Chinchilla (70B params, 1.4T tokens)
$$C = 6 \times 70 \times 10^9 \times 1.4 \times 10^{12} = 5.9 \times 10^{23} \text{ FLOPs}$$

On 1000 H100s at 40% MFU:
$$\text{Time} = \frac{5.9 \times 10^{23}}{1000 \times 10^{15} \times 0.4} \approx 17 \text{ days}$$

## Beyond Simple Scaling

Recent research suggests the surface is more complex:

### Data Quality
$$L(N, D, q) = \frac{A}{N^\alpha} + \frac{B}{(qD)^\beta} + L_\infty$$

Where $q$ is a data quality multiplier. High-quality data can shift the optimal ratio.

### Architecture Efficiency
Different architectures have different $A$ values. MoE models achieve lower loss at the same $N_{\text{active}}$.

### Emergent Capabilities
Some capabilities emerge suddenly at scale, not following smooth power laws. The surface has discontinuities.

## Exercises

1. **Optimal allocation**: Given $C = 10^{24}$ FLOPs and Chinchilla scaling ($\alpha = \beta = 0.5$, $A = B = 400$), calculate the optimal model size and token count.

2. **Training time**: You have 512 H100 GPUs at 45% MFU. How long to train the optimal model from Exercise 1?

3. **Overtraining analysis**: LLaMA-2 7B was trained on 2T tokens.
   - What's the Chinchilla-optimal token count for 7B parameters?
   - By what factor is it overtrained?
   - Estimate the loss difference between this and training 7B on optimal tokens.

4. **Iso-loss curve**: Derive the equation for an iso-loss curve $L(N, D) = L_0$ in the $(N, D)$ plane. What is its shape?

5. **Marginal returns**: At the current training point $(N_0, D_0)$, you can either double parameters or double data. Which reduces loss more? Derive the condition for indifference.

## Key Takeaways

1. **Loss follows power laws** in both parameters and data, enabling principled compute allocation.

2. **The 20:1 rule**: Chinchilla-optimal training uses ~20 tokens per parameter.

3. **Optimal allocation**: Both $N^*$ and $D^*$ scale as $C^{0.5}$—double compute, $\sqrt{2}\times$ each.

4. **Marginal balance**: At optimum, the last FLOP spent on parameters vs data yields equal loss reduction.

5. **Practical trade-offs**: Inference costs often push toward overtraining smaller models.
