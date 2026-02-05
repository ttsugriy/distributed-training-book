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

<div class="notation-banner" markdown>
**Notation in this chapter:** $\Psi$ = parameters, $D$ = training tokens, $C$ = total compute, $\alpha,\beta$ = scaling exponents. See [Notation](../appendices/notation.md).
</div>

!!! abstract "Building On: Part I Foundations"
    This part assumes familiarity with the **three walls** (memory, time, cost) from [Chapter 1](../foundations/01-scale-imperative.md), the **extended roofline** model from [Chapter 2](../foundations/02-extended-roofline.md), and the **estimation mindset** from [Chapter 6](../foundations/06-estimation-discipline.md). We'll now ask: given that we must distribute training, how do we allocate our compute budget optimally between model size and training data?

## The Empirical Discovery

In 2020, researchers at OpenAI made a remarkable observation: language model loss follows smooth, predictable power laws. Plot log-loss against log-parameters or log-tokens, and you get straight lines.

This isn't obvious. Complex systems often exhibit chaotic behavior. But neural language models, across many orders of magnitude, follow:

$$L(\Psi) = \left(\frac{\Psi_c}{\Psi}\right)^{\alpha_{\Psi}}$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}$$

Where $\Psi_c$, $D_c$ are critical scales and $\alpha_{\Psi}$, $\alpha_D$ are power law exponents.

## The Loss Surface

Combining both dependencies:

$$L(\Psi, D) = \left[\left(\frac{\Psi_c}{\Psi}\right)^{\alpha_{\Psi} / \alpha} + \left(\frac{D_c}{D}\right)^{\alpha_D / \alpha}\right]^\alpha + L_\infty$$

Or in the simpler additive form often used:

$$L(\Psi, D) = \frac{A}{\Psi^{\alpha}} + \frac{B}{D^{\beta}} + L_\infty$$

Where:

| Symbol | Meaning | Typical Value |
|--------|---------|---------------|
| $\Psi$ | Number of parameters | $10^6$ to $10^{12}$ |
| $D$ | Training tokens | $10^9$ to $10^{13}$ |
| $A$ | Parameter scaling constant | ~400 |
| $B$ | Data scaling constant | ~400 |
| $\alpha$ | Parameter exponent | 0.076 (Kaplan) / 0.34 (Chinchilla) |
| $\beta$ | Data exponent | 0.095 (Kaplan) / 0.28 (Chinchilla) |
| $L_\infty$ | Irreducible loss | ~1.69 nats |

The irreducible loss $L_\infty$ represents the entropy of natural language—even a perfect model can't predict the unpredictable.

## The Compute Constraint

Training compute is dominated by matrix multiplications. For a transformer:

$$C \approx 6\Psi D$$

**Derivation**: Each token passes through layers where:

- Forward pass: ~$2\Psi$ FLOPs (matrix multiply is $2 \times$ parameters)
- Backward pass: ~$4\Psi$ FLOPs (gradient computation + weight updates)
- Total per token: ~$6\Psi$ FLOPs

For $D$ tokens:

$$C = 6\Psi D \text{ FLOPs}$$

!!! note "Practice"
    If your budget is fixed, compute $(\Psi, D)$ pairs from $C = 6\Psi D$ first, then decide whether you are training for loss (Chinchilla) or for inference cost (overtrain smaller models).

This creates a **constraint surface** in $(\Psi, D, C)$ space. For fixed $C$, we get a hyperbola:

$$D = \frac{C}{6\Psi}$$

## Minimizing Loss Under Compute Constraint

**Problem**: Minimize $L(\Psi, D)$ subject to $C = 6\Psi D$

**Method**: Lagrange multipliers

The Lagrangian:

$$\mathcal{L} = \frac{A}{\Psi^\alpha} + \frac{B}{D^\beta} + \lambda(C - 6\Psi D)$$

Taking partial derivatives:

$$\frac{\partial \mathcal{L}}{\partial \Psi} = -\frac{A\alpha}{\Psi^{\alpha+1}} - 6\lambda D = 0$$

$$\frac{\partial \mathcal{L}}{\partial D} = -\frac{B\beta}{D^{\beta+1}} - 6\lambda \Psi = 0$$

From the first equation:

$$\lambda = -\frac{A\alpha}{6D\Psi^{\alpha+1}}$$

Substituting into the second:

$$\frac{B\beta}{D^{\beta+1}} = \frac{A\alpha \Psi}{D\Psi^{\alpha+1}} = \frac{A\alpha}{\Psi^\alpha D}$$

Therefore:

$$\frac{B\beta}{D^\beta} = \frac{A\alpha}{\Psi^\alpha}$$

This says: **at the optimum, the marginal contribution to loss reduction from parameters equals that from data**.

Rearranging:

$$\frac{\Psi^\alpha}{D^\beta} = \frac{A\alpha}{B\beta}$$

## The Optimal Allocation

Using $C = 6\Psi D$ to eliminate one variable. Substituting $D = C/(6\Psi)$ into the optimality condition $\Psi^\alpha / D^\beta = A\alpha/(B\beta)$:

$$\Psi^{\alpha+\beta} = \frac{A\alpha}{B\beta} \cdot \left(\frac{C}{6}\right)^\beta$$

$$\Psi^* = \left(\frac{A\alpha}{B\beta}\right)^{\frac{1}{\alpha+\beta}} \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}$$

Similarly, substituting $\Psi = C/(6D)$:

$$D^* = \left(\frac{B\beta}{A\alpha}\right)^{\frac{1}{\alpha+\beta}} \left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha+\beta}}$$

**Key insight**: Both $\Psi^*$ and $D^*$ are power laws in $C$.

If $\alpha \approx \beta$ (Chinchilla: $\alpha \approx 0.34$, $\beta \approx 0.28$):

$$\Psi^* \propto C^{0.45}, \quad D^* \propto C^{0.55}$$

The optimal ratio:

$$\frac{D^*}{\Psi^*} = \left(\frac{B\beta}{A\alpha}\right)^{\frac{2}{\alpha+\beta}} \left(\frac{C}{6}\right)^{\frac{\alpha-\beta}{\alpha+\beta}}$$

Note that the ratio depends (weakly) on $C$ unless $\alpha = \beta$. For Chinchilla parameters, the exponent $(\alpha - \beta)/(\alpha + \beta) \approx 0.10$ is small, so the ratio varies slowly with compute budget. Empirically, Chinchilla found that $D^*/\Psi^* \approx 20$ for the compute budgets they explored.

**The 20:1 Rule**: Optimal training uses ~20 tokens per parameter. This is an empirically observed ratio from Hoffmann et al. (2022), not a universal constant—it depends weakly on the compute budget $C$. See Chapter 8 for a thorough reconciliation.

## Kaplan vs Chinchilla

Two influential papers reached different conclusions:

| Paper | $\alpha$ | $\beta$ | Optimal $\Psi^* \propto$ | Optimal $D^* \propto$ | Tokens/Param |
|-------|----------|---------|----------------------|----------------------|--------------|
| Kaplan (2020) | 0.076 | 0.095 | $C^{0.73}$ | $C^{0.27}$ | ~1.7 |
| Chinchilla (2022) | 0.34 | 0.28 | $C^{0.50}$ | $C^{0.50}$ | ~20 |

*Note: The Kaplan optimal scaling exponents (0.73, 0.27) were empirically fit rather than derived from α and β. The theoretical derivation $\Psi^* \propto C^{\beta/(\alpha+\beta)}$ gives different values, suggesting different fitting methodologies.*

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
        10⁹    10¹⁰    10¹¹      \Psi
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

$$L_{\text{overtrained}}(\Psi_{\text{small}}, D_{\text{large}}) = L_{\text{optimal}}(\Psi_{\text{large}}, D_{\text{optimal}})$$

LLaMA models train on 1-2T tokens, far exceeding Chinchilla ratios, to reduce serving costs.

## The Frontier Model Equation

Combining scaling laws with hardware:

$$\text{Time} = \frac{6\Psi D}{\text{GPUs} \times \text{FLOPs/GPU} \times \text{MFU}}$$

Where MFU (Model FLOP Utilization) is typically 30-50%.

**Example**: Chinchilla (70B params, 1.4T tokens)

$$C = 6 \times 70 \times 10^9 \times 1.4 \times 10^{12} = 5.9 \times 10^{23} \text{ FLOPs}$$

On 1000 H100s (~989 TFLOP/s dense FP16/BF16 each) at 40% MFU:

$$\text{Time} = \frac{5.9 \times 10^{23}}{1000 \times 9.89 \times 10^{14} \times 0.4} = \frac{5.9 \times 10^{23}}{3.96 \times 10^{17}} \approx 1.49 \times 10^6 \text{ s} \approx 17.3 \text{ days}$$

## Beyond Simple Scaling

Recent research suggests the surface is more complex:

### Data Quality
$$L(\Psi, D, q) = \frac{A}{\Psi^\alpha} + \frac{B}{(qD)^\beta} + L_\infty$$

Where $q$ is a data quality multiplier. High-quality data can shift the optimal ratio.

### Architecture Efficiency
Different architectures have different $A$ values. MoE models achieve lower loss at the same $\Psi_{\text{active}}$.

### Emergent Capabilities
Some capabilities emerge suddenly at scale, not following smooth power laws. The surface has discontinuities.

## Exercises

1. **Optimal allocation**: Given $C = 10^{24}$ FLOPs and Chinchilla scaling ($\alpha = \beta = 0.5$, $A = B = 400$), calculate the optimal model size and token count.

??? success "Solution"
    **Using the 20:1 rule for Chinchilla-optimal allocation:**

    The optimal ratio is $D^*/\Psi^* \approx 20$.

    Substituting into the compute constraint $C = 6\Psi D$:

    $$C = 6 \times \Psi^* \times 20\Psi^* = 120(\Psi^*)^2$$

    **Solving for optimal model size:**
    $$\Psi^* = \sqrt{\frac{C}{120}} = \sqrt{\frac{10^{24}}{120}} = \sqrt{8.33 \times 10^{21}}$$

    $$\Psi^* = 2.89 \times 10^{10.5} \approx \boxed{91\text{B parameters}}$$

    **Optimal token count:**
    $$D^* = 20 \times \Psi^* = 20 \times 91 \times 10^9 = \boxed{1.82\text{T tokens}}$$

    **Verification:**
    $$C = 6 \times 91 \times 10^9 \times 1.82 \times 10^{12} = 9.94 \times 10^{23} \approx 10^{24} \checkmark$$

    | Parameter | Value |
    |-----------|-------|
    | Optimal $\Psi^*$ | 91B |
    | Optimal $D^*$ | 1.82T |
    | Tokens/parameter | 20 |

2. **Training time**: You have 512 H100 GPUs at 45% MFU. How long to train the optimal model from Exercise 1?

??? success "Solution"
    **Using the training time formula:**

    $$T = \frac{C}{\text{GPUs} \times \text{Peak FLOP/s} \times \text{MFU}}$$

    **Given:**

    - $C = 10^{24}$ FLOPs
    - GPUs = 512
    - H100 peak = 989 TFLOP/s = $9.89 \times 10^{14}$ FLOP/s
    - MFU = 45% = 0.45

    **Calculation:**

    $$T = \frac{10^{24}}{512 \times 9.89 \times 10^{14} \times 0.45}$$

    $$T = \frac{10^{24}}{4.56 \times 10^{17}} = 2.19 \times 10^6 \text{ seconds}$$

    **Converting to days:**
    $$T = \frac{2.19 \times 10^6}{3600 \times 24} = \boxed{25.4 \text{ days}}$$

    **Practical considerations:**

    | Factor | Impact |
    |--------|--------|
    | Checkpointing overhead | Add ~5-10% |
    | Hardware failures | Plan for ~10% downtime |
    | Realistic timeline | ~30-32 days |

3. **Overtraining analysis**: LLaMA-2 7B was trained on 2T tokens.
   - What's the Chinchilla-optimal token count for 7B parameters?
   - By what factor is it overtrained?
   - Estimate the loss difference between this and training 7B on optimal tokens.

??? success "Solution"
    **Part 1: Chinchilla-optimal token count**

    Using the 20:1 rule:

    $$D^* = 20 \times \Psi = 20 \times 7 \times 10^9 = \boxed{140\text{B tokens}}$$

    **Part 2: Overtraining factor**

    $$\text{Overtraining factor} = \frac{D_{\text{actual}}}{D^*} = \frac{2 \times 10^{12}}{140 \times 10^9} = \boxed{14.3\times}$$

    **Part 3: Loss difference estimation**

    Using $L(\Psi, D) = \frac{A}{\Psi^\alpha} + \frac{B}{D^\beta} + L_\infty$ with Chinchilla exponents ($\alpha = 0.34$, $\beta = 0.28$):

    The data-dependent term improvement:

    $$\Delta L_{\text{data}} = B\left(\frac{1}{D_{\text{opt}}^\beta} - \frac{1}{D_{\text{overtrain}}^\beta}\right)$$

    Ratio of data terms:

    $$\frac{(140\text{B})^{0.28}}{(2\text{T})^{0.28}} = \left(\frac{140}{2000}\right)^{0.28} = 0.07^{0.28} \approx 0.47$$

    The overtrained model achieves ~53% reduction in the data-dependent loss term compared to Chinchilla-optimal (the ratio 0.47 means paying only 47% of the data penalty).

    **Estimated improvement: ~0.1-0.2 nats lower loss** from overtraining.

    **Key insight**: Overtraining trades training compute for inference efficiency. LLaMA-2 7B performs comparably to a ~15-20B Chinchilla-optimal model while being 2-3× cheaper to serve.

4. **Iso-loss curve**: Derive the equation for an iso-loss curve $L(\Psi, D) = L_0$ in the $(\Psi, D)$ plane. What is its shape?

??? success "Solution"
    **Starting from the loss equation:**

    $$L_0 = \frac{A}{\Psi^\alpha} + \frac{B}{D^\beta} + L_\infty$$

    **Rearranging for the iso-loss curve:**

    Let $L' = L_0 - L_\infty$ (the reducible loss):

    $$L' = \frac{A}{\Psi^\alpha} + \frac{B}{D^\beta}$$

    **Solving for $D$ as a function of $\Psi$:**

    $$\frac{B}{D^\beta} = L' - \frac{A}{\Psi^\alpha}$$

    $$D^\beta = \frac{B}{L' - A/\Psi^\alpha} = \frac{B\Psi^\alpha}{L'\Psi^\alpha - A}$$

    $$\boxed{D = \left(\frac{B\Psi^\alpha}{L'\Psi^\alpha - A}\right)^{1/\beta}}$$

    **Shape analysis:**

    | Property | Value |
    |----------|-------|
    | Vertical asymptote | $\Psi = (A/L')^{1/\alpha}$ |
    | Horizontal asymptote | $D = (B/L')^{1/\beta}$ |
    | Shape | Hyperbola-like curve in first quadrant |
    | Curvature | Convex toward origin |

    **Geometric interpretation:**

    - For fixed loss $L_0$, there's a family of $(\Psi, D)$ pairs that achieve it
    - Larger models need less data to reach the same loss (and vice versa)
    - The curve asymptotes show the minimum resources needed even with infinite investment in the other dimension

5. **Marginal returns**: At the current training point $(\Psi_0, D_0)$, you can either double parameters or double data. Which reduces loss more? Derive the condition for indifference.

??? success "Solution"
    **Loss reduction from doubling $\Psi$:**

    $$\Delta L_\Psi = \frac{A}{\Psi_0^\alpha} - \frac{A}{(2\Psi_0)^\alpha} = \frac{A}{\Psi_0^\alpha}\left(1 - \frac{1}{2^\alpha}\right) = \frac{A}{\Psi_0^\alpha}(1 - 2^{-\alpha})$$

    **Loss reduction from doubling $D$:**

    $$\Delta L_D = \frac{B}{D_0^\beta} - \frac{B}{(2D_0)^\beta} = \frac{B}{D_0^\beta}(1 - 2^{-\beta})$$

    **Indifference condition ($\Delta L_\Psi = \Delta L_D$):**

    $$\frac{A}{\Psi_0^\alpha}(1 - 2^{-\alpha}) = \frac{B}{D_0^\beta}(1 - 2^{-\beta})$$

    $$\boxed{\frac{A(1 - 2^{-\alpha})}{\Psi_0^\alpha} = \frac{B(1 - 2^{-\beta})}{D_0^\beta}}$$

    **For Chinchilla ($\alpha = \beta$, $A = B$):**

    The condition simplifies to:

    $$\Psi_0^\alpha = D_0^\alpha \implies \Psi_0 = D_0$$

    But the 20:1 rule ($D^*/\Psi^* = 20$) suggests $A \neq B$. The actual indifference point is at the Chinchilla optimum where marginal returns are equal.

    **Decision rule:**

    | Condition | Action |
    |-----------|--------|
    | $\frac{A(1-2^{-\alpha})}{\Psi_0^\alpha} > \frac{B(1-2^{-\beta})}{D_0^\beta}$ | Double parameters |
    | $\frac{A(1-2^{-\alpha})}{\Psi_0^\alpha} < \frac{B(1-2^{-\beta})}{D_0^\beta}$ | Double data |
    | Equal | Either choice equivalent |

    **Practical implication:** If you're at a Chinchilla-optimal point, doubling either has equal marginal benefit. Most pre-2022 models were undertrained on data, making doubling $D$ more valuable.

## Key Takeaways

1. **Loss follows power laws** in both parameters and data, enabling principled compute allocation.

2. **The 20:1 rule**: Chinchilla-optimal training uses ~20 tokens per parameter.

3. **Optimal allocation**: Both $\Psi^*$ and $D^*$ scale as $C^{0.5}$—double compute, $\sqrt{2}\times$ each.

4. **Marginal balance**: At optimum, the last FLOP spent on parameters vs data yields equal loss reduction.

5. **Practical trade-offs**: Inference costs often push toward overtraining smaller models.
