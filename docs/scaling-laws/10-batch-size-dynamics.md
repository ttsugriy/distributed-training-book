---
title: "Batch Size and Learning Dynamics"
subtitle: "Critical Batch Size, Learning Rate Scaling, and LARS/LAMB"
---

<div class="chapter-opener" markdown>
Batch size is not just a memory parameter—it fundamentally affects learning dynamics. There's a critical batch size beyond which returns diminish rapidly. Understanding this is essential for scaling to thousands of GPUs.
</div>

<div class="investigation-question" markdown>
**The Question**: You want to scale from 8 GPUs to 10,000 GPUs. The naive approach: increase batch size 1,250×. But models trained with batch size 1M often fail to converge. What's the limit, and how do we push past it?
</div>

## Gradient Noise and Batch Size

Each minibatch provides a noisy estimate of the true gradient:

$$g_B = \frac{1}{B} \sum_{i=1}^{B} \nabla L(x_i) = \nabla L + \epsilon_B$$

Where $\epsilon_B$ is the noise with variance:

$$\text{Var}(\epsilon_B) = \frac{\sigma^2}{B}$$

Here $\sigma^2$ is the per-sample gradient variance (a property of the data and model). Larger batch → lower noise → more reliable gradient direction.

But here's the key insight: **noise isn't always bad**.

### The Beneficial Role of Noise

Gradient noise:
1. Helps escape sharp minima (which generalize poorly)
2. Provides implicit regularization
3. Enables exploration of loss landscape

Too little noise → may converge to sharp, non-generalizing minima.

## The Critical Batch Size

McCandlish et al. (2018) derived the critical batch size $B_{\text{crit}}$:

$$B_{\text{crit}} = \frac{G^2}{H}$$

Where:

- $G^2 = \mathbb{E}[||\nabla L||^2]$: expected gradient norm squared
- $H = \mathbb{E}[(\nabla L)^T \nabla^2 L (\nabla L)]$: curvature along gradient

Equivalently, using the noise scale:

$$B_{\text{crit}} = \frac{\text{tr}(\Sigma)}{\text{tr}(H)}$$

Where $\Sigma$ is the gradient covariance.

### Interpretation

- **$B < B_{\text{crit}}$**: Training is **noise-dominated**. Each step is small due to noisy gradients. Need many steps.

- **$B \approx B_{\text{crit}}$**: Optimal balance. Steps are reliable but not wasting compute.

- **$B > B_{\text{crit}}$**: Training is **curvature-dominated**. Extra samples provide diminishing returns. Compute is wasted.

### Empirical Values

Critical batch size varies by task and training stage:

| Domain | Typical $B_{\text{crit}}$ |
|--------|---------------------------|
| ImageNet (early training) | 2K - 8K |
| ImageNet (late training) | 16K - 64K |
| Language models (small) | 256 - 2K |
| Language models (large) | 2M - 8M |

Note: $B_{\text{crit}}$ **increases during training** as the model approaches a minimum and curvature decreases.

## The Perfect Scaling Law

Below $B_{\text{crit}}$, training scales perfectly:

$$\text{Steps}(B) = \frac{S_0 \cdot B_0}{B}$$

Where $S_0$ is steps at baseline batch $B_0$.

Doubling batch size → halving steps → same wall-clock time per step → 2× faster training.

**Total compute stays constant**:

$$C = S \cdot B \cdot (\text{FLOPs per sample}) = \text{constant}$$

## The Diminishing Returns Regime

Above $B_{\text{crit}}$, the relationship becomes:

$$\text{Steps}(B) = S_{\min} + \frac{S_{\text{noise}}}{B}$$

Where $S_{\min}$ is the minimum steps regardless of batch size (curvature limit).

As $B \to \infty$:

$$\text{Steps}(B) \to S_{\min}$$

You can't reduce steps below $S_{\min}$ no matter how large the batch.

**Compute waste**:

$$\text{Wasted compute} = (B - B_{\text{crit}}) \cdot S(B) \cdot (\text{FLOPs per sample})$$

## Learning Rate Scaling

When increasing batch size, you must adjust learning rate. The question is: how?

### Linear Scaling Rule (Goyal et al., 2017)

$$\eta(B) = \eta_0 \cdot \frac{B}{B_0}$$

**Intuition**: If batch size doubles, the gradient is twice as reliable, so we can take twice as large a step.

**Valid when**:

- $B \leq B_{\text{crit}}$
- Using SGD with momentum

**Derivation**: Consider the update over $k$ steps with batch $B_0$ vs 1 step with batch $kB_0$:

Small batch:

$$\Delta w = -\eta_0 \sum_{i=1}^k g_i \approx -k\eta_0 \bar{g}$$

Large batch:

$$\Delta w' = -\eta' \cdot g_{kB_0} \approx -\eta' \bar{g}$$

For equivalence: $\eta' = k\eta_0$

### Square Root Scaling

$$\eta(B) = \eta_0 \cdot \sqrt{\frac{B}{B_0}}$$

**Intuition**: The noise in the gradient scales as $1/\sqrt{B}$, so learning rate should scale with noise reduction.

**Valid when**:

- Beyond $B_{\text{crit}}$
- Loss landscape is more complex

**Derivation**: From the perspective of SGD convergence rate in convex optimization, the optimal learning rate is $\eta \propto 1/\sqrt{B}$ for noisy gradients.

### Which to Use?

| Regime | Scaling Rule |
|--------|-------------|
| $B \ll B_{\text{crit}}$ | Linear |
| $B \approx B_{\text{crit}}$ | Between linear and sqrt |
| $B > B_{\text{crit}}$ | Square root |
| $B \gg B_{\text{crit}}$ | Constant (no benefit to increasing) |

Practical approach: **linear scaling with warmup**, then reduce if instability.

## Warmup

Large learning rates at the start of training cause divergence. Solution: warmup.

### Linear Warmup

$$\eta(t) = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}$$

for $t \leq T_{\text{warmup}}$.

### Why Warmup Helps

Early in training:
1. Gradients are large and noisy
2. Loss landscape curvature is high
3. Model is far from any minimum

Large steps cause:

- Gradient explosion
- Catastrophic updates
- Divergence

Warmup allows the model to "find its footing" before taking large steps.

### Warmup Duration

Rule of thumb:

$$T_{\text{warmup}} \approx \frac{B}{B_0} \cdot T_0$$

Where $T_0$ is warmup steps at baseline batch.

For very large batches (>64K), longer warmup may be needed.

## Layer-wise Adaptive Learning Rates

Different layers have different gradient magnitudes. Standard learning rate works poorly for very deep or very wide networks at large batch sizes.

### LARS (You et al., 2017)

Layer-wise Adaptive Rate Scaling for SGD:

$$\eta_l = \eta \cdot \phi(||w_l||, ||\nabla w_l||)$$

Where the trust ratio is:

$$\phi = \frac{||w_l||}{||\nabla w_l|| + \lambda ||w_l||}$$

Here $\lambda$ is the weight decay coefficient (typically 0.0001 to 0.001).

**Intuition**: Scale the learning rate by the ratio of weight norm to gradient norm. Prevents any layer from updating too much relative to its current scale.

The update becomes:

$$w_l \leftarrow w_l - \eta \cdot \phi_l \cdot (\nabla w_l + \lambda w_l)$$

**LARS enabled training ImageNet with batch size 32K in 1 hour** (vs. days with standard SGD).

### LAMB (You et al., 2019)

Layer-wise Adaptive Moments for Batch training—combines LARS with Adam:

$$m_l^{(t)} = \beta_1 m_l^{(t-1)} + (1-\beta_1) \nabla w_l$$

$$v_l^{(t)} = \beta_2 v_l^{(t-1)} + (1-\beta_2) (\nabla w_l)^2$$

$$\hat{m}_l = \frac{m_l}{1-\beta_1^t}, \quad \hat{v}_l = \frac{v_l}{1-\beta_2^t}$$

$$r_l = \frac{\hat{m}_l}{\sqrt{\hat{v}_l} + \epsilon}$$

Where $\epsilon$ is a small constant for numerical stability (typically $10^{-8}$).

Then apply LARS-style trust ratio:

$$\phi_l = \frac{||w_l||}{||r_l + \lambda w_l||}$$

$$w_l \leftarrow w_l - \eta \cdot \phi_l \cdot (r_l + \lambda w_l)$$

**LAMB enabled training BERT with batch size 65K** in 76 minutes (vs. 3 days with Adam).

### Comparison

| Method | Base Optimizer | Max Batch (ImageNet) | Max Batch (BERT) |
|--------|---------------|---------------------|-----------------|
| SGD + Linear LR | SGD | ~8K | N/A |
| LARS | SGD | 32K | ~8K |
| Adam + Linear LR | Adam | ~16K | ~16K |
| LAMB | Adam | ~32K | 65K |

## The Batch Size vs. Time Trade-off

Larger batch sizes enable data parallelism across more GPUs. But returns diminish.

### Scaling Efficiency

Define scaling efficiency:

$$E(B) = \frac{S(B_0)/S(B)}{B/B_0}$$

- $E = 1$: Perfect scaling (linear speedup)
- $E < 1$: Sub-linear scaling
- $E \to 0$: Wasted compute

Below $B_{\text{crit}}$: $E \approx 1$

Above $B_{\text{crit}}$: $E$ drops rapidly

### When to Scale

**Scale batch** when:

- Wall-clock time is the constraint
- You're below $B_{\text{crit}}$
- GPU utilization is high

**Don't scale batch** when:

- Already above $B_{\text{crit}}$
- Final quality matters more than speed
- Hyperparameter tuning is difficult

## Gradient Accumulation

Can't fit large batch in memory? Accumulate gradients:

```python
optimizer.zero_grad()
for i, batch in enumerate(mini_batches):
    loss = model(batch) / accumulation_steps
    loss.backward()  # Accumulates gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Mathematically equivalent** to large batch, but:

- More forward/backward passes
- Same memory as small batch
- Slower than true data parallelism

Use when: GPU memory limits effective batch size, but $B_{\text{crit}}$ hasn't been reached.

## Dynamic Batch Sizing

$B_{\text{crit}}$ increases during training. Optimal strategy: **increase batch size during training**.

### LLAMA's Approach

LLaMA increased batch size mid-training:

- Start: batch size 2M tokens
- After $t$ steps: increase to 4M tokens

Benefits:
1. Early training: smaller batch for exploration
2. Late training: larger batch for efficiency
3. Total steps reduced

### Adaptive Scaling

Monitor gradient noise scale:

$$\text{noise scale} = \frac{||\text{Var}(\nabla L)||}{||\mathbb{E}[\nabla L]||^2}$$

When noise scale drops, safe to increase batch size.

## Practical Recipe for Large-Batch Training

1. **Establish baseline**: Train with small batch (256-1024), find optimal $\eta_0$

2. **Estimate $B_{\text{crit}}$**: Double batch, check if steps halve. Stop when they don't.

3. **Scale with linear rule**: $\eta = \eta_0 \cdot B/B_0$ up to $B_{\text{crit}}$

4. **Use warmup**: $T_{\text{warmup}} \propto B/B_0$

5. **Consider LARS/LAMB**: Essential for $B > 8K$ typically

6. **Monitor carefully**:

   - Loss spikes → reduce LR or increase warmup
   - Slow convergence → may have exceeded $B_{\text{crit}}$
   - Layer-wise gradient norms → check for imbalance

7. **Dynamic batch**: Increase batch size as training progresses

## Exercises

1. **Critical batch size**: A model trains in 100K steps with batch size 256. With batch size 1024, it trains in 28K steps. With batch size 4096, it trains in 15K steps. Estimate $B_{\text{crit}}$.

??? success "Solution"
    **Given data:**

    | Batch Size | Steps to Convergence |
    |------------|---------------------|
    | 256 | 100,000 |
    | 1,024 | 28,000 |
    | 4,096 | 15,000 |

    **Using the diminishing returns model:**

    $$S(B) = S_{\min} + \frac{S_{\text{noise}}}{B}$$

    **Check perfect scaling from B=256 to B=1024 (4× increase):**

    If scaling were perfect: $S(1024) = 100,000 / 4 = 25,000$

    Actual: 28,000 steps → already seeing diminishing returns

    **Set up equations:**

    From $S(256) = 100,000$:

    $$S_{\min} + \frac{S_{\text{noise}}}{256} = 100,000$$

    From $S(1024) = 28,000$:

    $$S_{\min} + \frac{S_{\text{noise}}}{1024} = 28,000$$

    **Solve:**

    Subtract second from first:

    $$S_{\text{noise}} \left(\frac{1}{256} - \frac{1}{1024}\right) = 72,000$$

    $$S_{\text{noise}} \times \frac{3}{1024} = 72,000$$

    $$S_{\text{noise}} = 24,576,000$$

    Substitute back:

    $$S_{\min} = 28,000 - \frac{24,576,000}{1024} = 28,000 - 24,000 = 4,000$$

    **Verify with B=4096:**
    $$S(4096) = 4,000 + \frac{24,576,000}{4096} = 4,000 + 6,000 = 10,000$$

    Actual: 15,000 steps (discrepancy suggests model is approximate)

    **Estimate $B_{\text{crit}}$:**

    The critical batch size is where noise and curvature contributions are equal:

    $$\frac{S_{\text{noise}}}{B_{\text{crit}}} = S_{\min}$$

    $$B_{\text{crit}} = \frac{S_{\text{noise}}}{S_{\min}} = \frac{24,576,000}{4,000} = \boxed{6,144}$$

    **Interpretation:** Batch sizes above ~6K will show significant diminishing returns. The data shows we're already past $B_{\text{crit}}$ at 4096, confirming the estimate is in the right range.

2. **Learning rate scaling**: You scale from batch 256 with $\eta = 0.001$ to batch 4096. What learning rate should you use under (a) linear scaling, (b) square root scaling?

??? success "Solution"
    **Given:**

    - Base batch: $B_0 = 256$
    - Base learning rate: $\eta_0 = 0.001$
    - Target batch: $B = 4096$
    - Scaling factor: $B/B_0 = 16$

    **(a) Linear scaling:**

    $$\eta = \eta_0 \times \frac{B}{B_0} = 0.001 \times 16 = \boxed{0.016}$$

    **(b) Square root scaling:**

    $$\eta = \eta_0 \times \sqrt{\frac{B}{B_0}} = 0.001 \times \sqrt{16} = 0.001 \times 4 = \boxed{0.004}$$

    **Which to use?**

    | Batch Size | Relative to $B_{\text{crit}}$ | Recommended Scaling |
    |------------|-------------------------------|---------------------|
    | 4,096 | Below 6K (from Exercise 1) | Linear (0.016) |
    | 8,192 | Above 6K | Square root |
    | 16,384 | Well above | Constant or sqrt |

    **Practical note:** Start with linear scaling (0.016) but use warmup. If training is unstable, fall back to square root (0.004).

3. **Compute efficiency**: With batch 512, training takes 50K steps. With batch 8192, training takes 6K steps. Calculate the scaling efficiency $E(8192)$.

??? success "Solution"
    **Given:**

    - Base batch: $B_0 = 512$, steps $S_0 = 50,000$
    - Target batch: $B = 8192$, steps $S = 6,000$

    **Scaling efficiency formula:**

    $$E(B) = \frac{S_0/S(B)}{B/B_0}$$

    **Calculate:**

    $$E(8192) = \frac{50,000/6,000}{8,192/512} = \frac{8.33}{16} = \boxed{0.52 = 52\%}$$

    **Interpretation:**

    | Metric | Value |
    |--------|-------|
    | Batch increase | 16× |
    | Step reduction | 8.33× |
    | Scaling efficiency | 52% |
    | "Wasted" compute | 48% |

    **Perfect scaling would give:**
    $$S_{\text{perfect}} = \frac{50,000}{16} = 3,125 \text{ steps}$$

    Actual: 6,000 steps → 1.92× more steps than perfect scaling.

    **Conclusion:** At batch 8192, we're well past $B_{\text{crit}}$. Almost half the compute is "wasted" in the sense that it doesn't reduce training steps. However, if wall-clock time is the constraint, this may still be worthwhile.

4. **LARS derivation**: Show that the LARS trust ratio $\phi = ||w||/||\nabla w||$ ensures that the relative update $||\Delta w||/||w||$ is approximately constant across layers.

??? success "Solution"
    **LARS update rule:**

    $$\Delta w_l = -\eta \cdot \phi_l \cdot \nabla w_l$$

    Where the trust ratio is:

    $$\phi_l = \frac{||w_l||}{||\nabla w_l||}$$

    (ignoring weight decay for simplicity)

    **Relative update magnitude:**

    $$\frac{||\Delta w_l||}{||w_l||} = \frac{\eta \cdot \phi_l \cdot ||\nabla w_l||}{||w_l||}$$

    $$= \frac{\eta \cdot \frac{||w_l||}{||\nabla w_l||} \cdot ||\nabla w_l||}{||w_l||}$$

    $$= \frac{\eta \cdot ||w_l||}{||w_l||}$$

    $$= \boxed{\eta}$$

    **Key insight:** The relative update $||\Delta w||/||w||$ equals $\eta$ for **all layers**, regardless of:

    - Weight magnitude $||w_l||$
    - Gradient magnitude $||\nabla w_l||$

    **Why this matters:**

    | Layer | Without LARS | With LARS |
    |-------|--------------|-----------|
    | Small weights, large gradients | Huge relative update | $\eta$ |
    | Large weights, small gradients | Tiny relative update | $\eta$ |
    | Any layer | $\eta \cdot ||\nabla w|| / ||w||$ | $\eta$ |

    **Consequence:** All layers update at the same relative rate, preventing:
    - Early layers from updating too slowly
    - Output layers from updating too aggressively
    - Training instability at large batch sizes

    This is why LARS enables training with batch sizes of 32K+ where standard SGD fails.

5. **Gradient accumulation**: You have 8 GPUs with batch 32 each (256 total) but need effective batch 2048. How many accumulation steps? If each forward-backward takes 100ms, and all-reduce takes 20ms, what's the time per effective step?

??? success "Solution"
    **Given:**

    - GPUs: 8
    - Per-GPU batch: 32
    - Current effective batch: $8 \times 32 = 256$
    - Target effective batch: 2,048
    - Forward-backward time: 100ms
    - All-reduce time: 20ms

    **Accumulation steps:**

    $$\text{Accumulation steps} = \frac{\text{Target batch}}{\text{Current batch}} = \frac{2048}{256} = \boxed{8}$$

    **Time breakdown per effective step:**

    | Phase | Count | Time Each | Total |
    |-------|-------|-----------|-------|
    | Forward-backward passes | 8 | 100ms | 800ms |
    | All-reduce (only at end) | 1 | 20ms | 20ms |
    | **Total** | | | **820ms** |

    **Comparison with true data parallelism (64 GPUs):**

    If we had 64 GPUs with batch 32 each:
    - Forward-backward: 100ms
    - All-reduce: ~40ms (slightly higher for more GPUs)
    - Total: ~140ms

    **Efficiency comparison:**

    $$\text{Accumulation slowdown} = \frac{820\text{ms}}{140\text{ms}} \approx 5.9\times$$

    **Key insight:** Gradient accumulation trades time for memory. It's useful when:
    1. $B_{\text{crit}}$ hasn't been reached yet
    2. GPU memory limits batch size
    3. More GPUs aren't available

6. **Dynamic batching**: You want to train for 1M tokens/step initially, ramping to 4M tokens/step. If you switch at the midpoint of training, how many fewer gradient updates do you perform compared to constant 1M tokens/step?

??? success "Solution"
    **Setup:**

    Let total training be $D$ tokens (e.g., 2T tokens).

    **Constant batch size (1M tokens/step):**

    $$\text{Updates}_{\text{constant}} = \frac{D}{1\text{M}} = D \text{ updates (in millions)}$$

    **Dynamic batch size:**

    - First half ($D/2$ tokens) at 1M tokens/step:

    $$\text{Updates}_1 = \frac{D/2}{1\text{M}} = \frac{D}{2\text{M}}$$

    - Second half ($D/2$ tokens) at 4M tokens/step:

    $$\text{Updates}_2 = \frac{D/2}{4\text{M}} = \frac{D}{8\text{M}}$$

    - Total dynamic updates:

    $$\text{Updates}_{\text{dynamic}} = \frac{D}{2\text{M}} + \frac{D}{8\text{M}} = \frac{4D + D}{8\text{M}} = \frac{5D}{8\text{M}}$$

    **Reduction:**

    $$\text{Reduction} = \frac{D}{\text{M}} - \frac{5D}{8\text{M}} = \frac{8D - 5D}{8\text{M}} = \frac{3D}{8\text{M}}$$

    $$\text{Reduction fraction} = \frac{3D/8}{D} = \boxed{37.5\%}$$

    **Concrete example with D = 2T tokens:**

    | Strategy | Updates | Wall-clock (relative) |
    |----------|---------|----------------------|
    | Constant 1M | 2,000,000 | 1.0× |
    | Dynamic (1M→4M) | 1,250,000 | ~0.75× |
    | Reduction | 750,000 | 25% faster |

    **Wait—why is wall-clock only 25% faster if we have 37.5% fewer updates?**

    The 4M batch steps take ~2× longer than 1M batch steps (more compute per step). But we gain on communication overhead being amortized over more tokens.

    **Actual wall-clock speedup** depends on:
    - Communication overhead fraction
    - GPU utilization at each batch size
    - Whether we're below $B_{\text{crit}}$ for both batch sizes

    **Summary:** Dynamic batching reduces gradient updates by 37.5%, translating to ~20-30% wall-clock speedup depending on system characteristics.

## Key Takeaways

1. **Critical batch size exists**: Beyond $B_{\text{crit}}$, compute is wasted.

2. **Linear scaling works below $B_{\text{crit}}$**: Learning rate $\propto$ batch size.

3. **Warmup is essential**: Larger batches need longer warmup.

4. **LARS/LAMB enable extreme scaling**: Layer-wise adaptation for 30K+ batch sizes.

5. **$B_{\text{crit}}$ increases during training**: Dynamic batch sizing can exploit this.

6. **Wall-clock vs. compute trade-off**: Larger batch is faster but less efficient above $B_{\text{crit}}$.
