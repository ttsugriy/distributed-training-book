---
title: "Phase Transitions in Scaling"
subtitle: "When Smooth Laws Break and Capabilities Emerge"
---

<div class="chapter-opener" markdown>
Scaling is not smooth. There are regimes where model capacity dominates, others where optimizer noise dominates, and transitions between them. Understanding these phases lets us predict when capabilities will emerge—and when throwing more compute won't help.
</div>

<div class="investigation-question" markdown>
**The Question**: A 10B parameter model can't do multi-step arithmetic. A 100B model can. Where did this capability come from? The loss curves are smooth, but the capability appeared suddenly. How do we explain—and predict—these phase transitions?
</div>

## The Puzzle of Emergence

Plot loss against compute: you get a smooth power law. Plot capability against compute: you often get a step function.

```
Loss                          Capability
  │                              │
  │╲                             │        ┌────
  │ ╲                            │        │
  │  ╲                           │        │
  │   ╲                          │────────┘
  │    ╲___                      │
  └────────────→ Compute         └────────────→ Compute
    Smooth decay                   Sharp transition
```

This is the **emergence puzzle**: smooth loss improvement hides discrete capability acquisition.

## Types of Phase Transitions

### Type 1: Capability Emergence

Some capabilities appear suddenly at scale:

| Capability | Approximate Threshold |
|------------|----------------------|
| In-context learning | ~1B parameters |
| Chain-of-thought reasoning | ~60B parameters |
| Multi-step arithmetic | ~100B parameters |
| Theory of mind | ~100B+ parameters |

These aren't gradual improvements. Below threshold: 0% accuracy. Above: 80%+ accuracy.

**Mathematical model**: Let capability $c$ depend on loss via:

$$P(c | L) = \text{sigmoid}\left(\frac{L_c - L}{\tau}\right) = \frac{1}{1 + e^{-(L_c - L)/\tau}}$$

Where:

- $L_c$: critical loss threshold for capability $c$
- $\tau$: transition sharpness

As $\tau \to 0$, the transition becomes a step function.

### Type 2: Training Dynamics Phases

The optimization process itself has phases:

**Phase 1: Random Guessing**
- Loss ≈ log(vocab_size)
- Model outputs uniform distribution
- Duration: first few hundred steps

**Phase 2: Unigram Learning**
- Model learns token frequencies
- Rapid initial loss drop
- Duration: ~1% of training

**Phase 3: Bigram/N-gram**
- Local correlations learned
- Slower improvement
- Duration: ~5% of training

**Phase 4: Semantic Learning**
- Long-range dependencies
- Power law regime
- Duration: bulk of training

**Phase 5: Memorization**
- Training loss continues dropping
- Validation loss plateaus
- Overfitting begins

### Type 3: Grokking

A phenomenon where:
1. Model memorizes training data (training loss → 0)
2. Validation loss stays high (no generalization)
3. Suddenly, after extended training, validation loss drops
4. Model has "grokked" the underlying pattern

```
Loss
  │
  │ Training     Validation
  │    ↓            ↓
  │    ●───────────────────
  │     ╲
  │      ╲    ●●●●●●╲
  │       ╲          ╲
  │        ╲____      ╲____
  │                 ↑
  └─────────────────┼────→ Steps
              Grokking point
```

Grokking suggests generalization happens via a distinct phase transition, not gradual improvement.

## The 4+3 Phase Model

Caballero et al. (2023) identified a richer phase structure in the $(D_{\text{data}}, D_{\text{task}})$ plane, where $D$ measures complexity.

### The Four Main Phases

**Phase I: Model-Capacity Limited**

The model is too small to represent the target function.

$$L \approx L_{\infty}(N) = \frac{A}{N^\alpha}$$

Loss is determined by model size, independent of data or optimization.

*Symptoms*:

- Adding more data doesn't help
- Longer training doesn't help
- Need a bigger model

**Phase II: Optimizer-Noise Limited**

Gradient noise prevents convergence to the true minimum.

$$L \approx L_{\text{opt}}(B, \eta) = \frac{\sigma^2}{B \cdot f(\eta)}$$

Where $\sigma^2$ is the variance of per-sample gradients, $B$ is batch size, $\eta$ is learning rate, and $f(\eta)$ captures learning rate effects.

*Symptoms*:

- Larger batch sizes help
- Learning rate tuning matters a lot
- Loss fluctuates around a floor

**Phase III: Data Limited**

Not enough data to learn the task.

$$L \approx \frac{B}{D^\beta}$$

*Symptoms*:

- More data directly reduces loss
- Model may be overfitting
- Validation loss >> training loss

**Phase IV: Compute-Optimal**

The balanced regime where Chinchilla optimality holds.

$$L = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty$$

*Symptoms*:

- Both model size and data matter
- Smooth power law scaling
- Optimal allocation is $D \approx 20N$

### The Three Transition Zones

Between phases, behavior is complex:

**I→IV Transition**: As $N$ increases, you move from capacity-limited to balanced.

**II→IV Transition**: As $B$ increases, optimizer noise decreases until data/compute limits dominate.

**III→IV Transition**: As $D$ increases, data limitation relaxes.

## Double Descent

A striking non-monotonic phenomenon:

```
Test Loss
    │
    │  ●
    │   ●        ●
    │    ●      ● ●
    │     ●    ●   ●
    │      ●  ●     ●
    │       ●●       ●────
    │         ↑
    └─────────┼──────────→ Model Size
         Interpolation
           threshold
```

**Classical regime** (small models): Bigger = better generalization

**Interpolation threshold** (model just fits training data): Overfitting peak

**Modern regime** (overparameterized): Bigger = better again

This occurs because:
1. Small models underfit → high bias
2. Medium models memorize → high variance
3. Large models find smooth interpolations → low variance

**For distributed training**: Don't stop at the interpolation threshold. Push through to the modern regime.

## Emergent Capabilities: A Deeper Look

### The Metric Matters

Wei et al. (2022) showed that emergence depends on how you measure:

| Metric | Appears Emergent? |
|--------|-------------------|
| Accuracy (exact match) | Yes |
| Token-level log-likelihood | No |
| Partial credit scoring | Sometimes |

With log-likelihood, capabilities improve smoothly. With exact-match accuracy, they appear suddenly.

**Implication**: The "emergence" may be an artifact of discrete evaluation metrics, not the model's internal representations.

### The Circuit Formation Hypothesis

Capabilities may emerge when internal "circuits" complete:

1. Individual components develop gradually
2. Circuit requires all components
3. Capability appears when last component forms

Like building a bridge: progress on foundations is invisible until the span connects.

### Predicting Emergence

Given current trends, when will capability $c$ appear?

**Method 1**: Extrapolate loss curve, estimate $L_c$

$$N_c \approx \left(\frac{A}{L_c - L_\infty}\right)^{1/\alpha}$$

**Method 2**: Use linear probes

Train a linear classifier on intermediate representations. When linear probe accuracy exceeds random:

- The representation contains the capability
- Full capability may emerge soon

**Method 3**: Partial capability metrics

Design graded evaluations. Look for smooth improvement that predicts discrete threshold.

## Implications for Distributed Training

### 1. Scale Planning

Know which phase you're in:

| Phase | Optimization Strategy |
|-------|----------------------|
| Capacity-limited | Scale model (more GPUs for TP/PP) |
| Data-limited | Improve data pipeline |
| Optimizer-limited | Tune hyperparameters, increase batch |
| Compute-optimal | Balanced scaling |

### 2. Checkpoint Strategy

Near phase transitions:

- Checkpoint more frequently
- Capabilities may appear between evaluations
- Don't stop just before a transition

### 3. Batch Size Dynamics

In optimizer-noise-limited phase:

- Larger batches help
- Can scale batch size with training

In data-limited phase:

- Batch size doesn't help loss
- But larger batch = faster iteration

### 4. Curriculum Effects

Some evidence that training order affects phase transitions:

- Easy examples first may accelerate Phase I→IV transition
- Hard examples early may delay grokking

### 5. Compute Allocation

If targeting a specific capability:

$$C_{\text{needed}} = 6 \cdot N_c \cdot D_{\text{opt}}(N_c)$$

Where $N_c$ is the capability threshold model size.

Underprovisioning compute below $C_{\text{needed}}$ wastes everything—you'll never reach the threshold.

## The Scaling Hypothesis

A strong form of the scaling hypothesis:

> Given enough compute, any capability will emerge.

Evidence for:

- Larger models consistently gain capabilities
- No capability has been found that doesn't eventually appear

Evidence against:

- Some capabilities may require architectural changes
- Data quality limits may be fundamental
- Compute/time may be practically infeasible

**Practical stance**: Assume capabilities will emerge, but plan for uncertainty in thresholds.

## Case Study: Arithmetic Capability

Tracking multi-digit addition across scales:

| Model Size | 2-digit | 3-digit | 4-digit | 5-digit |
|------------|---------|---------|---------|---------|
| 1B | 95% | 60% | 10% | 0% |
| 10B | 99% | 90% | 50% | 5% |
| 100B | 99% | 99% | 85% | 40% |
| 500B | 99% | 99% | 95% | 80% |

Each digit complexity has its own phase transition. More digits = higher threshold.

This suggests capabilities have **nested phase structures**: easier variants emerge first.

## Exercises

1. **Phase identification**: A 10B model trained on 200B tokens has training loss 2.1 and validation loss 2.3. Adding more data to 500B tokens reduces training loss to 2.0 but validation loss stays at 2.3. What phase is this model in?

??? success "Solution"
    **Observations:**

    | Condition | Training Loss | Validation Loss |
    |-----------|---------------|-----------------|
    | 200B tokens | 2.1 | 2.3 |
    | 500B tokens | 2.0 | 2.3 |

    **Key indicators:**

    1. Validation loss doesn't improve when adding more data
    2. Training loss continues to decrease (2.1 → 2.0)
    3. Gap between train/val loss exists and persists

    **Diagnosis: Phase I — Capacity-Limited**

    The model has insufficient capacity to generalize better. Adding data improves training loss (more optimization steps) but not validation loss (representation bottleneck).

    $$L_{\text{val}} \approx L_\infty(N) = \frac{A}{N^\alpha} + \text{irreducible}$$

    **Evidence against other phases:**

    | Phase | Expected Behavior | Observed? |
    |-------|-------------------|-----------|
    | Data-limited | More data → lower val loss | No ✗ |
    | Optimizer-limited | Large batch → lower loss | Not tested |
    | Compute-optimal | Both N and D matter | Only N matters |

    **Recommendation:** Scale model size (increase N) rather than data or training time.

2. **Emergence prediction**: A capability appears at 50% accuracy for 50B parameters. At 10B parameters, accuracy is 5%. Assuming the transition follows a sigmoid with $\tau = 0.1$ nats, estimate the loss threshold $L_c$ for this capability.

??? success "Solution"
    **Using the emergence model:**

    $$P(c | L) = \text{sigmoid}\left(\frac{L_c - L}{\tau}\right) = \frac{1}{1 + e^{-(L_c - L)/\tau}}$$

    **From the 50B data point (50% accuracy):**

    At 50% accuracy, the sigmoid argument is 0:

    $$\text{sigmoid}(0) = 0.5 \implies \frac{L_c - L_{50B}}{\tau} = 0$$

    $$\boxed{L_c = L_{50B}}$$

    **From the 10B data point (5% accuracy):**

    $$0.05 = \frac{1}{1 + e^{-(L_c - L_{10B})/\tau}}$$

    Solving for the exponent:

    $$1 + e^{-(L_c - L_{10B})/0.1} = 20$$
    $$e^{-(L_c - L_{10B})/0.1} = 19$$
    $$-(L_c - L_{10B})/0.1 = \ln(19) = 2.94$$
    $$L_c - L_{10B} = -0.294 \text{ nats}$$

    **Estimating the losses:**

    Using scaling law $L(N) \propto N^{-\alpha}$ with $\alpha \approx 0.34$:

    $$\frac{L_{10B} - L_\infty}{L_{50B} - L_\infty} = \left(\frac{50}{10}\right)^{0.34} = 5^{0.34} = 1.70$$

    If $L_{50B} = L_c$ and $L_{10B} = L_c + 0.294$:

    The capability threshold $L_c$ is approximately **the loss achieved by a 50B model** at the given training stage.

    **Interpretation:** To predict when capability emerges, extrapolate the loss curve and find when $L = L_c$.

3. **Double descent**: You're training a model and observe test loss increasing. You have compute to either (a) train longer or (b) scale up the model 2×. Which is more likely to help, and why?

??? success "Solution"
    **Analyzing the situation:**

    Test loss increasing indicates we're near the **interpolation threshold** (double descent peak).

    ```
    Test Loss
        │
        │   ●  ← You are here
        │  ● ●
        │ ●   ●
        │●     ●
        │       ●────
        └────────────→ Model Size / Training Time
    ```

    **Option (a): Train longer**

    - May worsen overfitting initially
    - Could eventually lead to grokking (if the task allows)
    - Risky: may never recover

    **Option (b): Scale model 2×**

    - Moves into the overparameterized regime
    - Large models find smoother solutions
    - More likely to reach "modern regime" of double descent

    **Recommendation: (b) Scale up the model 2×**

    | Option | Expected Outcome | Risk |
    |--------|------------------|------|
    | Train longer | Grokking possible but uncertain | May never recover |
    | Scale 2× | Likely enters modern regime | Compute cost |

    **Mathematical justification:**

    At interpolation threshold:

    $$N \approx D_{\text{train}}$$

    Scaling to $2N$ while keeping $D$ fixed:

    $$\frac{N_{\text{new}}}{D} = 2 \gg 1$$

    This moves firmly into the overparameterized regime where implicit regularization helps generalization.

4. **Capability targeting**: You need chain-of-thought reasoning, which emerges around 60B parameters. Your compute budget is $C = 10^{23}$ FLOPs. Is this sufficient? If not, what compute is needed?

??? success "Solution"
    **Compute required for 60B Chinchilla-optimal:**

    Using $D^* = 20N$ and $C = 6ND$:

    $$C_{\text{needed}} = 6 \times N \times 20N = 120N^2$$
    $$C_{\text{needed}} = 120 \times (60 \times 10^9)^2 = 120 \times 3.6 \times 10^{21}$$
    $$C_{\text{needed}} = 4.32 \times 10^{23} \text{ FLOPs}$$

    **Comparison:**

    | Budget | Required | Sufficient? |
    |--------|----------|-------------|
    | $10^{23}$ | $4.32 \times 10^{23}$ | **No** (4.3× short) |

    **Options with $C = 10^{23}$:**

    **Option 1: Smaller Chinchilla-optimal model**
    $$N^* = \sqrt{\frac{10^{23}}{120}} = 2.89 \times 10^{10} \approx 29\text{B}$$

    This is below the 60B threshold—**chain-of-thought won't emerge**.

    **Option 2: Undertrained 60B model**
    $$D = \frac{C}{6N} = \frac{10^{23}}{6 \times 60 \times 10^9} = 278\text{B tokens}$$

    Tokens/param = $278/60 = 4.6$ (severely undertrained)

    This might work if capability depends primarily on model size, not training.

    **Recommendation:**

    $$\boxed{C_{\text{needed}} = 4.32 \times 10^{23} \text{ FLOPs}}$$

    To guarantee chain-of-thought, budget ~4-5× more compute, or accept risk with undertrained 60B.

5. **Grokking detection**: How would you modify your training monitoring to detect grokking early? What metrics would you track?

??? success "Solution"
    **Grokking signature:**

    1. Training loss → 0 (memorization)
    2. Validation loss stays high
    3. Extended plateau
    4. Sudden validation loss drop (grokking)

    **Metrics to track:**

    | Metric | Purpose | Grokking Signal |
    |--------|---------|-----------------|
    | Train/val loss gap | Generalization | Large and stable before grokking |
    | Weight norm $\|W\|$ | Regularization progress | Decreasing during grokking |
    | Gradient norm $\|\nabla L\|$ | Optimization dynamics | Spikes before transition |
    | Hessian eigenvalues | Loss landscape | Sharpness decreases |
    | Linear probe accuracy | Internal representations | Improves before grokking |

    **Implementation:**

    ```python
    # Track these during training
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'generalization_gap': [],  # val - train
        'weight_norm': [],
        'gradient_norm': [],
        'probe_accuracy': [],  # linear probe on val set
    }

    # Grokking detection heuristics
    def detect_grokking_potential(metrics, window=1000):
        # Condition 1: Train loss very low, val loss high
        train_converged = metrics['train_loss'][-1] < 0.1
        val_high = metrics['val_loss'][-1] > 0.5

        # Condition 2: Gap stable for many steps
        gap_stable = std(metrics['generalization_gap'][-window:]) < 0.01

        # Condition 3: Weight norm still decreasing
        weight_decreasing = metrics['weight_norm'][-1] < metrics['weight_norm'][-window]

        return train_converged and val_high and weight_decreasing
    ```

    **Actionable recommendations:**

    1. **Don't stop early**: If train loss is low but val loss high, grokking may be imminent
    2. **Increase weight decay**: Can accelerate grokking
    3. **Checkpoint frequently**: Save models around potential transition
    4. **Use linear probes**: Early warning of representation learning

    **Key insight:** Grokking suggests memorization → generalization transition. Track weight dynamics, not just loss.

## Key Takeaways

1. **Smooth loss hides discrete capabilities**: Loss improves gradually; capabilities appear suddenly.

2. **Four phases exist**: Capacity-limited, optimizer-limited, data-limited, and compute-optimal.

3. **Double descent is real**: Don't stop at the interpolation threshold—push through.

4. **Emergence may be metric-dependent**: Use appropriate metrics to track progress toward capabilities.

5. **Know your phase**: Different phases require different optimization strategies.

6. **Plan for thresholds**: If targeting a capability, ensure sufficient compute to reach its threshold.
