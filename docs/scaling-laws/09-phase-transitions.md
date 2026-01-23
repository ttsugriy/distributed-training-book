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

$$P(c | L) = \sigma\left(\frac{L_c - L}{\tau}\right)$$

Where:
- $L_c$: critical loss threshold for capability $c$
- $\tau$: transition sharpness
- $\sigma$: sigmoid function

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

Where $B$ is batch size and $\eta$ is learning rate.

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

2. **Emergence prediction**: A capability appears at 50% accuracy for 50B parameters. At 10B parameters, accuracy is 5%. Assuming the transition follows a sigmoid with $\tau = 0.1$ nats, estimate the loss threshold $L_c$ for this capability.

3. **Double descent**: You're training a model and observe test loss increasing. You have compute to either (a) train longer or (b) scale up the model 2×. Which is more likely to help, and why?

4. **Capability targeting**: You need chain-of-thought reasoning, which emerges around 60B parameters. Your compute budget is $C = 10^{23}$ FLOPs. Is this sufficient? If not, what compute is needed?

5. **Grokking detection**: How would you modify your training monitoring to detect grokking early? What metrics would you track?

## Key Takeaways

1. **Smooth loss hides discrete capabilities**: Loss improves gradually; capabilities appear suddenly.

2. **Four phases exist**: Capacity-limited, optimizer-limited, data-limited, and compute-optimal.

3. **Double descent is real**: Don't stop at the interpolation threshold—push through.

4. **Emergence may be metric-dependent**: Use appropriate metrics to track progress toward capabilities.

5. **Know your phase**: Different phases require different optimization strategies.

6. **Plan for thresholds**: If targeting a capability, ensure sufficient compute to reach its threshold.
