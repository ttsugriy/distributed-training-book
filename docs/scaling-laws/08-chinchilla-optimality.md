---
title: "Chinchilla Optimality"
subtitle: "The 20:1 Ratio and Compute-Optimal Training"
---

<div class="chapter-opener" markdown>
For years, the field scaled models without enough data. Chinchilla revealed the optimal balance: approximately 20 tokens per parameter.
</div>

<div class="investigation-question" markdown>
**The Question**: GPT-3 has 175B parameters but trained on only 300B tokens—a ratio of 1.7:1. Chinchilla, with 70B parameters and 1.4T tokens (20:1), achieved the same loss with 4× less compute. What went wrong, and how do we compute "optimal"?
</div>

## The Pre-Chinchilla Era

Before 2022, the dominant scaling strategy was: **make the model bigger**.

This belief came from Kaplan et al. (2020), which suggested:

- Model size should scale as $N \propto C^{0.73}$
- Data should scale as $D \propto C^{0.27}$

Under this prescription, doubling compute meant:

- 1.66× more parameters
- 1.20× more tokens

The result: increasingly undertrained models.

| Model | Parameters | Tokens | Tokens/Param |
|-------|------------|--------|--------------|
| GPT-2 | 1.5B | 40B | 26.7 |
| GPT-3 | 175B | 300B | 1.7 |
| Gopher | 280B | 300B | 1.1 |
| Megatron-Turing | 530B | 270B | 0.5 |

Each model was larger than its predecessor but trained on roughly the same data. By Megatron-Turing, models saw each token only 0.5 times on average.

## The Chinchilla Methodology

Hoffmann et al. (2022) approached the problem differently. They trained 400+ models across a wide range of sizes (70M to 16B) and token counts, **each to convergence**.

Three independent methods converged on the same answer:

### Method 1: Fixed Model, Varying Data

For each model size $N$, fit:
$$L(D) = \frac{B}{D^\beta} + L_\infty(N)$$

Extract $L_\infty(N)$ for each size, then fit:
$$L_\infty(N) = \frac{A}{N^\alpha}$$

### Method 2: IsoFLOP Curves

Fix compute budget $C$. Train many $(N, D)$ pairs satisfying $C = 6ND$.
Plot loss vs $N$, find minimum.

For each $C$, there's an optimal $N^*(C)$:

```
Loss
  │
  │  ●                              C = 10²¹
  │   ●  ●
  │      ↘  ●
  │         ↘ ● ← Optimal N
  │            ● ●
  │               ●
  └──────────────────────────────→ N
```

### Method 3: Parametric Fitting

Fit all data simultaneously to:
$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

With Lagrange constraint $C = 6ND$, derive optimal scaling.

**All three methods agreed**: $N^* \propto C^{0.50}$, $D^* \propto C^{0.50}$.

## The 20:1 Ratio Derivation

From the optimal allocation (Chapter 7):

$$\frac{D^*}{N^*} = \frac{B\beta}{A\alpha}$$

Substituting Chinchilla's fitted values:

- $A = 406.4$, $\alpha = 0.34$
- $B = 410.7$, $\beta = 0.28$

$$\frac{D^*}{N^*} = \frac{410.7 \times 0.28}{406.4 \times 0.34} = \frac{115.0}{138.2} \approx 0.83$$

Wait—that's less than 1, not 20. What gives?

**The resolution**: The values above are for the multiplicative form. The commonly quoted "20:1" comes from a different parameterization and fitting procedure.

More precisely, from the paper:
$$N_{\text{opt}} = 0.6225 \cdot C^{0.4957}$$
$$D_{\text{opt}} = 1.8421 \cdot C^{0.5043}$$

At $C = 10^{21}$ FLOPs:

- $N_{\text{opt}} \approx 1.9 \times 10^{10}$ (19B)
- $D_{\text{opt}} \approx 4.0 \times 10^{11}$ (400B)
- Ratio: $D/N \approx 21$

The **20:1 rule** is a useful approximation:
$$D_{\text{optimal}} \approx 20 \cdot N$$

## Computing Optimal Allocations

Given compute budget $C$:

**Step 1**: Estimate optimal parameters
$$N^* \approx \sqrt{\frac{C}{6 \times 20}} = \sqrt{\frac{C}{120}}$$

**Step 2**: Estimate optimal tokens
$$D^* \approx 20 \cdot N^*$$

**Worked Example**: $C = 10^{24}$ FLOPs (≈GPT-4 training budget)

$$N^* \approx \sqrt{\frac{10^{24}}{120}} = \sqrt{8.33 \times 10^{21}} \approx 2.9 \times 10^{10}$$

$$D^* \approx 20 \times 2.9 \times 10^{10} = 5.8 \times 10^{11}$$

**Compute-optimal**: ~29B parameters, ~580B tokens.

Compare to GPT-4 rumors: 1.8T parameters (mixture-of-experts), trained on 13T tokens. This is heavily **overtrained** relative to Chinchilla—deliberately so for inference efficiency.

## The Undertrained Models

How "wrong" were pre-Chinchilla models?

### GPT-3 Analysis

- Parameters: $N = 175 \times 10^9$
- Tokens: $D = 300 \times 10^9$
- Compute: $C = 6ND = 3.15 \times 10^{23}$

Chinchilla-optimal for this compute:
$$N^* = \sqrt{\frac{3.15 \times 10^{23}}{120}} \approx 51B$$
$$D^* = 20 \times 51B = 1.02T$$

GPT-3 used **3.4× too many parameters** and **3.4× too few tokens**.

The loss penalty:
$$\Delta L \approx \frac{A}{N_{\text{GPT-3}}^\alpha} + \frac{B}{D_{\text{GPT-3}}^\beta} - \frac{A}{N^{*\alpha}} - \frac{B}{D^{*\beta}}$$

Chinchilla achieved GPT-3's loss with ~4× less compute.

### Gopher Analysis

- Parameters: $N = 280 \times 10^9$
- Tokens: $D = 300 \times 10^9$
- Compute: $C = 5.04 \times 10^{23}$

Chinchilla-optimal:
$$N^* \approx 65B, \quad D^* \approx 1.3T$$

Gopher was **4.3× overparameterized**.

## The Chinchilla Trap

Chinchilla optimality minimizes loss per FLOP. But this isn't always the right objective.

### The Inference Cost Problem

A Chinchilla-optimal 70B model with loss $L$ requires 70B multiply-adds per token at inference.

An overtrained 7B model (trained on 10× more data than Chinchilla-optimal) might achieve the same loss $L$ with **10× fewer inference FLOPs**.

Total cost comparison:

| Approach | Training Cost | Inference Cost (per token) |
|----------|--------------|---------------------------|
| Chinchilla 70B | $C$ | $70B$ MACs |
| Overtrained 7B | $1.5C$ | $7B$ MACs |

If you serve >$10^{13}$ tokens, overtraining pays off.

### The LLaMA Philosophy

LLaMA 1 and 2 deliberately overtrained:

| Model | Parameters | Tokens | Tokens/Param | vs Chinchilla |
|-------|------------|--------|--------------|---------------|
| LLaMA-7B | 7B | 1T | 143 | 7× overtrained |
| LLaMA-13B | 13B | 1T | 77 | 3.8× overtrained |
| LLaMA-65B | 65B | 1.4T | 22 | ~Chinchilla |
| LLaMA-2 7B | 7B | 2T | 286 | 14× overtrained |

The 7B model trains on 1-2T tokens—**7-14× more than Chinchilla optimal**—to minimize serving costs.

### Data Quality Considerations

The 20:1 rule assumes infinite homogeneous data. In practice:

1. **High-quality data is limited**: Once you've exhausted quality sources, more tokens ≠ better
2. **Repetition hurts**: Training on repeated data has diminishing returns
3. **Synthetic data**: Can extend effective dataset size but may have different scaling properties

## When to Deviate from Chinchilla

### Scenario 1: Inference-Dominated Workloads

If inference cost >> training cost, overtrain smaller models.

**Break-even analysis**: Let $r = \frac{\text{inference tokens}}{\text{training tokens}}$

Overtraining by factor $k$ (using $kD$ tokens on $N/k$ parameters) is profitable when:

$$r > \frac{k^2 \cdot \text{training cost per token}}{\text{inference cost per param}}$$

For typical ratios, overtraining pays off when serving >$10^{13}$ tokens.

### Scenario 2: Data Scarcity

If high-quality data is exhausted at $D_{\text{max}}$:

$$N^* = \frac{D_{\text{max}}}{20}$$

Training a larger model wastes compute on undertrained parameters.

### Scenario 3: Capability Thresholds

Some capabilities emerge at specific model sizes, regardless of training tokens.

Chain-of-thought reasoning appears around 60B+ parameters. If you need this capability, a 70B × 200B (undertrained) may outperform 7B × 2T (overtrained) for reasoning tasks.

### Scenario 4: Time Constraints

Training time scales with tokens. If wall-clock time is the constraint:

$$\text{Time} \propto \frac{D}{\text{throughput}}$$

Fewer tokens = faster training. An undertrained large model may be optimal for "get something working quickly."

## Post-Chinchilla Scaling Laws

Recent work has refined and extended Chinchilla:

### Compute-Optimal vs Downstream-Optimal

Chinchilla optimizes pretraining loss. But downstream task performance may have different scaling:

$$\text{Accuracy}(N, D) \neq f(L(N, D))$$

Some tasks benefit more from model size; others from data.

### Mixture-of-Experts Scaling

MoE models have different scaling:

- $N_{\text{total}}$ vs $N_{\text{active}}$
- Only active parameters contribute to FLOPs per token
- More total parameters can improve loss at fixed compute

Scaling law for MoE:
$$L(N_{\text{active}}, N_{\text{total}}, D) = \frac{A}{N_{\text{total}}^{\alpha_1} N_{\text{active}}^{\alpha_2}} + \frac{B}{D^\beta} + E$$

### Repeat Tokens

What if $D > D_{\text{unique}}$? Muennighoff et al. (2023) showed:

$$L(N, D, r) = L(N, D_{\text{unique}}) + \gamma \cdot \log(r)$$

Where $r = D / D_{\text{unique}}$ is the repetition ratio.

4 epochs ≈ 4× the unique data is often acceptable; beyond 16 epochs, returns diminish rapidly.

## Exercises

1. **Compute-optimal calculation**: You have $C = 10^{22}$ FLOPs. Calculate the Chinchilla-optimal model size and token count.

??? success "Solution"
    **Using the Chinchilla 20:1 rule:**

    From $C = 6ND$ and $D^* = 20N^*$:

    $$C = 6 \times N^* \times 20N^* = 120(N^*)^2$$

    **Solving for optimal model size:**
    $$N^* = \sqrt{\frac{C}{120}} = \sqrt{\frac{10^{22}}{120}} = \sqrt{8.33 \times 10^{19}}$$

    $$N^* = 9.13 \times 10^9 \approx \boxed{9.1\text{B parameters}}$$

    **Optimal token count:**
    $$D^* = 20 \times N^* = 20 \times 9.1 \times 10^9 = \boxed{182\text{B tokens}}$$

    **Verification:**
    $$C = 6 \times 9.1 \times 10^9 \times 182 \times 10^9 = 9.94 \times 10^{21} \approx 10^{22} \checkmark$$

    | Parameter | Value |
    |-----------|-------|
    | Compute budget | $10^{22}$ FLOPs |
    | Optimal $N^*$ | 9.1B |
    | Optimal $D^*$ | 182B |
    | Tokens/parameter | 20 |

2. **Undertrained analysis**: A model has 30B parameters and was trained on 150B tokens.
   - What's the compute used?
   - What's the Chinchilla-optimal allocation for that compute?
   - By what factor is the model over/underparameterized?

??? success "Solution"
    **Part 1: Compute used**

    $$C = 6ND = 6 \times 30 \times 10^9 \times 150 \times 10^9 = \boxed{2.7 \times 10^{22} \text{ FLOPs}}$$

    **Part 2: Chinchilla-optimal allocation**

    $$N^* = \sqrt{\frac{C}{120}} = \sqrt{\frac{2.7 \times 10^{22}}{120}} = \sqrt{2.25 \times 10^{20}}$$

    $$N^* = 1.5 \times 10^{10} = \boxed{15\text{B parameters}}$$

    $$D^* = 20 \times N^* = 20 \times 15 \times 10^9 = \boxed{300\text{B tokens}}$$

    **Part 3: Over/underparameterization factor**

    $$\text{Overparameterization} = \frac{N_{\text{actual}}}{N^*} = \frac{30\text{B}}{15\text{B}} = \boxed{2\times \text{ overparameterized}}$$

    $$\text{Undertraining} = \frac{D^*}{D_{\text{actual}}} = \frac{300\text{B}}{150\text{B}} = 2\times \text{ undertrained on data}$$

    **Summary:**

    | Metric | Actual | Optimal | Ratio |
    |--------|--------|---------|-------|
    | Parameters | 30B | 15B | 2× over |
    | Tokens | 150B | 300B | 2× under |
    | Tokens/param | 5 | 20 | 4× below optimal |

    The model is severely undertrained: it has 4× fewer tokens per parameter than Chinchilla-optimal.

3. **Inference break-even**: A 70B Chinchilla-optimal model costs $10M to train. A 7B overtrained model achieving similar loss costs $15M to train. Inference cost is $0.001 per 1M tokens for 70B, $0.0001 per 1M tokens for 7B. How many tokens must you serve before overtraining is profitable?

??? success "Solution"
    **Total cost model:**

    $$C_{\text{total}} = C_{\text{train}} + C_{\text{inference}} \times T$$

    Where $T$ is tokens served (in millions).

    **For 70B Chinchilla model:**
    $$C_{70B} = \$10\text{M} + \$0.001 \times T$$

    **For 7B overtrained model:**
    $$C_{7B} = \$15\text{M} + \$0.0001 \times T$$

    **Break-even condition:**
    $$\$10\text{M} + \$0.001 \times T = \$15\text{M} + \$0.0001 \times T$$

    $$\$0.001T - \$0.0001T = \$5\text{M}$$

    $$\$0.0009 \times T = \$5\text{M}$$

    $$T = \frac{\$5 \times 10^6}{\$0.0009} = 5.56 \times 10^9 \text{ million tokens}$$

    $$T = \boxed{5.56 \times 10^{15} \text{ tokens} \approx 5.6 \text{ quadrillion tokens}}$$

    **Interpretation:**

    | Tokens Served | Cheaper Option |
    |---------------|----------------|
    | < 5.6 quadrillion | 70B Chinchilla |
    | > 5.6 quadrillion | 7B overtrained |

    For context, ChatGPT reportedly serves ~100T tokens/day. At that rate:

    $$\text{Break-even time} = \frac{5.6 \times 10^{15}}{100 \times 10^{12}} \approx 56 \text{ days}$$

    For high-volume inference, overtraining pays off quickly.

4. **Data budget**: You have exactly 500B high-quality tokens. What's the largest model you should train?

??? success "Solution"
    **Using the Chinchilla 20:1 rule in reverse:**

    If $D_{\text{max}} = 500\text{B}$ tokens and optimal ratio is $D^*/N^* = 20$:

    $$N^* = \frac{D_{\text{max}}}{20} = \frac{500 \times 10^9}{20} = \boxed{25\text{B parameters}}$$

    **Compute required:**
    $$C = 6ND = 6 \times 25 \times 10^9 \times 500 \times 10^9 = 7.5 \times 10^{22} \text{ FLOPs}$$

    **Why not larger?**

    | Model Size | Issue |
    |------------|-------|
    | > 25B | Undertrained (tokens/param < 20) |
    | < 25B | Wasted data capacity |
    | = 25B | Chinchilla-optimal for data budget |

    **Alternative: Accept undertraining**

    If you train a 50B model on 500B tokens:

    - Tokens/param = 10 (half of optimal)
    - ~20% higher loss than 25B model trained on same data
    - But larger model may have emergent capabilities

    **Recommendation:** 25B is optimal for loss; larger sizes trade loss for capability.

5. **MoE analysis**: A dense 70B model and a MoE with 70B active / 1T total parameters both train on 1.4T tokens. Which achieves lower loss? (Assume MoE gets ~1.5× the loss reduction per parameter from total vs active)

??? success "Solution"
    **Dense 70B model:**

    Using $L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty$:

    $$L_{\text{dense}} \propto \frac{1}{(70\text{B})^\alpha}$$

    **MoE model analysis:**

    The MoE has:
    - Active parameters: $N_{\text{active}} = 70\text{B}$
    - Total parameters: $N_{\text{total}} = 1\text{T}$

    With 1.5× loss reduction from total parameters:

    $$L_{\text{MoE}} \propto \frac{1}{(N_{\text{active}})^\alpha \cdot (N_{\text{total}}/N_{\text{active}})^{\alpha/2}}$$

    The effective parameter count for loss scaling:

    $$N_{\text{eff}} = N_{\text{active}} \cdot \left(\frac{N_{\text{total}}}{N_{\text{active}}}\right)^{0.5} = 70\text{B} \cdot \left(\frac{1000\text{B}}{70\text{B}}\right)^{0.5}$$

    $$N_{\text{eff}} = 70\text{B} \cdot 3.78 = 265\text{B}$$

    **Loss comparison:**

    | Model | Effective $N$ | Relative Loss Term |
    |-------|---------------|-------------------|
    | Dense 70B | 70B | $(70\text{B})^{-0.34} = 1.00$ |
    | MoE 70B/1T | 265B | $(265\text{B})^{-0.34} = 0.69$ |

    **The MoE achieves ~31% lower parameter-dependent loss** with the same compute per forward pass.

    **Why MoE wins:**

    - Same inference cost (70B active params)
    - More knowledge stored in experts (1T total params)
    - Each token routes to specialists
    - Effective capacity >> active capacity

    **Caveat:** Training MoE requires ~1T parameters in memory/communication, increasing infrastructure complexity. The 1.5× factor is empirical and varies by architecture.

## Key Takeaways

1. **Pre-Chinchilla models were undertrained**: GPT-3, Gopher, etc. used 3-4× too many parameters relative to data.

2. **The 20:1 rule**: Compute-optimal training uses ~20 tokens per parameter.

3. **Chinchilla optimizes loss/FLOP**: This isn't the same as minimizing inference cost or maximizing capability.

4. **Overtraining is often rational**: When inference costs dominate, smaller overtrained models win.

5. **Know your constraints**: Data limits, inference budget, time pressure, and capability requirements all shift the optimal allocation.
