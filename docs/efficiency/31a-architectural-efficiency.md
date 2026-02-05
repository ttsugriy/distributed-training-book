---
title: "Architecture-Aware Efficiency"
subtitle: "GQA, Sliding Window Attention, MLA, and DualPipe"
---

<div class="chapter-opener" markdown>
The most impactful efficiency gains often come not from systems tricks but from architectural choices that reduce communication, memory, or compute at their source. This chapter covers four architectural innovations that change the distributed training landscape — understanding them here prepares us for the case studies in Part VIII.
</div>

<div class="investigation-question" markdown>
**The Question**: Standard Multi-Head Attention stores separate K, V projections per head. With 128 heads and 128K sequence length, KV cache alone can exceed 100 GB. Can we redesign the attention mechanism itself to reduce this by 10–50×, without losing quality?
</div>

!!! abstract "Chapter Map"
    **Prerequisites**: [Chapter 15](../parallelism/15-tensor-parallelism-linearity.md) (tensor parallelism), [Chapter 17](../parallelism/17-sequence-parallelism-decomposability.md) (sequence parallelism), [Chapter 16](../parallelism/16-pipeline-parallelism-separability.md) (pipeline schedules)

    **Key insight**: Architectural choices (attention pattern, KV sharing, pipeline scheduling) are the highest-leverage efficiency knobs because they reduce the fundamental work, not just how it's distributed.

## Grouped-Query Attention (GQA)

### The KV Cache Problem

In standard Multi-Head Attention (MHA), each of $A$ attention heads has its own Key and Value projections:

$$\text{KV cache per token} = 2 \times A \times d_h \times s \text{ bytes}$$

where $d_h = H/A$ is the per-head dimension and $s$ is bytes per element. For a 70B model with $A=64$, $H=8192$, BF16:

$$\text{KV cache per token} = 2 \times 64 \times 128 \times 2 = 32{,}768 \text{ bytes} \approx 32 \text{ KB}$$

For 128K context: $32 \text{ KB} \times 128{,}000 = 4.1 \text{ GB per layer}$. With 80 layers: **328 GB** — exceeding a single GPU.

### GQA: Sharing KV Heads

Grouped-Query Attention (Ainslie et al., 2023) groups $A$ query heads into $g$ groups, each sharing a single KV head:

$$\text{KV cache (GQA)} = 2 \times g \times d_h \times s$$

The reduction factor vs MHA is $A/g$:

| Variant | KV Heads ($g$) | Reduction vs MHA | Quality Impact |
|---------|---------------|------------------|----------------|
| MHA | $g = A$ (e.g., 64) | 1× (baseline) | — |
| GQA-8 | $g = 8$ | 8× | Minimal |
| GQA-1 (MQA) | $g = 1$ | 64× | Slight degradation |

LLaMA 2 70B and LLaMA 3 use GQA-8 ($g = 8$). This reduces KV cache by 8× with negligible quality loss.

### Distributed Training Implications

| Aspect | MHA | GQA-8 |
|--------|-----|-------|
| KV cache memory | $2AHs$ per token | $2gHs/A \cdot A = 2gd_h s$ per token |
| TP communication (KV) | AllGather $A$ KV heads | AllGather $g$ KV heads (8× less) |
| Sequence parallelism memory | $O(S \cdot A \cdot d_h)$ | $O(S \cdot g \cdot d_h)$ |
| Inference batch capacity | Limited by KV cache | 8× more sequences per GPU |

!!! note "Practice"
    GQA is now the default for all major LLMs. If designing a new model, start with GQA-8 unless you have a specific reason for full MHA.

## Sliding Window Attention (SWA)

### From Global to Local Attention

Standard attention computes all-pairs interactions: $O(S^2)$ per layer. Sliding Window Attention restricts each token to attend only to the $w$ nearest tokens:

$$\text{Attention}(Q_i, K, V) = \text{softmax}\left(\frac{Q_i K_{[i-w:i]}^T}{\sqrt{d_h}}\right) V_{[i-w:i]}$$

**Memory**: $O(S \times w)$ instead of $O(S^2)$

**Effective context**: With $L$ layers, information propagates $L \times w$ tokens — a 32-layer model with $w=4096$ has effective context of 131K tokens.

### Combined with GQA

SWA and GQA compose multiplicatively:

$$\text{KV cache (SWA + GQA)} = 2 \times g \times d_h \times w \times s$$

This is independent of sequence length $S$ — the cache is bounded by the window size.

| Configuration | KV Cache per Layer (BF16, $H=4096$, $S=32K$) |
|---------------|----------------------------------------------|
| MHA, global attention | $2 \times 32 \times 128 \times 32K \times 2 = 524$ MB |
| GQA-8, global | $2 \times 8 \times 128 \times 32K \times 2 = 131$ MB |
| GQA-8, SWA ($w=4096$) | $2 \times 8 \times 128 \times 4096 \times 2 = 16.8$ MB |

**31× reduction** from combining both techniques.

### Distributed Training Implications

- **Sequence parallelism**: SWA makes Ring Attention cheaper — each chunk only needs $w$ tokens of context from its neighbor, not the full preceding sequence
- **Pipeline parallelism**: Smaller activation tensors at stage boundaries
- **Memory**: Can increase batch size with the freed memory, improving MFU

Mistral 7B demonstrated that SWA + GQA can match models 2× their size — see [Chapter 36](../synthesis/36-case-study-mistral.md) for the full analysis.

## Multi-Head Latent Attention (MLA)

### Beyond GQA: Compressing the Latent Space

GQA reduces KV heads. MLA (DeepSeek-V2/V3) goes further by compressing the KV representation into a low-dimensional latent space:

$$c_t = W_{\text{DKV}} \cdot h_t \qquad K_t = W_{\text{UK}} \cdot c_t \qquad V_t = W_{\text{UV}} \cdot c_t$$

where:

- $h_t \in \mathbb{R}^H$ is the hidden state
- $c_t \in \mathbb{R}^{d_c}$ is the compressed latent ($d_c \ll H$)
- $W_{\text{DKV}} \in \mathbb{R}^{d_c \times H}$ compresses, $W_{\text{UK}}, W_{\text{UV}}$ decompress

**KV cache stores only $c_t$**, not full K, V:

$$\text{KV cache (MLA)} = d_c \times s \text{ per token per layer}$$

With $d_c = 512$ and $H = 7168$ (DeepSeek-V3): compression ratio $\approx H/d_c \approx 14\times$ beyond even GQA.

### Comparison

| Method | KV Cache per Token per Layer (BF16) | Relative to MHA |
|--------|-------------------------------------|-----------------|
| MHA ($A=128$, $d_h=128$) | $2 \times 128 \times 128 \times 2 = 65.5$ KB | 1× |
| GQA-8 | $2 \times 8 \times 128 \times 2 = 4.1$ KB | 0.063× |
| MLA ($d_c = 512$) | $512 \times 2 = 1.0$ KB | 0.015× |

MLA achieves **~65× compression** vs MHA. This enables DeepSeek-V3 to serve 671B parameters with practical KV cache sizes.

### Training Implications

During training, the decompression $K_t = W_{\text{UK}} c_t$ can be absorbed into the query projection (a matrix algebra trick), avoiding explicit decompression. This means:

- **Forward pass**: Same FLOPs as standard attention (decompression is fused)
- **Memory**: Only $c_t$ stored for backward pass → dramatic activation memory savings
- **Communication**: Smaller activation tensors for TP and SP

See [Chapter 35](../synthesis/35-case-study-deepseek.md) for DeepSeek-V3's full use of MLA.

## DualPipe: Bidirectional Pipeline Scheduling

### The Bubble Problem Revisited

Standard 1F1B pipeline parallelism has bubble fraction $(p-1)/(m+p-1)$. For $p=16$, $m=32$: bubble = 32%. Zero-Bubble schedules (ZB-H1) address this but add memory pressure.

### DualPipe: Two Pipelines, One Pass

DualPipe (DeepSeek-V3) splits micro-batches into two streams flowing in opposite directions through the pipeline:

```
Stream A: Stage 0 → Stage 1 → ... → Stage P-1   (forward direction)
Stream B: Stage P-1 → Stage P-2 → ... → Stage 0  (reverse direction)
```

Each stage alternates between processing micro-batches from both streams. While one stream is in the communication phase (sending activations), the other is in the compute phase.

### Bubble Reduction

$$\text{Bubble}_{\text{DualPipe}} \approx \frac{p-1}{2m}$$

Compare to 1F1B: $(p-1)/(m+p-1)$. For $p=16$, $m=32$:

- 1F1B: $15/47 = 31.9\%$
- DualPipe: $15/64 = 23.4\%$

The improvement grows with $m$: as $m \to \infty$, 1F1B bubble $\to 0$ slowly, while DualPipe bubble $\to 0$ twice as fast.

### Key Insight: Overlapping Communication and Compute Across Streams

The real power of DualPipe is that while stream A waits for an activation transfer, stream B can compute — and vice versa. This converts pipeline communication time into productive computation, effectively hiding the inter-stage latency.

**Constraint**: Requires sufficient memory to hold activations for both streams simultaneously. Memory overhead is approximately $2\times$ compared to standard 1F1B.

See [Chapter 35](../synthesis/35-case-study-deepseek.md) for DualPipe's role in DeepSeek-V3's training.

## Summary: Architecture as Efficiency

| Innovation | Primary Saving | Reduction Factor | Adopted By |
|------------|---------------|------------------|------------|
| GQA | KV cache memory | 4–64× | LLaMA 2/3, Mistral, Gemma |
| SWA | Attention memory/compute | $S/w$ × | Mistral, Mixtral |
| MLA | KV cache + activation memory | 14–65× | DeepSeek-V2/V3 |
| DualPipe | Pipeline bubble time | ~2× | DeepSeek-V3 |

These are not systems optimizations — they are **architectural choices that change the fundamental resource requirements**. Understanding them is essential for the case studies that follow.

## Key Takeaways

1. **GQA is the new default**: Sharing KV heads across query groups reduces cache 8× with negligible quality loss.

2. **SWA bounds memory independent of sequence length**: Attention memory becomes $O(S \times w)$ instead of $O(S^2)$.

3. **MLA compresses the KV bottleneck further**: Low-rank latent representations achieve 65× KV cache reduction.

4. **DualPipe halves effective bubble fraction**: Bidirectional micro-batch streams hide inter-stage communication.

5. **Architecture co-designs with distribution**: The best efficiency gains come from reducing the work at its source, not just distributing it better.
