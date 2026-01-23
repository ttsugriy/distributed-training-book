---
title: "Sequence Parallelism from Decomposability"
subtitle: "Sharding Along the Sequence Dimension"
---

::: {.chapter-opener}
When sequences grow to millions of tokens, even a single attention computation won't fit in memory. Sequence parallelism exploits the decomposability of attention to split along the sequence dimension.
:::

::: {.investigation-question}
**The Question**: Attention is O(S²) in sequence length. For S=1M tokens, that's 10^12 attention scores. How do we compute this when no single GPU can hold the attention matrix?
:::

## The Decomposability of Attention

Standard attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

The softmax is the challenge: it requires the full row to normalize.

## Ring Attention

**Key insight**: We can compute softmax incrementally.

For numerical stability, softmax uses:
$$\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_j e^{x_j - m}}$$

where $m = \max(x)$.

We can update this as we see more values:
$$m_{\text{new}} = \max(m_{\text{old}}, m_{\text{chunk}})$$
$$\text{sum}_{\text{new}} = \text{sum}_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \text{sum}_{\text{chunk}} \cdot e^{m_{\text{chunk}} - m_{\text{new}}}$$

This is **associative**!

## The Ring Attention Algorithm

```
Each GPU holds: Q_i (query chunk), K_i, V_i (local KV)

for step in range(P):
    # Compute attention with current K, V
    local_attn = attend(Q_i, K_current, V_current)
    
    # Update running softmax incrementally
    update_output(local_attn)
    
    # Send K, V to next GPU in ring
    K_current, V_current = ring_send_recv(K_current, V_current)
```

Communication overlaps with computation.

## Ulysses vs Ring

| Approach | Communication | Best When |
|----------|---------------|-----------|
| Ulysses (all-to-all) | AlltoAll of QKV | High bandwidth |
| Ring (P2P) | Point-to-point | Can overlap with compute |

*[Detailed comparison to be completed]*

## Exercises

*[To be completed]*
