---
title: "Notation and Minimal Formalism"
---

This appendix defines the core symbols used throughout the book. When in doubt, refer back here.

## Minimal Formalism

We model distributed training as the interaction of three resources:

- **Memory** (what must fit on each device)
- **Compute** (FLOPs per step per device)
- **Communication** (bytes transferred across links)

The basic cost model is:

$$T = T_{\text{compute}} + T_{\text{comm}}$$

With overlap, the effective step time is:

$$T_{\text{step}} \approx \max(T_{\text{compute}}, T_{\text{comm}})$$

For communication, we use the α-β model:

$$T(n) = \alpha + \frac{n}{\beta}$$

## Symbols

| Symbol | Meaning | Typical Units |
|---|---|---|
| $\Psi$ | Parameters | count |
| $N$ | Parameters (legacy in some chapters) | count |
| $D$ | Training tokens | tokens |
| $B$ | Global batch (sequences) | sequences/step |
| $b$ | Per-GPU batch (sequences) | sequences/step |
| $S$ | Sequence length | tokens |
| $H$ | Hidden size | dimension |
| $L$ | Layers | count |
| $P$ | Number of GPUs | count |
| $F$ | Peak throughput per GPU | FLOP/s |
| $\beta$ | Bandwidth | bytes/s |
| $\alpha$ | Latency | seconds |
| $b_{\text{tok}}$ | Bytes per token | bytes/token |
| $\rho$ | Read amplification | ratio |
| $I_{\text{io}}$ | Data I/O intensity | FLOPs/byte |
| $T$ | Time | seconds |
| MFU | Model FLOP Utilization | ratio |

Total tokens per step are $B \cdot S$ unless stated otherwise.

## Interpretation Tags

When you see labeled callouts, read them as follows:

- **Theory**: Algorithmic lower bounds and idealized models.
- **Practice**: Empirical ranges, framework behavior, and overheads.

## Conventions

- **GB/TB** are decimal (10^9/10^12 bytes)
- **GiB/TiB** are binary (2^30/2^40 bytes)
- Unless stated, FLOP/s assume dense FP16/BF16
