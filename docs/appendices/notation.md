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
| $\Psi$ or $N$ | Parameters | count |
| $D$ | Training tokens | tokens |
| $B$ | Global batch (tokens) | tokens/step |
| $b$ | Per-GPU batch (tokens) | tokens/step |
| $S$ | Sequence length | tokens |
| $H$ | Hidden size | dimension |
| $L$ | Layers | count |
| $P$ | Number of GPUs | count |
| $F$ | Peak throughput per GPU | FLOP/s |
| $\beta$ | Bandwidth | bytes/s |
| $\alpha$ | Latency | seconds |
| $T$ | Time | seconds |
| MFU | Model FLOP Utilization | ratio |

## Conventions

- **GB/TB** are decimal (10^9/10^12 bytes)
- **GiB/TiB** are binary (2^30/2^40 bytes)
- Unless stated, FLOP/s assume dense FP16/BF16
