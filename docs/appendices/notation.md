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

## Core Symbols (All Parts)

| Symbol | Meaning | Typical Units |
|---|---|---|
| $\Psi$ | Number of model parameters | count |
| $D$ | Training tokens (dataset size) | tokens |
| $C$ | Total compute budget | FLOPs |
| $B$ | Global batch size (sequences) | sequences/step |
| $b$ | Per-GPU batch size (sequences) | sequences/step |
| $S$ | Sequence length | tokens |
| $H$ | Hidden dimension | dimension |
| $L$ | Number of transformer layers | count |
| $A$ | Number of attention heads | count |
| $V$ | Vocabulary size | count |
| $P$ | Total number of GPUs / processes | count |
| $F$ | Peak throughput per GPU | FLOP/s |
| MFU | Model FLOP Utilization | ratio |
| HFU | Hardware FLOP Utilization (includes recompute) | ratio |
| $T$ | Time (context-dependent: step, total, etc.) | seconds |
| $R$ | Hourly rate per GPU | USD/hr |

Total tokens per step are $B \cdot S$ unless stated otherwise.

## Scaling Laws (Part II)

!!! note "Symbol reuse"
    In Part II, $\alpha$ and $\beta$ denote scaling law exponents. From Part III onward, they denote network latency and bandwidth. The meaning should be clear from context.

| Symbol | Meaning | Typical Values |
|---|---|---|
| $\alpha$ | Parameter scaling exponent | 0.34 (Chinchilla) |
| $\beta$ | Data scaling exponent | 0.28 (Chinchilla) |
| $A, B$ | Scaling law constants | ~400 |
| $L_\infty$ | Irreducible loss | ~1.69 nats |

## Communication (Parts III–VIII)

| Symbol | Meaning | Typical Units |
|---|---|---|
| $\alpha$ | Network latency (per message) | seconds (~μs) |
| $\beta$ | Network bandwidth | bytes/s |
| $n$ | Message size | bytes |
| $n^*$ | Crossover point ($\alpha \cdot \beta$) | bytes |
| $I_{\text{net}}$ | Communication intensity (FLOPs/byte communicated) | FLOPs/byte |
| $I_{\text{mem}}$ | Memory intensity (FLOPs/byte from HBM) | FLOPs/byte |
| $I_{\text{io}}$ | Data I/O intensity | FLOPs/byte |
| $b_{\text{tok}}$ | Bytes per token (from storage) | bytes/token |
| $\rho$ | Read amplification | ratio |

## Parallelism Dimensions (Parts IV–VI)

| Symbol | Meaning | Typical Range |
|---|---|---|
| DP | Data parallelism degree | 1–512 |
| TP | Tensor parallelism degree | 1–8 (within node) |
| PP | Pipeline parallelism degree | 1–64 |
| CP | Context (sequence) parallelism degree | 1–32 |
| EP | Expert parallelism degree | 1–64 |
| $G$ | GPUs per node | 4 or 8 |
| $N$ | Number of nodes (also: parameters in some legacy contexts) | 1–1000s |
| $p$ | Pipeline stages | count |
| $m$ | Micro-batches per pipeline batch | count |
| $v$ | Virtual pipeline stages (interleaved schedule) | count |

## Memory (Part V)

| Symbol | Meaning | Notes |
|---|---|---|
| $M_{\text{params}}$ | Memory for parameters | $2\Psi$ bytes (FP16) |
| $M_{\text{grad}}$ | Memory for gradients | $2\Psi$ bytes (FP16) |
| $M_{\text{opt}}$ | Memory for optimizer states | $12\Psi$ bytes (AdamW FP32) |
| $M_{\text{act}}$ | Activation memory | Depends on $B, S, H, L$ |
| $k$ | Checkpoint interval (layers between checkpoints) | $\sqrt{L}$ optimal |

## Expert Parallelism (Part IV, Ch. 18)

| Symbol | Meaning | Notes |
|---|---|---|
| $E$ | Total number of experts | 8–256 |
| $k$ | Number of active experts per token | 1–4 |
| $C_f$ | Capacity factor | 1.0–1.5 |
| $f_i$ | Fraction of tokens routed to expert $i$ | ratio |
| $p_i$ | Average router probability for expert $i$ | ratio |

## Efficiency (Part VII)

| Symbol | Meaning | Notes |
|---|---|---|
| $w$ | Sliding window size | tokens |
| $g$ | Number of KV heads (GQA) | $g \leq A$ |
| $r$ | Rank (PowerSGD) or repetition ratio | context-dependent |
| $\tau$ | Staleness (async SGD) | steps |
| $H$ | Sync interval (Local SGD) | steps |
| $s$ | Quantization levels | count |

## Interpretation Tags

When you see labeled callouts, read them as follows:

- **Theory**: Algorithmic lower bounds and idealized models.
- **Practice**: Empirical ranges, framework behavior, and overheads.

## Conventions

- **GB/TB** are decimal ($10^9$/$10^{12}$ bytes)
- **GiB/TiB** are binary ($2^{30}$/$2^{40}$ bytes)
- Unless stated, FLOP/s assume dense FP16/BF16 on H100 SXM (~989 TFLOP/s)
- See [Hardware Assumptions](hardware-assumptions.md) for full accelerator specs
