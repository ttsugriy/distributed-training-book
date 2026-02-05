---
title: "Quick Reference"
---

A consolidated reference of the most-used formulas, decision tables, and environment variables from the book. Keep this page bookmarked.

## Core Equations

### Training Compute

$$C = 6\Psi D \qquad T_{\text{train}} = \frac{C}{P \cdot F \cdot \text{MFU}}$$

### Memory (Mixed Precision, AdamW)

| Component | Formula | 70B Model |
|-----------|---------|-----------|
| Parameters (FP16) | $2\Psi$ | 140 GB |
| Gradients (FP16) | $2\Psi$ | 140 GB |
| Optimizer (FP32) | $12\Psi$ | 840 GB |
| **Total static** | $16\Psi$ | **1.12 TB** |

### Activation Memory (Per Layer, No Checkpointing)

$$M_{\text{act}}^{\text{layer}} \approx BSH \cdot \left(34 + 5\frac{AS}{H}\right) \text{ bytes}$$

With checkpointing every $k$ layers ($k^* = \sqrt{L}$ optimal):

$$M_{\text{act}}^{\text{ckpt}} \approx \frac{L}{k} \cdot 2BSH + k \cdot M_{\text{act}}^{\text{layer}}$$

### Chinchilla Optimal Allocation

$$\Psi^* \approx \sqrt{\frac{C}{120}} \qquad D^* \approx 20\,\Psi^* \qquad \frac{D^*}{\Psi^*} \approx 20$$

### Tokens per Second

$$\text{tokens/s} = \frac{P \cdot F \cdot \text{MFU}}{6\Psi}$$

## Communication Cost Formulas (α-β Model)

All formulas use: $P$ = processes, $n$ = message size (bytes), $\alpha$ = latency, $\beta$ = bandwidth (bytes/s).

| Collective | Latency Term | Bandwidth Term | Best For |
|------------|-------------|----------------|----------|
| **AllReduce** (ring) | $2(P-1)\alpha$ | $2\frac{P-1}{P}\frac{n}{\beta}$ | Large messages |
| **AllReduce** (tree) | $2\log_2 P \cdot \alpha$ | $2\log_2 P \cdot \frac{n}{\beta}$ | Small messages |
| **AllReduce** (RHD) | $2\log_2 P \cdot \alpha$ | $2\frac{P-1}{P}\frac{n}{\beta}$ | Power-of-2 $P$ |
| **ReduceScatter** | $(P-1)\alpha$ | $\frac{P-1}{P}\frac{n}{\beta}$ | ZeRO, FSDP |
| **AllGather** | $(P-1)\alpha$ | $\frac{P-1}{P}\frac{n}{\beta}$ | ZeRO-3 forward |
| **All-to-All** | $(P-1)\alpha$ | $\frac{P-1}{P}\frac{n}{\beta}$ | MoE routing |
| **Broadcast** (tree) | $\log_2 P \cdot \alpha$ | $\log_2 P \cdot \frac{n}{\beta}$ | Weight init |

### Hierarchical AllReduce ($G$ GPUs/node, $N$ nodes)

$$T_{\text{hier}} = \underbrace{2(G{-}1)\alpha_{\text{NV}} + 2\tfrac{G{-}1}{G}\tfrac{n}{\beta_{\text{NV}}}}_{\text{intra-node}} + \underbrace{2(N{-}1)\alpha_{\text{net}} + 2\tfrac{N{-}1}{N}\tfrac{n/G}{\beta_{\text{net}}}}_{\text{inter-node (1/G data)}}$$

### Bus Bandwidth Correction Factors

| Collective | Correction Factor |
|------------|-------------------|
| AllReduce | $2(P-1)/P$ |
| AllGather / ReduceScatter / AlltoAll | $(P-1)/P$ |
| Broadcast / Reduce | 1 |

$$\text{busbw} = \text{algbw} \times \text{correction factor} \qquad \text{algbw} = n / T$$

### Ridge Points (H100 SXM)

| Link | Bandwidth | Ridge Point ($F/\beta$) |
|------|-----------|------------------------|
| NVLink 4.0 | 900 GB/s | ~1,099 FLOPs/byte |
| InfiniBand NDR 400 | 50 GB/s | ~19,780 FLOPs/byte |

## Parallelism Communication Per Step

| Strategy | Volume Per GPU | Collective | Link |
|----------|---------------|------------|------|
| Data Parallel | $\approx 2\Psi \cdot s$ bytes | AllReduce | Network |
| Tensor Parallel | $4L \times BSH \cdot 2$ bytes | AllReduce | NVLink |
| Pipeline Parallel | $\text{micro-batches} \times BSH \cdot 2$ bytes | Send/Recv | Network |
| ZeRO-3 (per layer) | $2 \times n_{\text{layer}}$ bytes | AG + RS | Network |

($s$ = bytes per element, e.g. 2 for FP16)

## Pipeline Bubble Fractions

| Schedule | Bubble Fraction | Memory (per GPU) |
|----------|----------------|------------------|
| GPipe | $\frac{p-1}{m+p-1}$ | $O(m)$ activations |
| 1F1B | $\frac{p-1}{m+p-1}$ | $O(p)$ activations |
| Interleaved ($v$ virtual stages) | $\frac{p-1}{m \cdot v + p - 1}$ | $O(p)$ activations |
| Zero-Bubble (ZB-H1) | $\approx 0$ | $O(p)$ activations |

$p$ = pipeline stages, $m$ = micro-batches per batch.

## ZeRO Memory Savings

| Stage | What's Sharded | Memory Per GPU | Extra Communication |
|-------|---------------|----------------|---------------------|
| Baseline (DDP) | Nothing | $16\Psi$ | $2\Psi s$ (AllReduce) |
| ZeRO-1 | Optimizer states | $4\Psi + 12\Psi/P$ | Same as DDP |
| ZeRO-2 | + Gradients | $2\Psi + 14\Psi/P$ | Same as DDP |
| ZeRO-3 | + Parameters | $16\Psi/P$ | $3 \times 2\Psi s$ (AG + RS) |

## Parallelism Placement Decision Procedure

```
1. MEMORY FIT?
   Total static memory = 16Ψ/P_shard + activations
   If no → add ZeRO stages, activation checkpointing, or more PP/TP

2. TP DEGREE (within node, NVLink):
   TP = min(8, GPUs_per_node)  — typically 4 or 8
   Only increase if per-GPU params don't fit

3. PP DEGREE (across nodes, tolerates latency):
   PP = ceil(layers / max_layers_per_device)
   Use m ≥ 4×PP micro-batches to keep bubbles < 20%

4. DP DEGREE (outer dimension):
   DP = P / (TP × PP)
   Use gradient accumulation if DP × local_batch < B_crit

5. CP DEGREE (if sequence > 8K):
   CP = S / S_max_per_GPU
   Ring Attention or Ulysses depending on S and H

6. EP DEGREE (if MoE):
   EP = E / experts_per_GPU
   Ensure AlltoAll volume fits in network budget
```

## Key NCCL Environment Variables

| Variable | Purpose | Common Values |
|----------|---------|---------------|
| `NCCL_ALGO` | Force collective algorithm | `Ring`, `Tree`, `CollnetDirect` |
| `NCCL_PROTO` | Communication protocol | `Simple`, `LL`, `LL128` |
| `NCCL_TOPO_FILE` | Custom topology file | Path to XML |
| `NCCL_DEBUG` | Debug logging level | `INFO`, `WARN`, `TRACE` |
| `NCCL_IB_DISABLE` | Disable InfiniBand | `0` (default), `1` |
| `NCCL_P2P_LEVEL` | P2P communication level | `NVL`, `PIX`, `PHB`, `SYS` |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA level | `SYS`, `PHB`, `PIX`, `LOC` |
| `NCCL_SOCKET_IFNAME` | Network interface | `eth0`, `ib0`, etc. |
| `NCCL_BUFFSIZE` | Per-channel buffer size | Bytes (default 4 MB) |

## Common Sanity Checks

| What to Check | Formula | Red Flag |
|---------------|---------|----------|
| MFU | $\frac{6\Psi \cdot \text{tokens/s}}{P \cdot F}$ | < 30% |
| Comm fraction | $T_{\text{comm}} / T_{\text{step}}$ | > 40% exposed |
| Pipeline bubble | $(p-1)/(m+p-1)$ | > 25% |
| Memory utilization | $M_{\text{used}} / M_{\text{HBM}}$ | > 95% (fragmentation risk) |
| Data starvation | tokens/s vs $PF \cdot \text{MFU} / (6\Psi)$ | Data loader < required |
