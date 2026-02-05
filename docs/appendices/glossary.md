---
title: "Glossary"
---

## A

**Activation Recomputation** (also: *Gradient Checkpointing*): Technique that discards intermediate activations during the forward pass and recomputes them during the backward pass, trading ~33% extra compute for significant memory savings.

**AllGather**: Collective operation where each process contributes a shard; result is the concatenation of all shards on all processes.

**AllReduce**: Collective operation combining values from all processes and distributing the result to all. Equivalent to ReduceScatter followed by AllGather for associative/commutative reductions (algorithmic equivalence, not a semantic inverse).

**All-to-All (AlltoAll)**: Collective operation performing a transpose; each process sends different data to each other process.

**Arithmetic Intensity**: Ratio of FLOPs to bytes accessed from memory. Determines whether an operation is compute-bound or memory-bound.

## B

**Bubble**: Idle time in pipeline parallelism caused by pipeline startup/teardown.

**Bucket**: Collection of small tensors aggregated for a single AllReduce to amortize latency.

## C

**Chinchilla Scaling**: Compute-optimal training where tokens ≈ 20× parameters.

**Communication Intensity**: Ratio of FLOPs to bytes communicated over network. Determines whether an operation is network-bound.

**Context Parallelism (CP)**: Parallelism strategy splitting the sequence dimension across devices specifically for the attention computation, enabling long-context training (e.g., via Ring Attention).

**Critical Batch Size**: Batch size at which gradient noise equals gradient signal. Beyond this, larger batches yield diminishing returns.

## D

**Data Parallelism (DP)**: Parallelism strategy replicating the model across devices, splitting data. Synchronizes via gradient AllReduce.

**Device Mesh**: N-dimensional abstraction for organizing devices with different parallelism strategies along each axis.

**DiLoCo**: Distributed Low-Communication training. An approach where workers perform local SGD steps with infrequent global synchronization, reducing inter-node communication.

**DualPipe**: Advanced pipeline parallelism technique (DeepSeek-V3) that interleaves two pipelines to approximately halve the bubble fraction compared to standard 1F1B.

## E

**Expert Parallelism (EP)**: Parallelism strategy distributing MoE experts across devices.

## F

**FlashAttention**: IO-aware exact attention algorithm that avoids materializing the full $O(S^2)$ attention matrix by tiling the computation to exploit GPU SRAM, reducing memory from $O(S^2)$ to $O(S)$.

**FSDP**: Fully Sharded Data Parallel. PyTorch's ZeRO-style sharding (supports multiple sharding modes, including ZeRO-3-like).

## G

**GQA (Grouped-Query Attention)**: Attention variant where multiple query heads share a single key-value head, reducing KV cache memory and parameter count while preserving quality. Used in LLaMA 2/3, Mistral, etc.

**Gradient Accumulation**: Computing gradients over multiple micro-batches before synchronization, effectively increasing batch size.

**Gradient Checkpointing**: See *Activation Recomputation*.

## H

**HBM**: High Bandwidth Memory. GPU memory type with up to ~3 TB/s bandwidth (device-dependent).

## L

**LAMB/LARS**: Layer-wise Adaptive Rate Scaling. Optimizer modifications for large batch training.

**Loss Scaling**: Technique in mixed-precision training where the loss is multiplied by a large factor before the backward pass to prevent small gradients from underflowing to zero in FP16, then unscaled before the optimizer step.

## M

**MFU**: Model FLOP Utilization. Ratio of achieved model FLOP/s (useful computation only) to theoretical peak FLOP/s.

**Micro-batch**: Subdivision of batch for pipeline parallelism.

**MLA (Multi-head Latent Attention)**: Attention variant (DeepSeek-V2/V3) that compresses key-value projections through a learned low-rank latent space, dramatically reducing KV cache memory.

**MoE (Mixture of Experts)**: Architecture with multiple parallel FFN "experts" per layer, where a routing mechanism selects a sparse subset (top-k) of experts per token, enabling larger models without proportionally increasing compute.

## N

**NVLink**: High-speed interconnect within a node. NVLink 4.0 (H100) provides ~900 GB/s bidirectional per GPU with NVSwitch; NVLink 3.0 (A100) provides ~600 GB/s.

## P

**Pipeline Parallelism (PP)**: Parallelism strategy splitting model layers across devices.

## R

**ReduceScatter**: Collective operation reducing values and scattering shards to each process.

**Ring Algorithm**: Bandwidth-optimal collective algorithm organizing processes in a logical ring.

**RoPE (Rotary Position Embedding)**: Position encoding method applying rotation matrices to query and key vectors, enabling relative position awareness and length extrapolation.

**Roofline Model**: Performance analysis framework bounding throughput by compute ceiling, memory ceiling, or (extended) network ceiling.

## S

**Sequence Parallelism (SP)**: Parallelism strategy splitting along the sequence dimension.

**Speculative Decoding**: Inference optimization using a smaller "draft" model to generate candidate tokens that are verified in parallel by the larger model, improving latency without changing outputs.

## T

**Tensor Parallelism (TP)**: Parallelism strategy splitting individual tensor operations (matrix multiplications) across devices.

## Z

**ZeRO**: Zero Redundancy Optimizer. Memory optimization sharding optimizer states (ZeRO-1), gradients (ZeRO-2), and parameters (ZeRO-3) across data parallel ranks.
