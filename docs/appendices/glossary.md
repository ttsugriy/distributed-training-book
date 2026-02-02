---
title: "Glossary"
---

## A

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

**Critical Batch Size**: Batch size at which gradient noise equals gradient signal. Beyond this, larger batches yield diminishing returns.

## D

**Data Parallelism (DP)**: Parallelism strategy replicating the model across devices, splitting data. Synchronizes via gradient AllReduce.

**Device Mesh**: N-dimensional abstraction for organizing devices with different parallelism strategies along each axis.

## E

**Expert Parallelism (EP)**: Parallelism strategy distributing MoE experts across devices.

## F

**FSDP**: Fully Sharded Data Parallel. PyTorch's ZeRO-style sharding (supports multiple sharding modes, including ZeRO-3-like).

## G

**Gradient Accumulation**: Computing gradients over multiple micro-batches before synchronization, effectively increasing batch size.

## H

**HBM**: High Bandwidth Memory. GPU memory type with up to ~3 TB/s bandwidth (device-dependent).

## L

**LAMB/LARS**: Layer-wise Adaptive Rate Scaling. Optimizer modifications for large batch training.

## M

**MFU**: Model FLOP Utilization. Ratio of achieved FLOP/s to theoretical peak.

**Micro-batch**: Subdivision of batch for pipeline parallelism.

## N

**NVLink**: High-speed interconnect within a node (~900 GB/s per GPU).

## P

**Pipeline Parallelism (PP)**: Parallelism strategy splitting model layers across devices.

## R

**ReduceScatter**: Collective operation reducing values and scattering shards to each process.

**Ring Algorithm**: Bandwidth-optimal collective algorithm organizing processes in a logical ring.

**Roofline Model**: Performance analysis framework bounding throughput by compute ceiling, memory ceiling, or (extended) network ceiling.

## S

**Sequence Parallelism (SP)**: Parallelism strategy splitting along the sequence dimension.

## T

**Tensor Parallelism (TP)**: Parallelism strategy splitting individual tensor operations (matrix multiplications) across devices.

## Z

**ZeRO**: Zero Redundancy Optimizer. Memory optimization sharding optimizer states (ZeRO-1), gradients (ZeRO-2), and parameters (ZeRO-3) across data parallel ranks.
