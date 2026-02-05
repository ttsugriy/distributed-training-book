---
title: "Hardware Assumptions and Units"
---

This book uses back-of-envelope estimates. To keep calculations consistent, we use the following defaults unless stated otherwise. Most examples target **H100 SXM**; the table below provides reference values for other common accelerators so readers can adapt calculations to their own hardware.

## Accelerator Reference Table

| Spec | A100 80 GB SXM | H100 80 GB SXM | H200 141 GB SXM | AMD MI300X 192 GB |
|------|----------------|-----------------|------------------|-------------------|
| **HBM capacity** | 80 GB (HBM2e) | 80 GB (HBM3) | 141 GB (HBM3e) | 192 GB (HBM3) |
| **HBM bandwidth** | ~2.0 TB/s | ~3.35 TB/s | ~4.8 TB/s | ~5.3 TB/s |
| **Dense BF16/FP16 peak** | ~312 TFLOP/s | ~989 TFLOP/s | ~989 TFLOP/s | ~1,307 TFLOP/s |
| **Dense FP8 peak** | N/A | ~1,979 TFLOP/s | ~1,979 TFLOP/s | ~2,615 TFLOP/s |
| **TDP** | 400 W | 700 W | 700 W | 750 W |
| **Intra-node interconnect** | NVLink 3.0 | NVLink 4.0 | NVLink 4.0 | Infinity Fabric |
| **Intra-node BW (per GPU)** | ~600 GB/s | ~900 GB/s | ~900 GB/s | ~896 GB/s |

*Notes*: TFLOP/s values are non-sparse. H200 shares the H100 compute die but has more and faster HBM. MI300X FLOP/s figures are AMD's published peak; achieved rates depend on ROCm software maturity. All bandwidth figures are aggregate bidirectional.

## Default Values Used in Examples

Throughout the book, unless a section explicitly states otherwise:

- **GPU**: NVIDIA H100 SXM 80 GB
- **Dense BF16/FP16 peak**: ~989 TFLOP/s (non-sparse)
- **Dense FP8 peak**: ~1,979 TFLOP/s (non-sparse)
- **MFU** is defined relative to the stated peak in the local context

If a section uses a different precision (e.g., FP8 or sparsity), it says so explicitly.

## Interconnect Bandwidth

| Link | Bandwidth (per GPU) | Typical Latency | Notes |
|------|---------------------|-----------------|-------|
| NVLink 3.0 (A100) | ~600 GB/s | ~1 μs | 12 links × 50 GB/s |
| NVLink 4.0 (H100/H200) | ~900 GB/s | ~1 μs | 18 links × 50 GB/s |
| InfiniBand NDR 400 | ~50 GB/s | ~1–2 μs | Per NIC, per direction |
| InfiniBand NDR 200 | ~25 GB/s | ~1–2 μs | Common in older clusters |
| 100 GbE (RoCE v2) | ~12.5 GB/s | ~5–10 μs | Some cloud providers |
| PCIe Gen5 x16 | ~64 GB/s | ~1 μs | CPU↔GPU, bidirectional |

When we compute ridge points, we divide peak FLOP/s by the stated link bandwidth. If a section uses per-direction instead of aggregate, it calls that out explicitly.

## Typical DGX/SuperPOD Configurations

| Configuration | GPUs/Node | Intra-node | Inter-node | Example |
|---------------|-----------|------------|------------|---------|
| DGX A100 | 8 × A100 | NVLink 3.0 (600 GB/s) | 8 × HDR 200 IB | Many cloud instances |
| DGX H100 | 8 × H100 | NVLink 4.0 (900 GB/s) | 8 × NDR 400 IB | Meta Grand Teton |
| DGX H200 | 8 × H200 | NVLink 4.0 (900 GB/s) | 8 × NDR 400 IB | Latest generation |

## Units

- **GB** and **TB** denote decimal units (1 GB = 10^9 bytes)
- **GiB** and **TiB** denote binary units (1 GiB = 2^30 bytes)
- **Tokens** are counted in raw tokens unless otherwise stated

If you need exact provisioning, convert these estimates to your platform's reporting units (often GiB).
