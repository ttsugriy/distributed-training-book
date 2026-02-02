---
title: "Hardware Assumptions and Units"
---

This book uses back-of-envelope estimates. To keep calculations consistent, we use the following defaults unless stated otherwise.

## Compute Throughput

- **Dense FP16/BF16 peak (H100 SXM)**: ~989 TFLOP/s (non-sparse)
- **FP8 peak (H100 SXM)**: ~1979 TFLOP/s (non-sparse)
- **MFU** is defined relative to the stated peak in the local context

If a section uses a different precision (e.g., FP8 or sparsity), it should say so explicitly.

## Bandwidth

- **NVLink 4.0**: ~900 GB/s (per GPU, aggregate)
- **InfiniBand NDR 400**: ~50 GB/s (per GPU, per direction)

When we compute ridge points, we divide peak FLOP/s by the stated link bandwidth. If a section uses per-direction instead of aggregate, it should call that out explicitly.

## Units

- **GB** and **TB** denote decimal units (1 GB = 10^9 bytes)
- **GiB** and **TiB** denote binary units (1 GiB = 2^30 bytes)
- **Tokens** are counted in raw tokens unless otherwise stated

If you need exact provisioning, convert these estimates to your platform's reporting units (often GiB).
