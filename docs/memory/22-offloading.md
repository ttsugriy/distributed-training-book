---
title: "Offloading: Extending Memory Beyond the GPU"
subtitle: "CPU and NVMe as Memory Hierarchy Extensions"
---

<div class="chapter-opener" markdown>
When GPU memory isn't enough, we don't just add more GPUs—we extend the memory hierarchy. Offloading treats CPU memory and NVMe storage as slower tiers of GPU memory, enabling training of models that exceed any single GPU's capacity.
</div>

<div class="investigation-question" markdown>
**The Question**: A 175B parameter model requires 350GB just for parameters in fp16. The largest GPU has 80GB. How do we train this model on a single node? What are the bandwidth constraints, and how do we hide latency?
</div>

## The Memory Hierarchy

Modern training systems have a three-tier memory hierarchy:

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Hierarchy                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────┐                                              │
│  │  GPU HBM   │  80GB, 3.35TB/s bandwidth (H100)              │
│  │  (Hot)     │  Compute happens here                        │
│  └─────┬──────┘                                              │
│        │ PCIe Gen4: 32GB/s                                   │
│        │ NVLink: 900GB/s (intra-node, H100)                   │
│        ▼                                                     │
│  ┌────────────┐                                              │
│  │ CPU DRAM   │  512GB-2TB, 200GB/s                          │
│  │  (Warm)    │  Optimizer states, gradients                 │
│  └─────┬──────┘                                              │
│        │ NVMe: 3-7GB/s per drive                             │
│        │ RAID: 10-30GB/s aggregate                           │
│        ▼                                                     │
│  ┌────────────┐                                              │
│  │   NVMe     │  10-100TB                                    │
│  │  (Cold)    │  Parameters, checkpoints                     │
│  └────────────┘                                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Bandwidth Analysis

The key insight: **bandwidth decreases exponentially down the hierarchy**.

| Tier | Capacity | Bandwidth | Latency |
|------|----------|-----------|---------|
| GPU HBM | 80GB | 3,350 GB/s | ~1μs |
| CPU DRAM | 1TB | 200 GB/s | ~100ns |
| NVMe SSD | 10TB | 7 GB/s | ~10μs |
| HDD | 100TB | 0.2 GB/s | ~10ms |

The bandwidth ratio GPU:CPU:NVMe is roughly 300:30:1.

### The Offloading Principle

**Principle**: Move data that isn't needed *right now* to cheaper storage, prefetch it before it's needed, and overlap transfers with computation.

$$\text{Effective Bandwidth} = \min(\text{Transfer Rate}, \text{Compute Time} / \text{Data Size})$$

If we can hide transfers behind computation, the effective bandwidth is infinite.

## CPU Offloading

CPU offloading moves optimizer states and gradients to CPU memory.

### What to Offload

**Good candidates for offloading**:

1. **Optimizer states**: Adam's m and v (accessed once per step)
2. **Master weights**: fp32 copies (needed only for update)
3. **Gradients**: After reduction, before optimizer step

**Poor candidates**:

1. **Activations**: Accessed during backward pass (high frequency)
2. **Parameters**: Needed for every forward/backward (critical path)

### ZeRO-Offload

DeepSpeed's ZeRO-Offload partitions the optimizer step:

```
┌─────────────────────────────────────────────────────────────┐
│                    ZeRO-Offload                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GPU Side:                     CPU Side:                     │
│  ┌──────────────────┐          ┌──────────────────┐          │
│  │ Forward Pass     │          │ Optimizer States │          │
│  │ (activations)    │          │ (m, v, fp32 w)   │          │
│  └────────┬─────────┘          └────────▲─────────┘          │
│           │                              │                   │
│  ┌────────▼─────────┐          ┌────────┴─────────┐          │
│  │ Backward Pass    │   ──────▶│ Optimizer Step   │          │
│  │ (gradients)      │ gradients│ (Adam update)    │          │
│  └────────┬─────────┘          └────────┬─────────┘          │
│           │                              │                   │
│  ┌────────▼─────────┐          ┌────────▼─────────┐          │
│  │ Update Weights   │◀──────── │ New Weights      │          │
│  │ (fp16 params)    │  weights │ (fp32 → fp16)    │          │
│  └──────────────────┘          └──────────────────┘          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

```python
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor

@dataclass
class OffloadConfig:
    """Configuration for CPU offloading."""
    offload_optimizer: bool = True
    offload_gradients: bool = True
    offload_parameters: bool = False  # Only for ZeRO-3 + offload
    pin_memory: bool = True  # Use pinned memory for faster transfers
    num_threads: int = 4  # CPU threads for optimizer step
    overlap_comm: bool = True  # Overlap CPU-GPU transfers

class CPUOffloadOptimizer:
    """
    Optimizer that offloads states to CPU memory.

    Memory layout:

    - GPU: fp16 parameters, gradients (temporary)
    - CPU: fp32 master weights, Adam states (m, v)

    Update flow:
    1. Gradients computed on GPU (fp16)
    2. Gradients transferred to CPU, cast to fp32
    3. Adam update on CPU (fp32)
    4. Updated weights transferred to GPU, cast to fp16
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        config: Optional[OffloadConfig] = None
    ):
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.config = config or OffloadConfig()

        self.step_count = 0

        # Create CPU copies and optimizer states
        self._init_offload_buffers()

        # Thread pool for parallel CPU operations
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_threads)

        # CUDA streams for overlapped transfers
        self.d2h_stream = torch.cuda.Stream()  # Device to host
        self.h2d_stream = torch.cuda.Stream()  # Host to device

    def _init_offload_buffers(self):
        """Initialize CPU buffers for parameters and optimizer states."""
        self.cpu_params: Dict[str, torch.Tensor] = {}
        self.cpu_m: Dict[str, torch.Tensor] = {}  # First moment
        self.cpu_v: Dict[str, torch.Tensor] = {}  # Second moment
        self.gpu_params: Dict[str, nn.Parameter] = {}

        # Gradient buffers (pinned for faster transfer)
        self.cpu_grads: Dict[str, torch.Tensor] = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            self.gpu_params[name] = param

            # CPU copies of parameters in fp32
            cpu_param = param.data.float().cpu()
            if self.config.pin_memory:
                cpu_param = cpu_param.pin_memory()
            self.cpu_params[name] = cpu_param

            # Initialize Adam states
            self.cpu_m[name] = torch.zeros_like(cpu_param)
            self.cpu_v[name] = torch.zeros_like(cpu_param)

            # Pinned gradient buffer
            grad_buffer = torch.empty_like(cpu_param)
            if self.config.pin_memory:
                grad_buffer = grad_buffer.pin_memory()
            self.cpu_grads[name] = grad_buffer

    def _transfer_gradients_to_cpu(self):
        """Asynchronously transfer gradients from GPU to CPU."""
        with torch.cuda.stream(self.d2h_stream):
            for name, param in self.gpu_params.items():
                if param.grad is not None:
                    # Non-blocking copy to pinned memory
                    self.cpu_grads[name].copy_(
                        param.grad.float(),
                        non_blocking=True
                    )

    def _cpu_adam_step(self, name: str):
        """Perform Adam update on CPU for a single parameter."""
        beta1, beta2 = self.betas

        param = self.cpu_params[name]
        grad = self.cpu_grads[name]
        m = self.cpu_m[name]
        v = self.cpu_v[name]

        # Bias correction
        bias_correction1 = 1 - beta1 ** self.step_count
        bias_correction2 = 1 - beta2 ** self.step_count

        # Update moments
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Compute update
        denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(self.eps)
        step_size = self.lr / bias_correction1

        # Apply weight decay (decoupled, like AdamW)
        if self.weight_decay != 0:
            param.add_(param, alpha=-self.lr * self.weight_decay)

        # Apply update
        param.addcdiv_(m, denom, value=-step_size)

    def _transfer_weights_to_gpu(self):
        """Asynchronously transfer updated weights from CPU to GPU."""
        with torch.cuda.stream(self.h2d_stream):
            for name, gpu_param in self.gpu_params.items():
                cpu_param = self.cpu_params[name]
                # Cast back to fp16/bf16 and copy to GPU
                gpu_param.data.copy_(
                    cpu_param.to(gpu_param.dtype),
                    non_blocking=True
                )

    def step(self):
        """Perform optimization step with CPU offloading."""
        self.step_count += 1

        # Step 1: Transfer gradients to CPU (async)
        self._transfer_gradients_to_cpu()

        # Synchronize D2H stream before CPU computation
        self.d2h_stream.synchronize()

        # Step 2: Perform Adam updates on CPU (parallel across parameters)
        if self.config.num_threads > 1:
            futures = [
                self.executor.submit(self._cpu_adam_step, name)
                for name in self.cpu_params.keys()
            ]
            for future in futures:
                future.result()  # Wait for completion
        else:
            for name in self.cpu_params.keys():
                self._cpu_adam_step(name)

        # Step 3: Transfer updated weights back to GPU (async)
        self._transfer_weights_to_gpu()

        # Synchronize H2D stream (can be deferred to next forward)
        # self.h2d_stream.synchronize()

    def synchronize(self):
        """Ensure all async operations are complete."""
        self.d2h_stream.synchronize()
        self.h2d_stream.synchronize()

    def zero_grad(self):
        """Clear GPU gradients."""
        for param in self.gpu_params.values():
            if param.grad is not None:
                param.grad.zero_()
```

### Memory Savings

For Adam optimizer with fp16 training, let $\Psi$ denote the number of model parameters:

| Component | Without Offload | With Offload |
|-----------|-----------------|--------------|
| Parameters (fp16) | $2\Psi$ bytes | $2\Psi$ bytes (GPU) |
| Gradients (fp16) | $2\Psi$ bytes | $2\Psi$ bytes (peak during backward) → 0 (after transfer) |
| Master weights (fp32) | $4\Psi$ bytes | $4\Psi$ bytes (CPU) |
| Adam m (fp32) | $4\Psi$ bytes | $4\Psi$ bytes (CPU) |
| Adam v (fp32) | $4\Psi$ bytes | $4\Psi$ bytes (CPU) |
| **Total GPU (steady-state)** | **$16\Psi$ bytes** | **$2\Psi$ bytes** |
| **Total GPU (peak)** | **$16\Psi$ bytes** | **$4\Psi$ bytes** |

**Memory reduction**: 4× on GPU memory.

### Bandwidth Requirements

For a step to complete in time $T$, let $\Psi$ be the number of parameters:

$$\text{Gradient transfer}: \frac{4\Psi}{B_{\text{PCIe}}} < T_{\text{backward}}$$

$$\text{Weight transfer}: \frac{2\Psi}{B_{\text{PCIe}}} < T_{\text{forward}}$$

Where $B_{\text{PCIe}}$ is the PCIe bandwidth (32 GB/s for Gen4).

With PCIe Gen4 (32 GB/s) and a 10B model (40GB gradients):

$$\text{Gradient transfer time} = \frac{40 \text{ GB}}{32 \text{ GB/s}} = 1.25s$$

If backward pass takes 2s, transfer can be hidden.

## NVMe Offloading

When CPU memory isn't enough, NVMe provides the next tier.

### ZeRO-Infinity

DeepSpeed's ZeRO-Infinity extends offloading to NVMe:

```
┌─────────────────────────────────────────────────────────────┐
│                    ZeRO-Infinity                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GPU:                                                        │
│  ┌────────────────────────────────────────────────────┐      │
│  │ Working Set: Current layer parameters + activations│      │
│  │ Size: O(hidden_dim²) per layer                     │      │
│  └────────────────────────────────────────────────────┘      │
│        ▲                                                     │
│        │ Prefetch                                            │
│  CPU:  ▼                                                     │
│  ┌────────────────────────────────────────────────────┐      │
│  │ Prefetch Buffer: Next $N$ layers                   │      │
│  │ Optimizer State Partition (if CPU offload)         │      │
│  └────────────────────────────────────────────────────┘      │
│        ▲                                                     │
│        │ Async I/O                                           │
│  NVMe: ▼                                                     │
│  ┌────────────────────────────────────────────────────┐      │
│  │ All Parameters (sharded)                           │      │
│  │ All Optimizer States (sharded)                     │      │
│  │ Checkpoint Data                                    │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Async I/O for NVMe

Standard file I/O blocks the CPU. For efficient NVMe access, we need async I/O:

```python
import os
import io
import asyncio
import aiofiles
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch

@dataclass
class NVMeConfig:
    """Configuration for NVMe offloading."""
    nvme_path: str = "/mnt/nvme/offload"
    block_size: int = 1024 * 1024  # 1MB blocks
    aio_queue_depth: int = 8  # Concurrent I/O operations
    prefetch_depth: int = 2  # Layers to prefetch
    pin_memory: bool = True

class NVMeOffloadManager:
    """
    Manages parameter offloading to NVMe storage.

    Uses memory-mapped files and async I/O for efficient access.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[NVMeConfig] = None
    ):
        self.model = model
        self.config = config or NVMeConfig()

        # Ensure offload directory exists
        os.makedirs(self.config.nvme_path, exist_ok=True)

        # Parameter metadata
        self.param_info: Dict[str, Tuple[torch.Size, torch.dtype, int]] = {}

        # Memory-mapped files
        self.mmap_files: Dict[str, np.memmap] = {}

        # Prefetch buffers (pinned CPU memory)
        self.prefetch_buffers: Dict[str, torch.Tensor] = {}

        # Track which parameters are currently in GPU
        self.gpu_resident: set = set()

        self._init_nvme_storage()

    def _init_nvme_storage(self):
        """Initialize NVMe storage for all parameters."""
        offset = 0

        for name, param in self.model.named_parameters():
            size_bytes = param.numel() * param.element_size()

            # Store metadata
            self.param_info[name] = (param.shape, param.dtype, offset)
            offset += size_bytes

            # Create memory-mapped file
            filepath = os.path.join(self.config.nvme_path, f"{name.replace('.', '_')}.bin")
            mmap = np.memmap(
                filepath,
                dtype=self._torch_to_numpy_dtype(param.dtype),
                mode='w+',
                shape=param.shape
            )

            # Write initial values
            if param.dtype == torch.bfloat16:
                mmap[:] = param.data.cpu().view(torch.uint16).numpy()
            else:
                mmap[:] = param.data.cpu().numpy()
            mmap.flush()

            self.mmap_files[name] = mmap

            # Create pinned prefetch buffer
            buffer = torch.empty(
                param.shape,
                dtype=param.dtype,
                device='cpu'
            )
            if self.config.pin_memory:
                buffer = buffer.pin_memory()
            self.prefetch_buffers[name] = buffer

            # Clear GPU parameter (replace with placeholder)
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)

    def _torch_to_numpy_dtype(self, dtype: torch.dtype) -> np.dtype:
        """Convert PyTorch dtype to NumPy dtype."""
        mapping = {
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.bfloat16: np.uint16,  # Store raw bits
            torch.int32: np.int32,
            torch.int64: np.int64,
        }
        return mapping.get(dtype, np.float32)

    async def _async_load(self, name: str) -> torch.Tensor:
        """Asynchronously load parameter from NVMe."""
        mmap = self.mmap_files[name]
        buffer = self.prefetch_buffers[name]
        shape, dtype, _ = self.param_info[name]

        # Async copy from mmap to buffer
        # In practice, use aiofiles or io_uring
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: buffer.copy_(
                torch.from_numpy(np.array(mmap)).view(dtype) if dtype == torch.bfloat16 else torch.from_numpy(np.array(mmap))
            )
        )

        return buffer

    async def _async_save(self, name: str, data: torch.Tensor):
        """Asynchronously save parameter to NVMe."""
        mmap = self.mmap_files[name]
        _, dtype, _ = self.param_info[name]

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: np.copyto(mmap, data.cpu().view(torch.uint16).numpy() if dtype == torch.bfloat16 else data.cpu().numpy())
        )
        mmap.flush()

    def prefetch(self, layer_names: List[str]):
        """Prefetch parameters for upcoming layers."""
        async def _prefetch_all():
            tasks = [self._async_load(name) for name in layer_names]
            await asyncio.gather(*tasks)

        asyncio.run(_prefetch_all())

    def load_to_gpu(self, name: str, device: torch.device) -> torch.Tensor:
        """Load parameter to GPU from prefetch buffer or NVMe."""
        if name in self.gpu_resident:
            return  # Already loaded

        buffer = self.prefetch_buffers[name]
        shape, dtype, _ = self.param_info[name]

        # Copy from pinned CPU to GPU
        gpu_tensor = buffer.to(device, non_blocking=True)

        self.gpu_resident.add(name)
        return gpu_tensor

    def offload_from_gpu(self, name: str, data: torch.Tensor):
        """Offload parameter from GPU back to NVMe."""
        if name not in self.gpu_resident:
            return

        # Async save to NVMe
        asyncio.run(self._async_save(name, data))

        self.gpu_resident.discard(name)

    def get_memory_stats(self) -> Dict[str, int]:
        """Get current memory usage statistics."""
        gpu_memory = sum(
            self.param_info[name][0].numel() *
            torch.tensor([], dtype=self.param_info[name][1]).element_size()
            for name in self.gpu_resident
        )

        total_memory = sum(
            info[0].numel() * torch.tensor([], dtype=info[1]).element_size()
            for info in self.param_info.values()
        )

        return {
            'gpu_memory': gpu_memory,
            'nvme_memory': total_memory,
            'offload_ratio': 1 - (gpu_memory / total_memory) if total_memory > 0 else 0
        }
```

### Bandwidth Optimization

NVMe bandwidth is precious. Optimization strategies:

**1. Sequential Access**

Random I/O is slow. Structure data for sequential reads:

```python
def pack_layers_sequentially(model: nn.Module, filepath: str):
    """
    Pack model parameters sequentially for optimal NVMe access.

    Instead of one file per parameter, pack all parameters
    in layer order for sequential prefetching.
    """
    layer_order = []
    total_bytes = 0

    with open(filepath, 'wb') as f:
        for name, param in model.named_parameters():
            data = param.data.cpu().numpy()
            offset = f.tell()
            f.write(data.tobytes())

            layer_order.append({
                'name': name,
                'offset': offset,
                'size': data.nbytes,
                'shape': data.shape,
                'dtype': str(data.dtype)
            })
            total_bytes += data.nbytes

    return layer_order, total_bytes
```

**2. Read-Ahead**

Prefetch next layers while processing current:

```python
class LayerPrefetcher:
    """Prefetch layers ahead of computation."""

    def __init__(
        self,
        layer_order: List[str],
        prefetch_depth: int = 2
    ):
        self.layer_order = layer_order
        self.prefetch_depth = prefetch_depth
        self.current_idx = 0
        self.prefetched: Dict[str, torch.Tensor] = {}

    def get_layer(self, name: str) -> torch.Tensor:
        """Get layer, triggering prefetch of next layers."""
        # Return prefetched if available
        if name in self.prefetched:
            tensor = self.prefetched.pop(name)
        else:
            # Synchronous load (cache miss)
            tensor = self._load_layer(name)

        # Trigger prefetch for upcoming layers
        self._prefetch_upcoming()

        return tensor

    def _prefetch_upcoming(self):
        """Prefetch next layers in background."""
        for i in range(self.prefetch_depth):
            idx = self.current_idx + i + 1
            if idx < len(self.layer_order):
                name = self.layer_order[idx]
                if name not in self.prefetched:
                    # Async load
                    self.prefetched[name] = self._async_load_layer(name)

        self.current_idx += 1
```

**3. Compression**

Reduce I/O volume with compression:

```python
import lz4.frame

def compress_parameter(tensor: torch.Tensor) -> bytes:
    """Compress parameter for NVMe storage."""
    data = tensor.numpy().tobytes()
    return lz4.frame.compress(data, compression_level=0)  # Fast mode

def decompress_parameter(
    compressed: bytes,
    shape: Tuple[int, ...],
    dtype: np.dtype
) -> torch.Tensor:
    """Decompress parameter from NVMe."""
    data = lz4.frame.decompress(compressed)
    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    return torch.from_numpy(array.copy())
```

For fp16 model weights, LZ4 typically achieves 1.3-1.5× compression with minimal CPU overhead.

## Overlapping Transfers with Computation

The key to efficient offloading: **overlap**.

### The Overlap Principle

```
Without overlap:
├──Transfer──┤├──Compute──┤├──Transfer──┤├──Compute──┤
             ▲            ▲             ▲            ▲
         Transfer     Compute       Transfer     Compute
           Wait                       Wait

With overlap:
├──Transfer─1─┤
     ├──Compute─0─┤
          ├──Transfer─2─┤
               ├──Compute─1─┤
                    ├──Transfer─3─┤
```

Effective time = max(Transfer time, Compute time), not sum.

### Double Buffering

Use two buffers: one for current computation, one for next transfer:

```python
class DoubleBufferedLoader:
    """
    Double-buffered parameter loading for overlapped transfers.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.buffers = [None, None]  # Two GPU buffers
        self.current_buffer = 0
        self.load_stream = torch.cuda.Stream()
        self.pending_load: Optional[str] = None

    def get_current_buffer(self) -> torch.Tensor:
        """Get buffer containing current layer's parameters."""
        return self.buffers[self.current_buffer]

    def start_next_load(
        self,
        cpu_tensor: torch.Tensor,
        shape: Tuple[int, ...]
    ):
        """Start async load of next layer's parameters."""
        next_buffer = 1 - self.current_buffer

        # Allocate or reuse buffer
        if (self.buffers[next_buffer] is None or
            self.buffers[next_buffer].shape != shape):
            self.buffers[next_buffer] = torch.empty(
                shape,
                dtype=cpu_tensor.dtype,
                device=self.device
            )

        # Async copy
        with torch.cuda.stream(self.load_stream):
            self.buffers[next_buffer].copy_(cpu_tensor, non_blocking=True)

    def swap_buffers(self):
        """Swap active buffer, synchronizing pending load."""
        self.load_stream.synchronize()
        self.current_buffer = 1 - self.current_buffer
```

### Compute-Communication Overlap Analysis

For layer $l$ with:

- Parameter size: $P_l$ bytes
- Compute time: $T_l^{compute}$
- Transfer bandwidth: $B$

Transfer time: $T_l^{transfer} = P_l / B$

**Overlap condition**: $T_l^{transfer} \leq T_{l-1}^{compute}$

For a transformer layer:

- Parameters: ~$12H^2$ (for hidden dim $H$)
- Compute: ~$24BSH^2$ FLOPs (for batch $B$, sequence $S$)

With PCIe Gen4 (32 GB/s) and A100 (312 TFLOPs fp16):

$$\frac{12H^2 \cdot 2}{32 \times 10^9} \leq \frac{24BSH^2}{312 \times 10^{12}}$$

Solving:

$$BS \geq \frac{312 \times 10^{12}}{32 \times 10^9} \approx 9{,}750$$

With batch size 1 and sequence 2048, overlap is **not** achievable; you need larger $B \cdot S$ or faster interconnects.

## Complete Offloading System

Here's a complete implementation combining CPU and NVMe offloading:

```python
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from collections import OrderedDict
import threading
import queue

class OffloadTier(Enum):
    GPU = "gpu"
    CPU = "cpu"
    NVME = "nvme"

@dataclass
class TensorLocation:
    """Tracks where a tensor is stored."""
    tier: OffloadTier
    offset: int = 0  # For NVMe
    size: int = 0
    is_loading: bool = False
    is_saving: bool = False

@dataclass
class OffloadSystemConfig:
    """Configuration for the complete offloading system."""
    # Memory limits
    gpu_memory_limit: int = 40 * (1024 ** 3)  # 40GB
    cpu_memory_limit: int = 256 * (1024 ** 3)  # 256GB

    # Offloading policy
    offload_optimizer_states: bool = True
    offload_gradients: bool = True
    offload_parameters: bool = False
    parameter_offload_tier: OffloadTier = OffloadTier.CPU

    # Performance
    prefetch_count: int = 2
    pin_memory: bool = True
    num_io_threads: int = 4

    # NVMe settings
    nvme_path: str = "/mnt/nvme/offload"
    use_compression: bool = False

class UnifiedOffloadManager:
    """
    Unified manager for GPU/CPU/NVMe offloading.

    Manages the complete memory hierarchy with automatic
    tier selection and overlap optimization.
    """

    def __init__(
        self,
        model: nn.Module,
        config: OffloadSystemConfig
    ):
        self.model = model
        self.config = config

        # Memory tracking
        self.gpu_used = 0
        self.cpu_used = 0

        # Tensor locations
        self.locations: Dict[str, TensorLocation] = {}

        # Buffers for each tier
        self.gpu_buffers: Dict[str, torch.Tensor] = {}
        self.cpu_buffers: Dict[str, torch.Tensor] = {}
        self.nvme_manager: Optional[NVMeOffloadManager] = None

        # Async operation tracking
        self.pending_loads: Dict[str, threading.Event] = {}
        self.pending_saves: Dict[str, threading.Event] = {}

        # I/O thread pool
        self.io_queue = queue.Queue()
        self.io_threads: List[threading.Thread] = []

        # CUDA streams for overlapped transfers
        self.load_stream = torch.cuda.Stream()
        self.save_stream = torch.cuda.Stream()

        self._init_offloading()
        self._start_io_threads()

    def _init_offloading(self):
        """Initialize offloading based on configuration."""
        for name, param in self.model.named_parameters():
            param_bytes = param.numel() * param.element_size()

            # Determine initial tier based on memory limits
            if self.gpu_used + param_bytes <= self.config.gpu_memory_limit:
                tier = OffloadTier.GPU
                self.gpu_used += param_bytes
                self.gpu_buffers[name] = param.data
            elif self.cpu_used + param_bytes <= self.config.cpu_memory_limit:
                tier = OffloadTier.CPU
                self.cpu_used += param_bytes
                cpu_tensor = param.data.cpu()
                if self.config.pin_memory:
                    cpu_tensor = cpu_tensor.pin_memory()
                self.cpu_buffers[name] = cpu_tensor
            else:
                tier = OffloadTier.NVME
                if self.nvme_manager is None:
                    self.nvme_manager = NVMeOffloadManager(
                        self.model,
                        NVMeConfig(nvme_path=self.config.nvme_path)
                    )

            self.locations[name] = TensorLocation(
                tier=tier,
                size=param_bytes
            )

    def _start_io_threads(self):
        """Start background I/O threads."""
        for _ in range(self.config.num_io_threads):
            t = threading.Thread(target=self._io_worker, daemon=True)
            t.start()
            self.io_threads.append(t)

    def _io_worker(self):
        """Background worker for I/O operations."""
        while True:
            try:
                op, name, data, event = self.io_queue.get(timeout=1.0)

                if op == 'load':
                    self._do_load(name)
                elif op == 'save':
                    self._do_save(name, data)

                if event:
                    event.set()

            except queue.Empty:
                continue

    def _do_load(self, name: str):
        """Perform synchronous load from lower tier."""
        location = self.locations[name]

        if location.tier == OffloadTier.CPU:
            # Load from CPU to GPU
            cpu_tensor = self.cpu_buffers[name]
            with torch.cuda.stream(self.load_stream):
                gpu_tensor = cpu_tensor.cuda(non_blocking=True)
            self.load_stream.synchronize()
            self.gpu_buffers[name] = gpu_tensor

        elif location.tier == OffloadTier.NVME:
            # Load from NVMe to CPU, then GPU
            cpu_tensor = self.nvme_manager.prefetch_buffers.get(name)
            if cpu_tensor is not None:
                self.cpu_buffers[name] = cpu_tensor
                with torch.cuda.stream(self.load_stream):
                    gpu_tensor = cpu_tensor.cuda(non_blocking=True)
                self.load_stream.synchronize()
                self.gpu_buffers[name] = gpu_tensor

    def _do_save(self, name: str, data: torch.Tensor):
        """Perform synchronous save to lower tier."""
        location = self.locations[name]

        if location.tier == OffloadTier.CPU:
            # Save to CPU
            with torch.cuda.stream(self.save_stream):
                self.cpu_buffers[name].copy_(data, non_blocking=True)
            self.save_stream.synchronize()

        elif location.tier == OffloadTier.NVME:
            # Save to CPU, then NVMe
            cpu_tensor = data.cpu()
            self.cpu_buffers[name] = cpu_tensor
            if self.nvme_manager:
                self.nvme_manager.offload_from_gpu(name, cpu_tensor)

    def ensure_gpu(self, name: str) -> torch.Tensor:
        """Ensure parameter is on GPU, loading if necessary."""
        location = self.locations[name]

        if name in self.gpu_buffers:
            return self.gpu_buffers[name]

        # Wait for any pending load
        if name in self.pending_loads:
            self.pending_loads[name].wait()
            del self.pending_loads[name]
            return self.gpu_buffers[name]

        # Synchronous load
        self._do_load(name)
        return self.gpu_buffers[name]

    def prefetch(self, names: List[str]):
        """Prefetch parameters asynchronously."""
        for name in names:
            if name not in self.gpu_buffers and name not in self.pending_loads:
                event = threading.Event()
                self.pending_loads[name] = event
                self.io_queue.put(('load', name, None, event))

    def offload(self, name: str, data: torch.Tensor):
        """Offload parameter from GPU."""
        location = self.locations[name]

        if location.tier == OffloadTier.GPU:
            return  # Already on GPU, no offload needed

        event = threading.Event()
        self.io_queue.put(('save', name, data, event))

        # Remove from GPU buffers
        if name in self.gpu_buffers:
            del self.gpu_buffers[name]

    def get_memory_summary(self) -> Dict[str, Dict[str, int]]:
        """Get memory usage summary by tier."""
        summary = {
            'gpu': {'count': 0, 'bytes': 0},
            'cpu': {'count': 0, 'bytes': 0},
            'nvme': {'count': 0, 'bytes': 0},
        }

        for name, location in self.locations.items():
            tier = location.tier.value
            summary[tier]['count'] += 1
            summary[tier]['bytes'] += location.size

        return summary
```

## Performance Analysis

### Throughput Model

Total step time with offloading:

$$T_{step} = \max(T_{compute}, T_{transfer}) + T_{sync}$$

Where:

- $T_{compute}$: Forward + backward pass time
- $T_{transfer}$: Total data movement time
- $T_{sync}$: Unavoidable synchronization overhead

### Efficiency Metric

Define offload efficiency:

$$\eta_{offload} = \frac{T_{compute}}{T_{step}} = \frac{T_{compute}}{\max(T_{compute}, T_{transfer}) + T_{sync}}$$

Target: $\eta_{offload} > 0.9$ (less than 10% overhead)

### When to Use Each Tier

| Scenario | Recommended Strategy |
|----------|---------------------|
| Model fits in GPU | No offloading |
| Model fits in GPU + optimizer offload | CPU offload optimizer |
| Model doesn't fit in GPU | ZeRO-3 + CPU offload |
| Model doesn't fit in CPU | ZeRO-Infinity (NVMe) |

### Case Study: 175B Model on Single Node

Configuration:

- 8× A100 80GB GPUs
- 2TB CPU memory
- 10TB NVMe array (RAID 0, 4 drives)

Memory requirements:

- Parameters: 350GB (fp16)
- Optimizer states: 1.4TB (fp32 Adam)
- Gradients: 350GB (fp16)
- Activations: Variable

Strategy:
1. ZeRO-3 across 8 GPUs: 350GB / 8 = 44GB parameters per GPU
2. Optimizer offload to CPU: 1.4TB / 8 = 175GB per GPU's share → CPU
3. Gradient offload: ReduceScatter to CPU, optimizer step on CPU
4. Activations: Checkpointing (no offload needed)

Result: Training possible with 80GB GPUs that couldn't hold even the parameters alone.

## Integration with Other Techniques

### Offloading + ZeRO

| ZeRO Stage | Offload Target | Benefit |
|------------|----------------|---------|
| ZeRO-1 | Optimizer to CPU | Optimizer memory → 0 on GPU |
| ZeRO-2 | Optimizer + Gradients to CPU | Only params on GPU |
| ZeRO-3 | All to CPU/NVMe | Minimal GPU memory |

### Offloading + Tensor Parallelism

TP reduces per-GPU memory; offloading reduces further:

$$\text{Memory per GPU} = \frac{\text{Total}}{TP \times \text{ZeRO Stage Factor}} + \text{Activations}$$

With TP=8 and ZeRO-3 on a 175B model:

$$= \frac{350 \text{ GB}}{8 \times 8} + \text{Activations} = 5.5 \text{ GB} + \text{Activations}$$

### Offloading + Pipeline Parallelism

PP naturally stages memory across time. Combine with offloading:

1. Only load parameters for current pipeline stage
2. Offload parameters for idle stages
3. Prefetch next stage's parameters during compute

## Exercises

1. **Bandwidth calculation**: A model has 70B parameters (fp16). Your system has PCIe Gen4 (32 GB/s). The forward pass takes 2 seconds. Can you fully overlap parameter loading? What batch size is needed?

??? success "Solution"
    **Parameter size:**

    $$M_{params} = 70 \times 10^9 \times 2 \text{ bytes} = 140 \text{ GB}$$

    **Time to transfer all parameters:**

    $$T_{transfer} = \frac{140 \text{ GB}}{32 \text{ GB/s}} = 4.375 \text{ seconds}$$

    **Can we fully overlap?**

    No! Transfer time (4.375s) > Forward pass time (2s).

    $$\text{Overlap ratio} = \frac{T_{transfer}}{T_{compute}} = \frac{4.375}{2} = 2.19$$

    We need 2.19× more compute time to hide the transfer.

    **Required batch size:**

    Compute scales linearly with batch size, so:

    $$B_{required} = B_{current} \times \frac{T_{transfer}}{T_{compute}} = B_{current} \times 2.19$$

    If the current forward pass (2s) is with batch size $B$, we need:

    $$B_{new} \geq 2.19 \times B$$

    **Alternative: Per-layer streaming**

    If we stream layer-by-layer (with ~80 layers for 70B):

    $$M_{layer} = \frac{140 \text{ GB}}{80} = 1.75 \text{ GB}$$

    $$T_{layer\_transfer} = \frac{1.75 \text{ GB}}{32 \text{ GB/s}} = 54.7 \text{ ms}$$

    $$T_{layer\_compute} = \frac{2000 \text{ ms}}{80} = 25 \text{ ms}$$

    Still transfer-bound per layer. Need double buffering + 2× batch size.

    | Configuration | Batch Multiplier | Fully Overlapped? |
    |---------------|------------------|-------------------|
    | Current | 1× | No |
    | 2× batch | 2× | Almost (80ms vs 55ms/layer) |
    | 2.5× batch | 2.5× | $\boxed{\text{Yes}}$ |

2. **Memory tiering**: Design an offloading strategy for a 500B parameter model on a node with 8× 80GB GPUs, 2TB CPU RAM, and 20TB NVMe. Calculate memory requirements for each tier.

??? success "Solution"
    **System capacity:**

    | Tier | Capacity | Bandwidth | Latency |
    |------|----------|-----------|---------|
    | GPU | 8 × 80 = 640 GB | 3.35 TB/s (HBM3) | ~ns |
    | CPU | 2 TB | ~200 GB/s (DDR5) | ~100ns |
    | NVMe | 20 TB | ~28 GB/s (4× Gen4) | ~μs |

    **Model memory requirements (training with AdamW):**

    $$M_{total} = 16\Psi = 16 \times 500 \times 10^9 = 8 \text{ TB}$$

    | Component | Size | Formula |
    |-----------|------|---------|
    | Parameters (fp16) | 1 TB | $2\Psi$ |
    | Gradients (fp16) | 1 TB | $2\Psi$ |
    | Optimizer (fp32) | 6 TB | $12\Psi$ |
    | **Total** | **8 TB** | $16\Psi$ |

    **Tiering Strategy:**

    ```
    ┌─────────────────────────────────────────────────────────┐
    │  GPU (640 GB total, ~80 GB per GPU)                     │
    │  ├── Active layer parameters: ~12 GB (2 layers)         │
    │  ├── Active gradients: ~12 GB                           │
    │  ├── Activation memory: ~40 GB                          │
    │  └── Working buffers: ~16 GB                            │
    ├─────────────────────────────────────────────────────────┤
    │  CPU RAM (2 TB)                                         │
    │  ├── Full parameters (prefetch buffer): 1 TB            │
    │  ├── Full gradients (accumulation buffer): 1 TB         │
    │  └── Reserved for OS/framework: 200 GB                  │
    ├─────────────────────────────────────────────────────────┤
    │  NVMe (20 TB)                                           │
    │  ├── Optimizer states: 6 TB                             │
    │  ├── Checkpoint storage: ~10 TB                         │
    │  └── Spare capacity: ~4 TB                              │
    └─────────────────────────────────────────────────────────┘
    ```

    **Data flow during training:**

    1. **Forward pass**: Stream parameters GPU ← CPU ← NVMe
    2. **Backward pass**: Compute gradients, accumulate in CPU
    3. **Optimizer step**: Update on NVMe (chunked), apply to parameters

    **Timing analysis (per step):**

    | Operation | Data Size | Bandwidth | Time |
    |-----------|-----------|-----------|------|
    | Params: NVMe → CPU | 1 TB | 28 GB/s | 36s |
    | Params: CPU → GPU | 1 TB | 32 GB/s | 31s |
    | Grads: GPU → CPU | 1 TB | 32 GB/s | 31s |
    | Opt update (NVMe) | 6 TB | 28 GB/s | 214s |

    **Optimization: Overlap everything**

    With proper prefetching and streaming:

    $$T_{step} \approx \max(T_{compute}, T_{transfer})$$

    | Configuration | Practical? |
    |---------------|------------|
    | All on GPU | No (need 8 TB, have 640 GB) |
    | GPU + CPU only | No (need 8 TB, have 2.6 TB) |
    | GPU + CPU + NVMe | $\boxed{\text{Yes}}$ |

3. **Overlap efficiency**: If compute time is 100ms and transfer time is 80ms with 5ms sync overhead, what is the offload efficiency? How would you improve it?

??? success "Solution"
    **Offload efficiency definition:**

    $$\eta_{offload} = \frac{T_{compute}}{T_{total}}$$

    Where $T_{total}$ is the actual wall-clock time per step.

    **Case 1: No overlap (sequential)**

    $$T_{total}^{seq} = T_{compute} + T_{transfer} + T_{sync} = 100 + 80 + 5 = 185 \text{ ms}$$

    $$\eta_{offload}^{seq} = \frac{100}{185} = 54\%$$

    **Case 2: Full overlap (pipelined)**

    With perfect overlap, transfer happens during compute:

    $$T_{total}^{overlap} = \max(T_{compute}, T_{transfer}) + T_{sync}$$

    $$T_{total}^{overlap} = \max(100, 80) + 5 = 105 \text{ ms}$$

    $$\eta_{offload}^{overlap} = \frac{100}{105} = \boxed{95.2\%}$$

    **Improvement strategies:**

    | Strategy | Effect | New Efficiency |
    |----------|--------|----------------|
    | Async transfers | Hide 80ms behind 100ms | 95.2% |
    | Reduce sync overhead | 5ms → 1ms | 99.0% |
    | Larger batches | Increase $T_{compute}$ | Higher |
    | Pinned memory | Reduce sync overhead | ~98% |
    | Double buffering | Enable full overlap | 95.2% |

    **Implementation for maximum overlap:**

    ```python
    class OverlappedOffloader:
        def __init__(self, model, device):
            self.stream_compute = torch.cuda.Stream()
            self.stream_transfer = torch.cuda.Stream()
            self.buffers = [None, None]  # Double buffer

        def step(self, layer_idx, input_tensor):
            current_buf = layer_idx % 2
            next_buf = (layer_idx + 1) % 2

            # Prefetch next layer (overlapped with compute)
            with torch.cuda.stream(self.stream_transfer):
                self.buffers[next_buf] = self.load_layer(layer_idx + 1)

            # Compute on current layer
            with torch.cuda.stream(self.stream_compute):
                output = self.forward_layer(layer_idx, input_tensor,
                                           self.buffers[current_buf])

            # Minimal sync point
            self.stream_compute.synchronize()
            return output
    ```

    **Sync overhead reduction:**

    The 5ms sync overhead can be reduced by:

    1. **CUDA events** instead of stream sync: ~0.1ms
    2. **Pinned memory**: Eliminates page fault overhead
    3. **Fewer sync points**: Sync every $N$ layers instead of every layer

4. **Double buffering**: Implement triple buffering for parameter loading. When would this be beneficial over double buffering?

??? success "Solution"
    **Triple buffering implementation:**

    ```python
    import torch
    from collections import deque
    from typing import Optional, List, Callable

    class TripleBufferedLoader:
        """
        Triple buffering for parameter loading.

        Maintains 3 GPU buffers:
        - Buffer 0: Currently being used for compute
        - Buffer 1: Next layer (ready for compute)
        - Buffer 2: Being filled from CPU/NVMe
        """

        def __init__(
            self,
            layer_size: int,
            dtype: torch.dtype = torch.float16,
            device: str = "cuda"
        ):
            # Allocate 3 GPU buffers
            self.buffers = [
                torch.empty(layer_size, dtype=dtype, device=device)
                for _ in range(3)
            ]

            # Transfer streams
            self.transfer_streams = [
                torch.cuda.Stream() for _ in range(2)
            ]
            self.compute_stream = torch.cuda.Stream()

            # Buffer state tracking
            self.buffer_ready = [False, False, False]
            self.current_transfer_stream = 0

        def prefetch_layer(
            self,
            layer_idx: int,
            buffer_idx: int,
            source_tensor: torch.Tensor
        ):
            """Async prefetch a layer into buffer."""
            stream = self.transfer_streams[self.current_transfer_stream]
            self.current_transfer_stream = 1 - self.current_transfer_stream

            with torch.cuda.stream(stream):
                self.buffers[buffer_idx].copy_(source_tensor, non_blocking=True)

            self.buffer_ready[buffer_idx] = True
            return stream

        def forward_with_triple_buffer(
            self,
            layers: List[Callable],
            cpu_params: List[torch.Tensor],
            input_tensor: torch.Tensor
        ) -> torch.Tensor:
            """
            Execute forward pass with triple buffering.
            """
            num_layers = len(layers)
            x = input_tensor

            # Initial prefetch: load first 2 layers
            events = [None, None]
            events[0] = self.prefetch_layer(0, 0, cpu_params[0])
            if num_layers > 1:
                events[1] = self.prefetch_layer(1, 1, cpu_params[1])

            for i in range(num_layers):
                compute_buf = i % 3
                prefetch_buf = (i + 2) % 3

                # Wait for current layer's transfer
                if events[i % 2] is not None:
                    events[i % 2].synchronize()

                # Start prefetching layer i+2
                if i + 2 < num_layers:
                    events[i % 2] = self.prefetch_layer(
                        i + 2, prefetch_buf, cpu_params[i + 2]
                    )

                # Compute layer i
                with torch.cuda.stream(self.compute_stream):
                    x = layers[i](x, self.buffers[compute_buf])

            self.compute_stream.synchronize()
            return x
    ```

    **When is triple buffering beneficial?**

    | Scenario | Double Buffer | Triple Buffer | Winner |
    |----------|---------------|---------------|--------|
    | $T_{transfer} < T_{compute}$ | Full overlap | No benefit | Draw |
    | $T_{transfer} = T_{compute}$ | Full overlap | No benefit | Draw |
    | $T_{transfer} > T_{compute}$ | Compute-bound gaps | Smoother pipeline | Triple |
    | Variable transfer times | May stall | Extra buffer absorbs variance | Triple |
    | High jitter/contention | Stalls on delays | Tolerates delays | Triple |

    **Mathematical analysis:**

    Let $T_c$ = compute time, $T_t$ = transfer time.

    **Double buffering:**

    $$T_{step}^{double} = \max(T_c, T_t)$$

    Stalls when transfer for layer $i+1$ isn't complete when layer $i$ finishes.

    **Triple buffering:**

    $$T_{step}^{triple} = \max(T_c, \frac{T_t}{2})$$

    With 2 transfers in flight, we effectively have twice the transfer bandwidth.

    **When triple buffering helps:**

    $$T_t > T_c \text{ (transfer-bound scenario)}$$

    **Example:**

    | Metric | Double Buffer | Triple Buffer |
    |--------|---------------|---------------|
    | $T_c = 50$ ms | | |
    | $T_t = 80$ ms | | |
    | Per-layer time | 80 ms | 50 ms |
    | Speedup | 1× | $\boxed{1.6\times}$ |

    Triple buffering is most beneficial when transfer time exceeds compute time, allowing the pipeline to stay compute-bound rather than transfer-bound.

5. **Compression trade-off**: LZ4 achieves 1.5× compression with 2GB/s compression throughput. For a 10GB tensor on NVMe (7GB/s), is compression worth it? Calculate total time with and without compression.

??? success "Solution"
    **Without compression:**

    $$T_{no\_compress} = \frac{10 \text{ GB}}{7 \text{ GB/s}} = 1.43 \text{ seconds}$$

    **With compression:**

    Compressed size:

    $$M_{compressed} = \frac{10 \text{ GB}}{1.5} = 6.67 \text{ GB}$$

    Compression time (on CPU):

    $$T_{compress} = \frac{10 \text{ GB}}{2 \text{ GB/s}} = 5.0 \text{ seconds}$$

    Transfer time:

    $$T_{transfer} = \frac{6.67 \text{ GB}}{7 \text{ GB/s}} = 0.95 \text{ seconds}$$

    Total with compression:

    $$T_{with\_compress} = T_{compress} + T_{transfer} = 5.0 + 0.95 = 5.95 \text{ seconds}$$

    **Comparison:**

    | Approach | Time | Speedup |
    |----------|------|---------|
    | No compression | 1.43s | 1× |
    | With compression | 5.95s | 0.24× (slower!) |

    $$\boxed{\text{Compression is NOT worth it}}$$

    **Break-even analysis:**

    Compression is beneficial when:

    $$T_{compress} + T_{transfer}^{compressed} < T_{transfer}^{uncompressed}$$

    $$\frac{M}{C_{throughput}} + \frac{M/R}{B_{NVMe}} < \frac{M}{B_{NVMe}}$$

    Where $R$ = compression ratio, $C$ = compression throughput.

    Solving for when compression helps:

    $$\frac{1}{C_{throughput}} < \frac{1}{B_{NVMe}} \times (1 - \frac{1}{R})$$

    $$C_{throughput} > \frac{B_{NVMe}}{1 - 1/R}$$

    For our case ($R = 1.5$, $B = 7$ GB/s):

    $$C_{required} > \frac{7}{1 - 1/1.5} = \frac{7}{0.33} = 21 \text{ GB/s}$$

    We need compression throughput > 21 GB/s, but LZ4 only provides 2 GB/s.

    **When compression IS beneficial:**

    | Scenario | Why |
    |----------|-----|
    | Very slow storage (HDD ~200 MB/s) | Transfer dominates |
    | High compression ratio (>10×) | Significant size reduction |
    | GPU-accelerated compression | Fast enough to overlap |
    | Sparse tensors | Extreme compression possible |

    **GPU-accelerated compression example:**

    With nvCOMP (GPU LZ4, ~50 GB/s):

    $$T_{compress}^{GPU} = \frac{10}{50} = 0.2 \text{s}$$

    $$T_{total}^{GPU} = 0.2 + 0.95 = 1.15 \text{s} < 1.43\text{s}$$

    GPU compression would provide $\boxed{1.24\times}$ speedup.

6. **Prefetch depth**: Derive the optimal prefetch depth given layer compute time $T_c$, transfer bandwidth $B$, and layer size $S$. When does increasing prefetch depth not help?

??? success "Solution"
    **Definitions:**

    - $T_c$ = compute time per layer
    - $B$ = transfer bandwidth (bytes/second)
    - $S$ = layer size (bytes)
    - $k$ = prefetch depth (number of layers prefetched ahead)

    **Transfer time per layer:**

    $$T_t = \frac{S}{B}$$

    **Optimal prefetch depth derivation:**

    For no stalls, we need enough time to transfer the next layer before we need it:

    $$k \times T_c \geq T_t$$

    Solving for minimum $k$:

    $$k^* = \left\lceil \frac{T_t}{T_c} \right\rceil = \left\lceil \frac{S}{B \times T_c} \right\rceil$$

    $$\boxed{k^* = \left\lceil \frac{S}{B \cdot T_c} \right\rceil}$$

    **Example calculation:**

    | Parameter | Value |
    |-----------|-------|
    | Layer size $S$ | 2 GB |
    | PCIe bandwidth $B$ | 32 GB/s |
    | Compute time $T_c$ | 50 ms |

    $$T_t = \frac{2}{32} = 62.5 \text{ ms}$$

    $$k^* = \left\lceil \frac{62.5}{50} \right\rceil = \left\lceil 1.25 \right\rceil = 2$$

    Need prefetch depth of 2 to avoid stalls.

    **When increasing prefetch depth does NOT help:**

    | Scenario | Why |
    |----------|-----|
    | Already compute-bound ($T_t < T_c$) | $k=1$ is sufficient |
    | Memory-limited | Can't allocate more buffers |
    | Diminishing returns | Beyond $k^*$, no improvement |
    | Variable layer sizes | May need adaptive $k$ |

    **Mathematical analysis of diminishing returns:**

    Effective throughput as function of $k$:

    $$\text{Throughput}(k) = \begin{cases}
    \frac{1}{T_t / k} & \text{if } k < k^* \text{ (transfer-bound)} \\
    \frac{1}{T_c} & \text{if } k \geq k^* \text{ (compute-bound)}
    \end{cases}$$

    Once $k \geq k^*$, we're compute-bound and more prefetching doesn't help.

    **Memory cost:**

    $$M_{prefetch} = (k + 1) \times S$$

    Extra memory for $k$ prefetched buffers plus one compute buffer.

    | Prefetch Depth | Throughput | Memory Cost |
    |----------------|------------|-------------|
    | $k = 0$ | $\frac{1}{T_c + T_t}$ | $S$ |
    | $k = 1$ | $\frac{1}{\max(T_c, T_t)}$ | $2S$ |
    | $k = k^*$ | $\frac{1}{T_c}$ | $(k^*+1)S$ |
    | $k > k^*$ | $\frac{1}{T_c}$ (no gain) | Wasted memory |

    **Adaptive prefetch strategy:**

    ```python
    def compute_optimal_prefetch(
        layer_sizes: List[int],
        bandwidth: float,
        compute_times: List[float]
    ) -> List[int]:
        """
        Compute per-layer optimal prefetch depth.
        Handles variable layer sizes and compute times.
        """
        prefetch_depths = []

        for size, t_compute in zip(layer_sizes, compute_times):
            t_transfer = size / bandwidth
            k_optimal = max(1, int(np.ceil(t_transfer / t_compute)))
            prefetch_depths.append(k_optimal)

        return prefetch_depths
    ```

    **Key insight:** Optimal prefetch is just enough to hide transfer latency—more wastes memory without improving throughput.

## Key Takeaways

1. **Memory is hierarchical**: GPU → CPU → NVMe, with decreasing bandwidth and increasing capacity.

2. **Overlap is essential**: Without overlap, offloading kills performance. With overlap, it's nearly free.

3. **Pinned memory matters**: Pinned (page-locked) CPU memory enables async GPU transfers.

4. **Prefetching hides latency**: Load next layer while computing current layer.

5. **ZeRO + Offload is powerful**: Combines memory reduction across GPUs with expansion via CPU/NVMe.

6. **Know your bandwidths**: PCIe (32 GB/s), NVMe (7 GB/s). Design accordingly.

7. **Compression can help**: For NVMe, light compression often improves effective bandwidth.
