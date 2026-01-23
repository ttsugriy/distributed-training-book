---
title: "Profiling Distributed Training"
subtitle: "Instrumenting, Measuring, and Optimizing Parallel Systems"
---

<div class="chapter-opener" markdown>
A distributed training run consumes thousands of GPU-hours. Yet most practitioners have no idea where that time goes. Profiling transforms intuition into data, revealing whether you're compute-bound, communication-bound, or simply waiting.
</div>

<div class="investigation-question" markdown>
**The Question**: Your 64-GPU training run achieves 35% MFU (Model FLOPS Utilization). Where is the other 65%? Is it communication? Memory bandwidth? Kernel launch overhead? Idle time? Without measurement, optimization is guesswork.
</div>

## The Profiling Imperative

At scale, inefficiency compounds. A 10% inefficiency on one GPU becomes 640 GPU-hours wasted per day on 64 GPUs. Understanding *exactly* where time goes is essential.

### What We Measure

Distributed training profiling examines four domains:

**1. Compute**
- Kernel execution time
- Tensor Core utilization
- Memory bandwidth utilization
- FLOPS achieved vs theoretical peak

**2. Communication**
- Collective operation duration
- Network bandwidth utilization
- Message sizes and frequencies
- Overlap with computation

**3. Memory**
- Peak allocation
- Fragmentation
- Data movement (HBM ↔ device, host ↔ device)
- Cache hit rates

**4. Orchestration**
- Kernel launch overhead
- Python/framework overhead
- Synchronization waits
- Load imbalance across workers

### The Time Budget

Every training step has a fixed time budget:

$$T_{\text{step}} = T_{\text{forward}} + T_{\text{backward}} + T_{\text{comm}} + T_{\text{optimizer}} + T_{\text{other}}$$

The goal: minimize $T_{\text{step}}$ while maintaining model quality.

With perfect overlap:
$$T_{\text{step}} = \max(T_{\text{compute}}, T_{\text{comm}})$$

Without overlap:
$$T_{\text{step}} = T_{\text{compute}} + T_{\text{comm}}$$

Profiling reveals where you are on this spectrum.

## Profiling Tools

### NVIDIA Nsight Systems

The gold standard for GPU profiling. Captures:

- CUDA kernel execution
- Memory operations
- NCCL collective calls
- CPU activity
- Inter-GPU communication

**Basic Usage**:
```bash
nsys profile -o trace \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --sample=cpu \
    python train.py
```

**For Distributed Training**:
```bash
# On each rank
nsys profile -o trace_rank${RANK} \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    torchrun --nproc_per_node=8 train.py
```

**Key Command Options**:
```bash
# Capture NCCL operations explicitly
nsys profile --trace=cuda,nvtx,osrt,nccl \
    --capture-range=cudaProfilerApi \
    python train.py

# Limit trace duration (avoid huge files)
nsys profile --duration=60 -o trace python train.py

# Export to multiple formats
nsys export -o trace.json --type=json trace.nsys-rep
```

### PyTorch Profiler

Built-in profiling with TensorBoard integration:

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    schedule=profiler.schedule(
        wait=1,      # Skip first step
        warmup=1,    # Warmup step
        active=3,    # Profile 3 steps
        repeat=1
    ),
    on_trace_ready=profiler.tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 5:
            break
        train_step(batch)
        prof.step()
```

**Distributed-Aware Profiling**:
```python
import torch.distributed as dist

def profile_distributed_training(model, dataloader, num_steps=5):
    """Profile distributed training with rank-aware output."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    profiler_config = profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler(
            f'./logs/rank_{rank}'
        ),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,  # Estimate FLOPS
    )

    with profiler_config as prof:
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break

            # Add NVTX markers for visibility
            with torch.cuda.nvtx.range(f"step_{step}"):
                with torch.cuda.nvtx.range("forward"):
                    output = model(batch)

                with torch.cuda.nvtx.range("backward"):
                    loss = compute_loss(output)
                    loss.backward()

                with torch.cuda.nvtx.range("optimizer"):
                    optimizer.step()
                    optimizer.zero_grad()

            prof.step()

    # Print summary for rank 0
    if rank == 0:
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=20
        ))
```

### NVTX Annotations

NVIDIA Tools Extension provides manual annotation:

```python
import torch.cuda.nvtx as nvtx

class NVTXAnnotatedModule(nn.Module):
    """Module with NVTX range annotations."""

    def __init__(self, module, name):
        super().__init__()
        self.module = module
        self.name = name

    def forward(self, x):
        with nvtx.range(self.name):
            return self.module(x)

# Annotate model layers
def annotate_model(model):
    """Add NVTX annotations to all layers."""
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            annotate_model(module)
        else:
            setattr(model, name, NVTXAnnotatedModule(module, name))
```

**Fine-Grained Collective Annotation**:
```python
import torch.distributed as dist

class ProfiledAllReduce:
    """AllReduce with detailed profiling."""

    def __init__(self, process_group=None):
        self.process_group = process_group
        self.call_count = 0
        self.total_bytes = 0
        self.total_time = 0

    def __call__(self, tensor):
        size_bytes = tensor.numel() * tensor.element_size()
        self.total_bytes += size_bytes

        with nvtx.range(f"AllReduce_{self.call_count}_{size_bytes/1e6:.1f}MB"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            dist.all_reduce(tensor, group=self.process_group)
            end.record()

            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            self.total_time += elapsed

        self.call_count += 1
        return tensor

    def summary(self):
        return {
            'calls': self.call_count,
            'total_bytes_GB': self.total_bytes / 1e9,
            'total_time_ms': self.total_time,
            'avg_bandwidth_GBps': (self.total_bytes / 1e9) / (self.total_time / 1e3)
        }
```

### NCCL Debug Output

Enable detailed NCCL logging:

```bash
# Basic info
export NCCL_DEBUG=INFO

# Detailed subsystem logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Trace all operations
export NCCL_DEBUG=TRACE

# Log to file per rank
export NCCL_DEBUG_FILE=nccl_log_%h_%p.txt
```

**Interpreting NCCL Logs**:
```
# Topology detection
NCCL INFO Ring 00 : 0 -> 1 -> 2 -> 3 -> 0

# Algorithm selection
NCCL INFO Channel 00/08 : 0[0] -> 1[1] via P2P/CUMEM

# Operation timing (with TRACE)
NCCL INFO AllReduce: opCount 42 bytes 104857600 datatype 7 op 0
```

### Memory Profiling

Track memory allocation and fragmentation:

```python
import torch

class MemoryProfiler:
    """Track GPU memory usage during training."""

    def __init__(self, device=None):
        self.device = device or torch.cuda.current_device()
        self.snapshots = []

    def snapshot(self, label=""):
        """Capture current memory state."""
        torch.cuda.synchronize()

        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_allocated = torch.cuda.max_memory_allocated(self.device)

        self.snapshots.append({
            'label': label,
            'allocated_GB': allocated / 1e9,
            'reserved_GB': reserved / 1e9,
            'max_allocated_GB': max_allocated / 1e9,
            'fragmentation': 1 - (allocated / reserved) if reserved > 0 else 0
        })

    def reset_peak(self):
        """Reset peak memory tracking."""
        torch.cuda.reset_peak_memory_stats(self.device)

    def report(self):
        """Print memory timeline."""
        print("\n=== Memory Profile ===")
        print(f"{'Label':<30} {'Allocated':>12} {'Reserved':>12} {'Peak':>12} {'Frag':>8}")
        print("-" * 74)
        for snap in self.snapshots:
            print(f"{snap['label']:<30} "
                  f"{snap['allocated_GB']:>10.2f}GB "
                  f"{snap['reserved_GB']:>10.2f}GB "
                  f"{snap['max_allocated_GB']:>10.2f}GB "
                  f"{snap['fragmentation']:>7.1%}")

# Usage during training
mem_profiler = MemoryProfiler()
mem_profiler.reset_peak()

mem_profiler.snapshot("after_model_init")
# ... training ...
mem_profiler.snapshot("after_forward")
mem_profiler.snapshot("after_backward")
mem_profiler.snapshot("after_optimizer_step")

mem_profiler.report()
```

**Detailed Memory Breakdown**:
```python
def memory_breakdown():
    """Get detailed memory breakdown by category."""
    snapshot = torch.cuda.memory_snapshot()

    categories = {}
    for block in snapshot:
        # Categorize by allocation context
        if 'frames' in block and block['frames']:
            context = block['frames'][0]['filename']
        else:
            context = 'unknown'

        if context not in categories:
            categories[context] = 0
        categories[context] += block['total_size']

    # Sort by size
    sorted_cats = sorted(categories.items(), key=lambda x: -x[1])

    print("\n=== Memory by Source ===")
    total = sum(categories.values())
    for source, size in sorted_cats[:10]:
        print(f"{source}: {size/1e9:.2f}GB ({size/total*100:.1f}%)")
```

## Interpreting Traces

### Nsight Systems Timeline

A typical distributed training trace shows:

```
Time →
|--Forward--|--Backward----|--AllReduce--|--Optimizer--|
       |                   |<--Overlap-->|
       |--Compute Stream---|
       |---Comm Stream-----|
```

**Key Patterns to Look For**:

1. **Gaps between kernels**: Indicates CPU overhead or kernel launch latency
2. **Sequential compute and comm**: No overlap, potential for optimization
3. **Long collective tails**: Straggler workers or network contention
4. **Memory copy operations**: Potential for prefetching or pinned memory

### Identifying Bottlenecks

```python
class BottleneckAnalyzer:
    """Analyze profiling data to identify bottlenecks."""

    def __init__(self):
        self.timings = {
            'forward': [],
            'backward': [],
            'allreduce': [],
            'optimizer': [],
            'data_loading': [],
        }

    def record(self, phase, duration_ms):
        self.timings[phase].append(duration_ms)

    def analyze(self):
        """Identify the primary bottleneck."""
        results = {}

        for phase, times in self.timings.items():
            if times:
                results[phase] = {
                    'mean_ms': sum(times) / len(times),
                    'max_ms': max(times),
                    'min_ms': min(times),
                    'variance': self._variance(times),
                }

        total = sum(r['mean_ms'] for r in results.values())

        print("\n=== Bottleneck Analysis ===")
        print(f"{'Phase':<15} {'Mean':>10} {'Max':>10} {'Variance':>10} {'%Total':>10}")
        print("-" * 55)

        for phase, stats in sorted(results.items(), key=lambda x: -x[1]['mean_ms']):
            pct = stats['mean_ms'] / total * 100 if total > 0 else 0
            print(f"{phase:<15} {stats['mean_ms']:>8.2f}ms {stats['max_ms']:>8.2f}ms "
                  f"{stats['variance']:>8.2f}ms {pct:>9.1f}%")

        # Identify bottleneck type
        compute_time = results.get('forward', {}).get('mean_ms', 0) + \
                      results.get('backward', {}).get('mean_ms', 0)
        comm_time = results.get('allreduce', {}).get('mean_ms', 0)

        print("\n=== Diagnosis ===")
        if comm_time > compute_time * 1.2:
            print("COMMUNICATION BOUND: Collective operations dominate.")
            print("  Recommendations:")
            print("  - Increase batch size to amortize communication")
            print("  - Enable gradient bucketing if not already")
            print("  - Check for network bottlenecks")
        elif compute_time > comm_time * 1.2:
            print("COMPUTE BOUND: Forward/backward computation dominates.")
            print("  Recommendations:")
            print("  - Good! System is well-utilized")
            print("  - Consider mixed precision if not already using")
            print("  - May be able to overlap more communication")
        else:
            print("BALANCED: Compute and communication roughly equal.")
            print("  Recommendations:")
            print("  - Verify overlap is enabled and effective")
            print("  - This is often the optimal regime")

    def _variance(self, values):
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
```

### Reading Collective Timelines

```
AllReduce Timeline (ideal overlap):
GPU 0: |--Compute--|--AllReduce--|        |--Compute--|
GPU 1: |--Compute--|--AllReduce--|        |--Compute--|
                   |<--overlap-->|

AllReduce Timeline (straggler):
GPU 0: |--Compute--|--AllReduce--|---wait---|--Compute--|
GPU 1: |--Compute--|------AllReduce--------|--Compute--|
                              ^
                        Slow GPU 1 affects all
```

**Detecting Stragglers**:
```python
import torch.distributed as dist

def detect_stragglers(num_iterations=10):
    """Measure timing variance across ranks."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_times = []

    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()

        # Simulate work with some variation
        dummy_work()

        torch.cuda.synchronize()
        local_time = time.time() - start
        local_times.append(local_time)

        # Gather all times to rank 0
        all_times = [torch.zeros(1) for _ in range(world_size)]
        dist.all_gather(all_times, torch.tensor([local_time]))

        if rank == 0:
            times = [t.item() for t in all_times]
            mean_time = sum(times) / len(times)
            max_time = max(times)
            straggler_rank = times.index(max_time)

            if max_time > mean_time * 1.1:  # 10% slower
                print(f"Straggler detected: rank {straggler_rank} "
                      f"({max_time:.3f}s vs mean {mean_time:.3f}s)")
```

## MFU and Efficiency Metrics

### Model FLOPS Utilization

MFU measures actual compute efficiency:

$$\text{MFU} = \frac{\text{Achieved FLOPS}}{\text{Peak FLOPS}}$$

**Calculating MFU**:
```python
def calculate_mfu(
    model_flops_per_sample: int,
    batch_size: int,
    step_time_seconds: float,
    num_gpus: int,
    peak_flops_per_gpu: float
) -> float:
    """
    Calculate Model FLOPS Utilization.

    Args:
        model_flops_per_sample: Forward + backward FLOPs per sample
        batch_size: Global batch size
        step_time_seconds: Time for one training step
        num_gpus: Number of GPUs used
        peak_flops_per_gpu: Theoretical peak FLOPS (e.g., 312 TFLOPS for A100)

    Returns:
        MFU as a fraction (0-1)
    """
    # Total FLOPs for this step
    total_flops = model_flops_per_sample * batch_size

    # Achieved FLOPS
    achieved_flops = total_flops / step_time_seconds

    # Peak system FLOPS
    peak_flops = peak_flops_per_gpu * num_gpus

    return achieved_flops / peak_flops

# Example: GPT-3 175B on 1024 A100s
model_flops = 6 * 175e9 * 2048  # ~2.15e15 FLOPs per sample (6 * N * seq_len)
batch_size = 1024
step_time = 60.0  # seconds
num_gpus = 1024
peak_per_gpu = 312e12  # A100 FP16 peak

mfu = calculate_mfu(model_flops, batch_size, step_time, num_gpus, peak_per_gpu)
print(f"MFU: {mfu:.1%}")  # Typically 30-50% for large models
```

### Hardware FLOPS Utilization (HFU)

HFU includes rematerialization:

$$\text{HFU} = \frac{\text{Achieved FLOPS including recomputation}}{\text{Peak FLOPS}}$$

```python
def calculate_hfu(
    model_flops_per_sample: int,
    batch_size: int,
    step_time_seconds: float,
    num_gpus: int,
    peak_flops_per_gpu: float,
    recomputation_ratio: float = 1.0  # 1.0 = no recomputation, 2.0 = full recomputation
) -> float:
    """
    Calculate Hardware FLOPS Utilization (includes recomputation).

    The recomputation_ratio accounts for activation checkpointing.
    """
    # Effective FLOPs including recomputation
    effective_flops = model_flops_per_sample * batch_size * recomputation_ratio

    achieved_flops = effective_flops / step_time_seconds
    peak_flops = peak_flops_per_gpu * num_gpus

    return achieved_flops / peak_flops
```

### Communication Efficiency

```python
def calculate_comm_efficiency(
    bytes_communicated: int,
    comm_time_seconds: float,
    network_bandwidth_bytes_per_sec: float
) -> float:
    """Calculate communication bandwidth efficiency."""
    achieved_bandwidth = bytes_communicated / comm_time_seconds
    return achieved_bandwidth / network_bandwidth_bytes_per_sec

# Example: AllReduce of 1GB gradients over 100Gbps InfiniBand
bytes_comm = 1e9
comm_time = 0.1  # 100ms
network_bw = 12.5e9  # 100Gbps = 12.5 GB/s

efficiency = calculate_comm_efficiency(bytes_comm, comm_time, network_bw)
print(f"Communication efficiency: {efficiency:.1%}")
```

### Overlap Efficiency

Measure how well computation and communication overlap:

```python
class OverlapEfficiencyTracker:
    """Track overlap between compute and communication."""

    def __init__(self):
        self.compute_time = 0
        self.comm_time = 0
        self.total_time = 0

    def record_step(self, compute_ms, comm_ms, step_ms):
        self.compute_time += compute_ms
        self.comm_time += comm_ms
        self.total_time += step_ms

    def overlap_efficiency(self):
        """
        Calculate overlap efficiency.

        Perfect overlap: total = max(compute, comm)
        No overlap: total = compute + comm

        Efficiency = 1 - (actual - theoretical_min) / (theoretical_max - theoretical_min)
        """
        theoretical_min = max(self.compute_time, self.comm_time)
        theoretical_max = self.compute_time + self.comm_time

        if theoretical_max == theoretical_min:
            return 1.0

        return 1 - (self.total_time - theoretical_min) / (theoretical_max - theoretical_min)

    def report(self):
        print(f"\n=== Overlap Efficiency ===")
        print(f"Total compute time: {self.compute_time:.2f}ms")
        print(f"Total comm time: {self.comm_time:.2f}ms")
        print(f"Total wall time: {self.total_time:.2f}ms")
        print(f"Theoretical min (perfect overlap): {max(self.compute_time, self.comm_time):.2f}ms")
        print(f"Theoretical max (no overlap): {self.compute_time + self.comm_time:.2f}ms")
        print(f"Overlap efficiency: {self.overlap_efficiency():.1%}")
```

## The Alpha-Beta Model in Practice

### Measuring α and β

The alpha-beta model predicts collective time:

$$T = \alpha + \frac{n}{\beta}$$

**Measuring on Your Hardware**:
```python
import time

def measure_alpha_beta(
    process_group,
    sizes_bytes: list,
    num_warmup: int = 5,
    num_measure: int = 20
) -> tuple:
    """
    Measure alpha (latency) and beta (bandwidth) for a process group.

    Returns:
        (alpha_seconds, beta_bytes_per_second)
    """
    rank = dist.get_rank()
    times = []

    for size in sizes_bytes:
        tensor = torch.zeros(size // 4, dtype=torch.float32, device='cuda')

        # Warmup
        for _ in range(num_warmup):
            dist.all_reduce(tensor, group=process_group)
            torch.cuda.synchronize()

        # Measure
        elapsed = []
        for _ in range(num_measure):
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_reduce(tensor, group=process_group)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)

        avg_time = sum(elapsed) / len(elapsed)
        times.append((size, avg_time))

    if rank == 0:
        # Linear regression: T = alpha + size/beta
        # Solve for alpha and beta using least squares
        import numpy as np
        sizes = np.array([s for s, _ in times])
        measured = np.array([t for _, t in times])

        # Design matrix: [1, size]
        X = np.column_stack([np.ones_like(sizes), sizes])
        # Solve: [alpha, 1/beta] = (X^T X)^{-1} X^T y
        coeffs = np.linalg.lstsq(X, measured, rcond=None)[0]

        alpha = coeffs[0]
        beta = 1 / coeffs[1]

        print(f"\n=== Alpha-Beta Measurement ===")
        print(f"Alpha (latency): {alpha*1e6:.2f} μs")
        print(f"Beta (bandwidth): {beta/1e9:.2f} GB/s")
        print(f"\nPredicted times:")
        for size, actual in times:
            predicted = alpha + size / beta
            error = abs(predicted - actual) / actual * 100
            print(f"  {size/1e6:.1f}MB: predicted={predicted*1e3:.2f}ms, "
                  f"actual={actual*1e3:.2f}ms, error={error:.1f}%")

        return alpha, beta

    return None, None

# Measure with various sizes
sizes = [1024, 64*1024, 256*1024, 1024*1024, 4*1024*1024, 16*1024*1024, 64*1024*1024]
alpha, beta = measure_alpha_beta(dist.group.WORLD, sizes)
```

### Using the Model for Prediction

```python
class CollectiveTimePredictor:
    """Predict collective operation times using alpha-beta model."""

    def __init__(self, alpha_seconds: float, beta_bytes_per_sec: float, world_size: int):
        self.alpha = alpha_seconds
        self.beta = beta_bytes_per_sec
        self.world_size = world_size

    def allreduce_ring(self, size_bytes: int) -> float:
        """Predict ring AllReduce time."""
        # Ring: 2(P-1) messages, each of size n/P
        # Total: 2(P-1)/P * n bytes sent
        effective_size = 2 * (self.world_size - 1) / self.world_size * size_bytes
        num_steps = 2 * (self.world_size - 1)
        return num_steps * self.alpha + effective_size / self.beta

    def allgather(self, size_bytes: int) -> float:
        """Predict AllGather time."""
        # Each rank gathers (P-1) chunks of size n
        effective_size = (self.world_size - 1) * size_bytes
        num_steps = self.world_size - 1
        return num_steps * self.alpha + effective_size / self.beta

    def reduce_scatter(self, size_bytes: int) -> float:
        """Predict ReduceScatter time."""
        # Similar to AllGather
        effective_size = (self.world_size - 1) / self.world_size * size_bytes
        num_steps = self.world_size - 1
        return num_steps * self.alpha + effective_size / self.beta

    def alltoall(self, size_bytes: int) -> float:
        """Predict AlltoAll time."""
        # Each rank sends (P-1) messages
        effective_size = (self.world_size - 1) / self.world_size * size_bytes
        num_steps = self.world_size - 1
        return num_steps * self.alpha + effective_size / self.beta

    def compare_algorithms(self, size_bytes: int):
        """Compare different algorithms for a given size."""
        ring = self.allreduce_ring(size_bytes)

        # Tree algorithm (better for small messages)
        tree_steps = 2 * math.ceil(math.log2(self.world_size))
        tree_time = tree_steps * self.alpha + size_bytes / self.beta * 2

        print(f"\n=== AllReduce Comparison for {size_bytes/1e6:.1f}MB ===")
        print(f"Ring: {ring*1e3:.2f}ms")
        print(f"Tree: {tree_time*1e3:.2f}ms")
        print(f"Recommended: {'Ring' if ring < tree_time else 'Tree'}")
```

## Profiling Different Parallelism Strategies

### Data Parallelism Profiling

Key metrics to track:

```python
class DDPProfiler:
    """Profiler specialized for DistributedDataParallel."""

    def __init__(self, model):
        self.model = model
        self.bucket_times = []
        self.hook_overheads = []
        self.gradient_sizes = []

        # Instrument buckets
        self._instrument_buckets()

    def _instrument_buckets(self):
        """Add timing instrumentation to DDP buckets."""
        # Access internal bucket info (PyTorch internals)
        if hasattr(self.model, '_module_copies'):
            # Extract bucket information
            pass

    def profile_step(self, batch):
        """Profile a single training step."""
        timings = {}

        # Forward pass
        torch.cuda.synchronize()
        forward_start = time.perf_counter()
        output = self.model(batch)
        torch.cuda.synchronize()
        timings['forward'] = time.perf_counter() - forward_start

        # Backward pass (triggers AllReduce)
        loss = output.sum()  # Dummy loss
        backward_start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        timings['backward_with_allreduce'] = time.perf_counter() - backward_start

        return timings

    def analyze_bucket_efficiency(self):
        """Analyze if bucket sizes are optimal."""
        if not self.bucket_times:
            print("No bucket timing data collected")
            return

        avg_bucket_time = sum(self.bucket_times) / len(self.bucket_times)

        print(f"\n=== Bucket Analysis ===")
        print(f"Number of buckets: {len(self.bucket_times)}")
        print(f"Average bucket AllReduce: {avg_bucket_time*1e3:.2f}ms")
        print(f"Total bucket AllReduce: {sum(self.bucket_times)*1e3:.2f}ms")
```

### Tensor Parallelism Profiling

```python
class TPProfiler:
    """Profiler for tensor parallelism."""

    def __init__(self, tp_degree: int):
        self.tp_degree = tp_degree
        self.allreduce_times = []
        self.allgather_times = []
        self.split_times = []

    def profile_layer(self, layer_fn, input_tensor):
        """Profile a tensor-parallel layer."""
        results = {}

        torch.cuda.synchronize()
        start = time.perf_counter()

        output = layer_fn(input_tensor)

        torch.cuda.synchronize()
        results['total_time'] = time.perf_counter() - start

        # Estimate communication fraction
        # For column-parallel linear: AllReduce after
        # For row-parallel linear: no communication

        return results

    def analyze_communication_fraction(self):
        """Analyze fraction of time spent in TP communication."""
        total_ar = sum(self.allreduce_times)
        total_ag = sum(self.allgather_times)

        print(f"\n=== Tensor Parallelism Communication ===")
        print(f"Total AllReduce time: {total_ar*1e3:.2f}ms ({len(self.allreduce_times)} calls)")
        print(f"Total AllGather time: {total_ag*1e3:.2f}ms ({len(self.allgather_times)} calls)")
```

### Pipeline Parallelism Profiling

```python
class PPProfiler:
    """Profiler for pipeline parallelism."""

    def __init__(self, num_stages: int, num_microbatches: int):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.stage_times = [[] for _ in range(num_stages)]
        self.bubble_time = 0

    def profile_schedule(self, schedule_fn):
        """Profile a pipeline schedule execution."""
        rank = dist.get_rank()
        stage = rank  # Assuming 1 stage per rank

        results = {
            'forward_times': [],
            'backward_times': [],
            'send_times': [],
            'recv_times': [],
            'idle_time': 0,
        }

        # Execute schedule with timing
        # ... implementation depends on pipeline framework

        return results

    def calculate_bubble_fraction(self):
        """Calculate pipeline bubble fraction."""
        # Bubble = (P - 1) * microbatch_time / total_time
        p = self.num_stages
        m = self.num_microbatches

        # For 1F1B schedule
        bubble_fraction = (p - 1) / (m + p - 1)

        print(f"\n=== Pipeline Bubble Analysis ===")
        print(f"Stages: {p}, Microbatches: {m}")
        print(f"Theoretical bubble fraction: {bubble_fraction:.1%}")
        print(f"Theoretical efficiency: {1 - bubble_fraction:.1%}")
```

### ZeRO Profiling

```python
class ZeROProfiler:
    """Profiler for ZeRO-style sharding."""

    def __init__(self, stage: int, world_size: int):
        self.stage = stage  # 1, 2, or 3
        self.world_size = world_size
        self.gather_times = []
        self.scatter_times = []
        self.optimizer_times = []

    def profile_forward(self, model, input_batch):
        """Profile forward pass with parameter gathering."""
        results = {'gather_time': 0, 'compute_time': 0}

        if self.stage == 3:
            # Track AllGather for each layer
            for name, param in model.named_parameters():
                gather_start = time.perf_counter()
                # AllGather param
                torch.cuda.synchronize()
                results['gather_time'] += time.perf_counter() - gather_start

        compute_start = time.perf_counter()
        output = model(input_batch)
        torch.cuda.synchronize()
        results['compute_time'] = time.perf_counter() - compute_start

        return results

    def analyze_memory_comm_tradeoff(self):
        """Analyze memory savings vs communication overhead."""
        print(f"\n=== ZeRO Stage {self.stage} Analysis ===")

        # Memory reduction factors
        memory_factors = {1: self.world_size, 2: self.world_size, 3: self.world_size}

        # Communication overhead factors (relative to data parallelism)
        comm_factors = {1: 1.0, 2: 1.0, 3: 1.5}  # Stage 3 adds AllGather

        print(f"Memory reduction: {memory_factors[self.stage]}x")
        print(f"Communication overhead: {comm_factors[self.stage]:.1f}x vs DP")
```

## Distributed Profiling Coordination

### Synchronized Profiling Across Ranks

```python
class DistributedProfiler:
    """Coordinate profiling across all ranks."""

    def __init__(self, output_dir: str):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.output_dir = Path(output_dir) / f"rank_{self.rank}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def profile_synchronized(self, train_fn, num_steps: int = 5):
        """
        Profile training with synchronized start across ranks.

        Ensures all ranks start profiling at the same time for
        aligned timelines.
        """
        # Barrier to synchronize start
        dist.barrier()

        # Start profiling
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.output_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for step in range(num_steps):
                with torch.cuda.nvtx.range(f"step_{step}"):
                    train_fn()
                prof.step()

        # Barrier to synchronize end
        dist.barrier()

        if self.rank == 0:
            print(f"Profiling complete. Traces saved to {self.output_dir.parent}")

    def aggregate_statistics(self, local_stats: dict) -> dict:
        """Aggregate statistics across all ranks."""
        # Gather all local stats to rank 0
        all_stats = [None for _ in range(self.world_size)]
        dist.all_gather_object(all_stats, local_stats)

        if self.rank == 0:
            # Aggregate
            aggregated = {}
            for key in local_stats.keys():
                values = [s[key] for s in all_stats if key in s]
                aggregated[key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std': (sum((v - sum(values)/len(values))**2 for v in values) / len(values)) ** 0.5
                }
            return aggregated
        return None
```

### Cross-Rank Timeline Alignment

```python
def align_timelines(traces_dir: Path):
    """
    Align profiler traces from multiple ranks using barrier timestamps.

    This helps identify relative timing across ranks.
    """
    import json

    traces = []
    for trace_file in traces_dir.glob("rank_*/trace.json"):
        with open(trace_file) as f:
            trace = json.load(f)
            rank = int(trace_file.parent.name.split('_')[1])
            traces.append((rank, trace))

    # Find barrier events in each trace
    barrier_times = {}
    for rank, trace in traces:
        for event in trace['traceEvents']:
            if 'name' in event and 'barrier' in event['name'].lower():
                if rank not in barrier_times:
                    barrier_times[rank] = []
                barrier_times[rank].append(event['ts'])

    # Calculate offsets relative to rank 0
    offsets = {0: 0}
    if 0 in barrier_times:
        ref_time = barrier_times[0][0]
        for rank in barrier_times:
            if rank != 0:
                offsets[rank] = barrier_times[rank][0] - ref_time

    print("Timeline offsets (μs):", offsets)
    return offsets
```

## Practical Profiling Workflow

### The Investigation Protocol

```
Step 1: Quick Baseline
├── Single step timing
├── GPU utilization (nvidia-smi)
└── Basic MFU estimate

Step 2: Identify Bottleneck Category
├── Compute vs Communication ratio
├── Memory pressure indicators
└── Straggler detection

Step 3: Detailed Analysis
├── Full Nsight trace (2-3 steps)
├── Memory breakdown
└── Collective timing breakdown

Step 4: Root Cause
├── Kernel-level analysis
├── Algorithm selection validation
└── Overlap efficiency measurement

Step 5: Validate Fix
├── A/B comparison
├── Confirm improvement
└── Check for regressions
```

### Quick Health Check

```python
def quick_health_check(model, dataloader, num_steps=3):
    """Fast profiling to identify obvious issues."""
    rank = dist.get_rank()

    timings = []
    for i, batch in enumerate(dataloader):
        if i >= num_steps:
            break

        torch.cuda.synchronize()
        start = time.perf_counter()

        # Training step
        with torch.cuda.nvtx.range(f"step_{i}"):
            output = model(batch)
            loss = output.sum()
            loss.backward()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    # Gather from all ranks
    all_timings = [None] * dist.get_world_size()
    dist.all_gather_object(all_timings, timings)

    if rank == 0:
        print("\n=== Quick Health Check ===")
        for r, times in enumerate(all_timings):
            avg = sum(times) / len(times)
            print(f"Rank {r}: avg={avg*1e3:.1f}ms, times={[f'{t*1e3:.1f}' for t in times]}")

        # Check for stragglers
        avg_times = [sum(t)/len(t) for t in all_timings]
        mean = sum(avg_times) / len(avg_times)
        max_time = max(avg_times)

        if max_time > mean * 1.1:
            slowest = avg_times.index(max_time)
            print(f"\n⚠️  Straggler detected: Rank {slowest} is {max_time/mean:.0%} of mean")
        else:
            print("\n✓ No obvious stragglers")
```

### Comparative Profiling

```python
class ABProfiler:
    """Compare two configurations."""

    def __init__(self):
        self.results_a = []
        self.results_b = []

    def profile_config(self, name: str, setup_fn, train_fn, num_steps: int = 10):
        """Profile a configuration."""
        setup_fn()  # Apply configuration

        timings = []
        for _ in range(num_steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            train_fn()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)

        if name == 'A':
            self.results_a = timings
        else:
            self.results_b = timings

    def compare(self):
        """Compare A and B configurations."""
        if not self.results_a or not self.results_b:
            print("Need both A and B results")
            return

        avg_a = sum(self.results_a) / len(self.results_a)
        avg_b = sum(self.results_b) / len(self.results_b)

        print("\n=== A/B Comparison ===")
        print(f"Config A: {avg_a*1e3:.2f}ms avg")
        print(f"Config B: {avg_b*1e3:.2f}ms avg")
        print(f"Difference: {(avg_b - avg_a)/avg_a*100:+.1f}%")

        # Statistical significance (simple t-test)
        if len(self.results_a) >= 5 and len(self.results_b) >= 5:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(self.results_a, self.results_b)
            print(f"p-value: {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")
```

## Common Issues and Diagnostics

### Issue: Low MFU Despite Powerful Hardware

```python
def diagnose_low_mfu():
    """Diagnostic checklist for low MFU."""
    checks = [
        ("Mixed precision enabled?", "torch.cuda.amp.autocast"),
        ("Tensor Cores used?", "Check for TF32/FP16 kernels in trace"),
        ("Batch size sufficient?", "Small batches → kernel launch overhead dominates"),
        ("Memory bandwidth limited?", "Check memory throughput in Nsight"),
        ("Python overhead?", "Check CPU utilization, GIL contention"),
        ("Data loading bottleneck?", "Profile DataLoader separately"),
    ]

    print("\n=== Low MFU Diagnostic Checklist ===")
    for check, how in checks:
        print(f"[ ] {check}")
        print(f"    → {how}")
```

### Issue: Communication Taking Too Long

```python
def diagnose_slow_communication():
    """Diagnostic for communication bottlenecks."""
    checks = [
        "Measure actual vs theoretical bandwidth",
        "Check for network congestion (multiple jobs)",
        "Verify NCCL algorithm selection (NCCL_DEBUG=INFO)",
        "Check for imbalanced work (stragglers)",
        "Verify overlap is working (Nsight timeline)",
        "Check bucket sizes for DDP",
    ]

    print("\n=== Slow Communication Diagnostic ===")
    for i, check in enumerate(checks, 1):
        print(f"{i}. {check}")
```

### Issue: Memory Pressure

```python
def diagnose_memory_pressure():
    """Diagnostic for memory issues."""

    # Check current state
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_alloc = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n=== Memory Diagnostic ===")
    print(f"Currently allocated: {allocated:.2f}GB")
    print(f"Reserved by PyTorch: {reserved:.2f}GB")
    print(f"Peak allocation: {max_alloc:.2f}GB")
    print(f"Fragmentation: {1 - allocated/reserved:.1%}")

    if max_alloc > reserved * 0.95:
        print("\n⚠️  Near memory limit - consider:")
        print("   - Gradient checkpointing")
        print("   - Smaller batch size")
        print("   - ZeRO stage 2 or 3")
        print("   - Offloading")
```

## Exercises

1. **MFU Measurement**: Implement a complete MFU measurement for your model. Compare theoretical FLOPS (from model architecture) with achieved FLOPS (from step time). What efficiency do you achieve?

2. **Alpha-Beta Calibration**: Measure α and β for your cluster. Use multiple message sizes (1KB to 1GB). Plot predicted vs actual times. How accurate is the linear model?

3. **Overlap Efficiency**: Profile your DDP training with NVTX annotations. Calculate overlap efficiency. What percentage of communication is hidden behind computation?

4. **Straggler Detection**: Run training on 8 GPUs. Artificially slow one GPU (insert sleep). Measure the impact on overall throughput. Implement automatic straggler detection.

5. **Bucket Optimization**: Profile DDP with different bucket sizes (1MB, 25MB, 100MB, 250MB). Measure step time for each. What's the optimal bucket size for your model?

6. **Memory Profiling**: Track memory usage through a training step. Identify the peak allocation point. What consumes the most memory: parameters, gradients, activations, or optimizer state?

## Key Takeaways

1. **Measure, don't guess**: Profiling transforms intuition into actionable data.

2. **Multiple tools for multiple purposes**: Nsight for detailed traces, PyTorch Profiler for quick checks, NCCL debug for communication analysis.

3. **MFU is the ultimate metric**: It captures how well you're using your hardware.

4. **The alpha-beta model predicts communication**: Calibrate it for your cluster to understand bottlenecks.

5. **Overlap efficiency matters**: The gap between compute+comm and max(compute, comm) is your opportunity.

6. **Stragglers kill scaling**: One slow GPU affects all GPUs in synchronous training.

7. **Profile iteratively**: Start broad, then zoom in on bottlenecks.

