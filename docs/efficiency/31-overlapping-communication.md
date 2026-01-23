---
title: "Overlapping Communication with Computation"
subtitle: "Hiding Latency Through Concurrent Execution"
---

::: {.chapter-opener}
The fastest communication is communication you don't wait for. By overlapping data transfers with computation, we can approach the theoretical limit where communication cost approaches zero.
:::

::: {.investigation-question}
**The Question**: A training step has 100ms of compute and 50ms of communication. The naive approach takes 150ms. Perfect overlap would take 100ms. What determines where you land between these extremes?
:::

## The Overlap Opportunity

Consider a typical training iteration:

```
Naive execution (sequential):

[Forward Pass] → [Backward Pass] → [AllReduce] → [Update]
     40ms            50ms             50ms         10ms
                                                        Total: 150ms
```

But computation and communication use different resources:
- **Compute**: GPU SMs (Streaming Multiprocessors)
- **Communication**: NVLink/InfiniBand + network interface

These can execute concurrently:

```
Overlapped execution:

[Forward Pass] → [Backward Pass + AllReduce overlap] → [Update]
     40ms                      60ms                       10ms
                                                              Total: 110ms
```

The key insight: **gradients for early layers are computed late in backpropagation**. We can communicate gradients for later layers while still computing gradients for earlier layers.

### Theoretical Speedup

Define:
- $T_c$: Compute time (forward + backward)
- $T_m$: Communication time (AllReduce)
- $\alpha$: Overlap fraction (0 = no overlap, 1 = perfect overlap)

Total time with overlap:

$$T_{\text{total}} = T_c + (1 - \alpha) \cdot T_m$$

**Speedup**:

$$\text{Speedup} = \frac{T_c + T_m}{T_c + (1 - \alpha) \cdot T_m} = \frac{1 + T_m/T_c}{1 + (1-\alpha) \cdot T_m/T_c}$$

For perfect overlap ($\alpha = 1$): Speedup $= 1 + T_m/T_c$

| $T_m/T_c$ | No Overlap | 50% Overlap | Perfect Overlap |
|-----------|------------|-------------|-----------------|
| 0.25 | 1.0× | 1.11× | 1.25× |
| 0.50 | 1.0× | 1.20× | 1.50× |
| 1.00 | 1.0× | 1.33× | 2.00× |
| 2.00 | 1.0× | 1.50× | 3.00× |

When communication dominates ($T_m \gg T_c$), overlap becomes critical.

## Backward Pass Structure

Understanding backward pass structure is key to overlap.

### Layer-by-Layer Backpropagation

```python
def backward_pass(model, loss):
    """Standard backward pass - layer by layer."""
    gradients = {}

    # Start from loss, work backward through layers
    grad_output = loss.backward()  # dL/d(output)

    # Layer N (last layer)
    grad_output, grad_params_N = layer_N.backward(grad_output)
    gradients['layer_N'] = grad_params_N
    # At this point, layer_N gradients are COMPLETE and can be communicated

    # Layer N-1
    grad_output, grad_params_N1 = layer_N1.backward(grad_output)
    gradients['layer_N-1'] = grad_params_N1
    # layer_N-1 gradients now complete

    # ... continue to layer 0

    return gradients
```

**Key observation**: Layer $k$'s gradients are complete before layer $k-1$'s computation starts.

### The Overlap Window

```
Layer computation during backward:

Layer 6: [=====]                              ← Gradients ready first
Layer 5:       [=====]
Layer 4:             [=====]
Layer 3:                   [=====]
Layer 2:                         [=====]
Layer 1:                               [=====]
Layer 0:                                     [=====]  ← Gradients ready last

Communication can start here:
         ↓
         [Comm L6][Comm L5][Comm L4][Comm L3][Comm L2][Comm L1][Comm L0]
```

The overlap window equals the backward compute time minus one layer.

## Gradient Bucketing

### Why Bucketing?

Small messages have high overhead:
- Per-message latency $\alpha$ dominates for small transfers
- Network underutilization
- NCCL kernel launch overhead

**Solution**: Aggregate gradients into buckets before communicating.

### Bucket Formation

```python
class GradientBucketer:
    """Accumulate gradients into fixed-size buckets."""

    def __init__(self, bucket_size_mb: float = 25.0):
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.buckets = []
        self.current_bucket = []
        self.current_size = 0

    def add_gradient(self, param_name: str, gradient: torch.Tensor):
        """Add a gradient to the current bucket."""
        grad_size = gradient.numel() * gradient.element_size()

        if self.current_size + grad_size > self.bucket_size_bytes:
            # Bucket full, finalize and start new bucket
            if self.current_bucket:
                self.buckets.append(self._finalize_bucket())
            self.current_bucket = []
            self.current_size = 0

        self.current_bucket.append((param_name, gradient))
        self.current_size += grad_size

    def _finalize_bucket(self) -> torch.Tensor:
        """Flatten bucket into contiguous tensor."""
        flat_grads = []
        for _, grad in self.current_bucket:
            flat_grads.append(grad.view(-1))
        return torch.cat(flat_grads)

    def flush(self):
        """Finalize any remaining gradients."""
        if self.current_bucket:
            self.buckets.append(self._finalize_bucket())
            self.current_bucket = []
            self.current_size = 0
```

### Optimal Bucket Size

Bucket size affects overlap quality:

**Too small**:
- High per-bucket latency overhead
- Many small AllReduce operations
- Communication cannot keep up with computation

**Too large**:
- Delayed start of communication
- Less overlap opportunity
- Memory pressure from buffering

**Finding optimal size**:

$$B^* = \underset{B}{\text{argmax}} \left[ \alpha(B) - \frac{\alpha_{\text{comm}}}{B} \right]$$

Where:
- $\alpha(B)$: Overlap fraction with bucket size $B$
- $\alpha_{\text{comm}}$: Per-bucket latency overhead

Empirically, 25-50 MB buckets work well for most configurations.

### PyTorch DDP Bucketing

```python
# PyTorch DDP bucket configuration
model = DistributedDataParallel(
    model,
    device_ids=[local_rank],
    bucket_cap_mb=25,  # Bucket size in MB
    gradient_as_bucket_view=True,  # Avoid extra memory copies
    find_unused_parameters=False,  # Faster if no unused params
)
```

## CUDA Streams for Overlap

### Stream Basics

CUDA streams enable concurrent execution:

```python
# Create separate streams
compute_stream = torch.cuda.Stream()
comm_stream = torch.cuda.Stream()

# Operations on different streams can overlap
with torch.cuda.stream(compute_stream):
    # Compute operations
    y = torch.matmul(A, B)

with torch.cuda.stream(comm_stream):
    # Communication operations
    dist.all_reduce(tensor, async_op=True)
```

### Synchronization with Events

```python
class StreamOverlap:
    """Manage compute-communication overlap with CUDA streams."""

    def __init__(self):
        self.compute_stream = torch.cuda.Stream()
        self.comm_stream = torch.cuda.Stream()
        self.comm_handles = []

    def backward_with_overlap(self, model, loss):
        """Backward pass with overlapped communication."""
        # Backward happens on compute stream
        with torch.cuda.stream(self.compute_stream):
            loss.backward()

        # As gradients become ready, queue communication
        for bucket in self.get_ready_buckets():
            # Record event when bucket's compute is done
            event = torch.cuda.Event()
            event.record(self.compute_stream)

            # Comm stream waits for compute
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(event)
                handle = dist.all_reduce(bucket, async_op=True)
                self.comm_handles.append(handle)

    def synchronize(self):
        """Wait for all communication to complete."""
        for handle in self.comm_handles:
            handle.wait()
        self.comm_stream.synchronize()
        self.comm_handles.clear()
```

### Stream Prioritization

```python
# High priority stream for communication
# (ensures comm doesn't starve)
comm_stream = torch.cuda.Stream(priority=-1)  # Higher priority

# Normal priority for compute
compute_stream = torch.cuda.Stream(priority=0)
```

## Backward Hook Mechanism

PyTorch uses hooks to trigger communication at the right time.

### Gradient Hooks

```python
class OverlappedAllReduce:
    """Use hooks to trigger AllReduce as gradients become ready."""

    def __init__(self, model, process_group):
        self.model = model
        self.process_group = process_group
        self.handles = []
        self.comm_stream = torch.cuda.Stream()

        # Register backward hooks
        for param in model.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._make_hook(param)
                )

    def _make_hook(self, param):
        def hook(grad):
            # Record when gradient is ready
            event = torch.cuda.Event()
            event.record()

            # Launch AllReduce on comm stream
            with torch.cuda.stream(self.comm_stream):
                self.comm_stream.wait_event(event)
                handle = dist.all_reduce(
                    param.grad,
                    group=self.process_group,
                    async_op=True
                )
                self.handles.append(handle)

            return grad
        return hook

    def finish_step(self):
        """Wait for all AllReduce operations."""
        self.comm_stream.synchronize()
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
```

### Bucket-Aware Hooks

```python
class BucketedOverlapAllReduce:
    """Bucket gradients before triggering AllReduce."""

    def __init__(self, model, process_group, bucket_size_mb=25):
        self.process_group = process_group
        self.bucket_size = int(bucket_size_mb * 1024 * 1024)

        # Group parameters into buckets (reverse order for backward)
        self.buckets = self._create_buckets(model)
        self.pending_grads = {}
        self.comm_stream = torch.cuda.Stream()
        self.handles = []

        # Register hooks
        for bucket_id, (params, _) in enumerate(self.buckets):
            for param in params:
                param.register_post_accumulate_grad_hook(
                    self._make_hook(param, bucket_id)
                )

    def _create_buckets(self, model):
        """Create buckets in reverse parameter order."""
        buckets = []
        current_params = []
        current_size = 0

        # Reverse order: last params first (computed first in backward)
        for param in reversed(list(model.parameters())):
            if not param.requires_grad:
                continue

            param_size = param.numel() * param.element_size()

            if current_size + param_size > self.bucket_size:
                if current_params:
                    flat_buffer = self._allocate_flat_buffer(current_params)
                    buckets.append((current_params.copy(), flat_buffer))
                current_params = []
                current_size = 0

            current_params.append(param)
            current_size += param_size

        if current_params:
            flat_buffer = self._allocate_flat_buffer(current_params)
            buckets.append((current_params, flat_buffer))

        return buckets

    def _make_hook(self, param, bucket_id):
        def hook(grad):
            self.pending_grads[param] = True
            self._check_bucket_ready(bucket_id)
            return grad
        return hook

    def _check_bucket_ready(self, bucket_id):
        """Launch AllReduce if all gradients in bucket are ready."""
        params, flat_buffer = self.buckets[bucket_id]

        if all(p in self.pending_grads for p in params):
            # Copy gradients to flat buffer
            offset = 0
            for param in params:
                numel = param.numel()
                flat_buffer[offset:offset+numel].copy_(param.grad.view(-1))
                offset += numel

            # Launch AllReduce
            with torch.cuda.stream(self.comm_stream):
                handle = dist.all_reduce(
                    flat_buffer,
                    group=self.process_group,
                    async_op=True
                )
                self.handles.append((handle, params, flat_buffer))
```

## Overlap Analysis

### Measuring Overlap Efficiency

```python
class OverlapProfiler:
    """Profile compute-communication overlap."""

    def __init__(self):
        self.compute_events = []
        self.comm_events = []

    def profile_step(self, model, data, target, criterion):
        """Profile one training step."""
        # Mark compute regions
        compute_start = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)
        comm_start = torch.cuda.Event(enable_timing=True)
        comm_end = torch.cuda.Event(enable_timing=True)

        # Forward pass
        compute_start.record()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        compute_end.record()

        # Communication (if using custom overlap, track separately)
        comm_start.record()
        # ... AllReduce ...
        comm_end.record()

        torch.cuda.synchronize()

        compute_time = compute_start.elapsed_time(compute_end)
        comm_time = comm_start.elapsed_time(comm_end)
        total_time = compute_start.elapsed_time(comm_end)

        # Calculate overlap
        sequential_time = compute_time + comm_time
        overlap_time = sequential_time - total_time
        overlap_fraction = overlap_time / comm_time if comm_time > 0 else 0

        return {
            'compute_ms': compute_time,
            'comm_ms': comm_time,
            'total_ms': total_time,
            'overlap_fraction': overlap_fraction,
            'speedup': sequential_time / total_time if total_time > 0 else 1
        }
```

### Overlap Visualization

```
Timeline visualization:

Time →  0ms    20ms    40ms    60ms    80ms   100ms
        |       |       |       |       |       |
Compute:[============================]
                  ↑ gradient ready
                  |
Comm:             [================]
                  ← overlap →

Overlap region: 60ms - 40ms = 20ms
Overlap fraction: 20ms / 40ms = 50%
```

### Identifying Overlap Bottlenecks

```python
def analyze_overlap_bottleneck(model, bucket_size_mb=25):
    """Identify what limits overlap."""
    params = list(model.parameters())

    # Time to compute all gradients
    total_param_bytes = sum(p.numel() * p.element_size() for p in params)

    # Number of buckets
    bucket_bytes = bucket_size_mb * 1024 * 1024
    num_buckets = (total_param_bytes + bucket_bytes - 1) // bucket_bytes

    # Compute time per bucket (assume uniform)
    compute_time_per_bucket = total_backward_time / num_buckets

    # Communication time per bucket
    # Using α-β model: T = α + n/β
    comm_time_per_bucket = alpha + bucket_bytes / bandwidth

    # Overlap is limited by slower of compute or comm per bucket
    if compute_time_per_bucket > comm_time_per_bucket:
        bottleneck = "compute"
        # Comm can keep up, limited by when gradients are ready
    else:
        bottleneck = "communication"
        # Comm slower than gradient production, will queue up

    return {
        'bottleneck': bottleneck,
        'compute_per_bucket': compute_time_per_bucket,
        'comm_per_bucket': comm_time_per_bucket,
        'num_buckets': num_buckets
    }
```

## Double Buffering

For continuous overlap across iterations, use double buffering.

### Weight Update During Communication

```python
class DoubleBufferedOptimizer:
    """
    Use two sets of weights to overlap update with next iteration.
    """

    def __init__(self, model, base_optimizer):
        self.model = model
        self.base_optimizer = base_optimizer

        # Two sets of weights
        self.weights_a = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        self.weights_b = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }
        self.current = 'a'

        # Streams
        self.compute_stream = torch.cuda.Stream()
        self.update_stream = torch.cuda.Stream()

    def step_async(self, gradients):
        """
        Apply gradients to inactive buffer while compute uses active buffer.
        """
        inactive = self.weights_b if self.current == 'a' else self.weights_a

        with torch.cuda.stream(self.update_stream):
            for name, grad in gradients.items():
                # Apply update to inactive weights
                inactive[name].add_(grad, alpha=-self.lr)

    def swap_buffers(self):
        """Swap active and inactive buffers."""
        # Wait for update to complete
        self.update_stream.synchronize()

        # Swap
        self.current = 'b' if self.current == 'a' else 'a'

        # Copy active weights to model
        active = self.weights_a if self.current == 'a' else self.weights_b
        for name, param in self.model.named_parameters():
            param.data.copy_(active[name])
```

### Pipeline Double Buffering

```python
class PipelinedDataLoader:
    """
    Prefetch next batch while current batch is processing.
    """

    def __init__(self, dataloader):
        self.dataloader = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        self._prefetch()

    def _prefetch(self):
        """Load and transfer next batch asynchronously."""
        try:
            batch = next(self.dataloader)
            with torch.cuda.stream(self.stream):
                self.next_batch = tuple(
                    t.cuda(non_blocking=True) if isinstance(t, torch.Tensor)
                    else t for t in batch
                )
        except StopIteration:
            self.next_batch = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration

        # Wait for prefetch to complete
        torch.cuda.current_stream().wait_stream(self.stream)

        batch = self.next_batch
        self._prefetch()  # Start next prefetch
        return batch
```

## ZeRO and Overlap

ZeRO optimizations require careful overlap strategies.

### ZeRO Stage 1: Optimizer State Overlap

```python
class ZeRO1WithOverlap:
    """
    ZeRO-1 with overlapped gradient reduction and optimizer step.
    """

    def __init__(self, model, optimizer, process_group):
        self.model = model
        self.optimizer = optimizer
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)

        # Partition optimizer states
        self.param_to_partition = self._partition_params()

        # Streams
        self.reduce_stream = torch.cuda.Stream()
        self.update_stream = torch.cuda.Stream()

    def backward_and_step(self, loss):
        """Overlapped backward, reduce-scatter, and optimizer step."""
        handles = []

        # Backward pass
        loss.backward()

        # For each bucket of gradients
        for bucket_params in self.buckets:
            # Flatten gradients
            flat_grad = self._flatten_grads(bucket_params)

            # ReduceScatter on reduce stream
            with torch.cuda.stream(self.reduce_stream):
                output = torch.zeros(
                    flat_grad.size(0) // self.world_size,
                    device=flat_grad.device
                )
                handle = dist.reduce_scatter(
                    output, [flat_grad.chunk(self.world_size)],
                    group=self.process_group,
                    async_op=True
                )
                handles.append((handle, bucket_params, output))

        # As reduce-scatters complete, apply optimizer on update stream
        for handle, bucket_params, reduced_grad in handles:
            handle.wait()

            with torch.cuda.stream(self.update_stream):
                self._apply_optimizer_step(bucket_params, reduced_grad)
```

### ZeRO Stage 3: Prefetching Parameters

```python
class ZeRO3PrefetchScheduler:
    """
    Schedule parameter gathering to overlap with compute.
    """

    def __init__(self, model, process_group, prefetch_count=2):
        self.model = model
        self.process_group = process_group
        self.prefetch_count = prefetch_count

        # Parameter sharding info
        self.param_shards = self._shard_parameters()

        # Prefetch streams
        self.prefetch_streams = [
            torch.cuda.Stream() for _ in range(prefetch_count)
        ]
        self.prefetch_buffers = {}

    def forward_with_prefetch(self, input_tensor):
        """Forward pass with parameter prefetching."""
        layers = list(self.model.children())

        # Start prefetching first layers
        for i in range(min(self.prefetch_count, len(layers))):
            self._start_prefetch(layers[i], stream_idx=i)

        # Execute layers with rolling prefetch
        x = input_tensor
        for i, layer in enumerate(layers):
            # Wait for current layer's parameters
            self._wait_prefetch(layer)

            # Execute layer
            x = layer(x)

            # Release parameters (for memory efficiency)
            self._release_params(layer)

            # Start prefetching future layer
            next_prefetch = i + self.prefetch_count
            if next_prefetch < len(layers):
                stream_idx = next_prefetch % self.prefetch_count
                self._start_prefetch(layers[next_prefetch], stream_idx)

        return x

    def _start_prefetch(self, layer, stream_idx):
        """Begin AllGather for layer's parameters."""
        stream = self.prefetch_streams[stream_idx]

        with torch.cuda.stream(stream):
            for param in layer.parameters():
                shard = self.param_shards[param]
                full_param = torch.empty(
                    param.numel() * self.world_size,
                    device=param.device
                )
                dist.all_gather_into_tensor(
                    full_param, shard, group=self.process_group
                )
                self.prefetch_buffers[param] = full_param

    def _wait_prefetch(self, layer):
        """Wait for layer's parameters to be gathered."""
        for param in layer.parameters():
            full_param = self.prefetch_buffers[param]
            param.data = full_param.view(param.shape)
```

## Communication-Computation Balance

### When Overlap Helps Most

Overlap is most beneficial when:

1. **Balanced compute/comm ratio**: Neither should dominate heavily
2. **Many small operations**: More opportunities to interleave
3. **High bandwidth networks**: Comm can keep up with compute

### The Overlap Limit

Even with perfect overlap, you're limited by:

$$T_{\text{min}} = \max(T_c, T_m)$$

If communication is slower than compute ($T_m > T_c$), you'll eventually queue up:

```
Compute-bound (Tc > Tm): Perfect overlap possible
Time:   [=========Compute==========]
             [===Comm===]
                    Total = Tc ✓

Communication-bound (Tm > Tc): Overlap limited
Time:   [===Compute===]
        [========Comm========]
                    Total = Tm ✗
```

### Rebalancing Strategies

When communication-bound:

```python
def rebalance_for_overlap(model_config, comm_config):
    """Adjust configuration to improve overlap."""
    tc = estimate_compute_time(model_config)
    tm = estimate_comm_time(model_config, comm_config)

    if tm > tc:
        # Communication bound - reduce communication volume
        options = [
            ("gradient_compression", "Reduce gradient bits"),
            ("increase_batch", "Larger batches, fewer steps"),
            ("tensor_parallelism", "Split model, reduce AllReduce size"),
        ]
    else:
        # Compute bound - communication easily hidden
        options = [
            ("smaller_buckets", "More granular overlap"),
            ("reduce_prefetch", "Save memory"),
        ]

    return options
```

## NCCL Overlap Patterns

### Group Operations

```python
def overlapped_allreduce_with_compute(tensors, compute_fn):
    """
    Overlap multiple AllReduce operations with compute.
    """
    # Start all AllReduce operations
    handles = []
    for tensor in tensors:
        handle = dist.all_reduce(tensor, async_op=True)
        handles.append(handle)

    # Do compute while communication proceeds
    result = compute_fn()

    # Wait for all communication
    for handle in handles:
        handle.wait()

    return result, tensors
```

### NCCL Groups for Batching

```python
def batched_collectives():
    """Batch multiple collectives for efficiency."""
    with dist.batch_isend_irecv([
        dist.P2POp(dist.isend, send_tensor, dst_rank),
        dist.P2POp(dist.irecv, recv_tensor, src_rank),
    ]) as handles:
        # All operations launched together
        pass

    # Wait for completion
    for handle in handles:
        handle.wait()
```

## Overlap in Pipeline Parallelism

Pipeline parallelism naturally creates overlap opportunities.

### Interleaved 1F1B Schedule

```python
class InterleavedPipelineOverlap:
    """
    Pipeline schedule that overlaps send/recv with compute.
    """

    def __init__(self, num_stages, num_microbatches):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches

        # Separate streams for compute and communication
        self.compute_stream = torch.cuda.Stream()
        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()

    def execute_stage(self, stage_fn, input_tensor, is_forward):
        """Execute one pipeline stage with overlapped communication."""
        # Receive input (if not first stage)
        if self.stage_id > 0:
            with torch.cuda.stream(self.recv_stream):
                recv_handle = self._recv_activation(input_tensor)
            torch.cuda.current_stream().wait_stream(self.recv_stream)

        # Compute
        with torch.cuda.stream(self.compute_stream):
            output = stage_fn(input_tensor)

        # Send output (if not last stage) - overlapped with next recv
        if self.stage_id < self.num_stages - 1:
            # Wait for compute
            self.send_stream.wait_stream(self.compute_stream)
            with torch.cuda.stream(self.send_stream):
                self._send_activation(output)

        return output
```

## Practical Implementation

### PyTorch DDP Configuration

```python
def configure_ddp_for_overlap(model, local_rank):
    """Configure DDP for optimal overlap."""
    return DistributedDataParallel(
        model.cuda(local_rank),
        device_ids=[local_rank],
        output_device=local_rank,

        # Bucketing
        bucket_cap_mb=25,

        # Avoid extra copies
        gradient_as_bucket_view=True,

        # Static graph for optimization (if model is fixed)
        static_graph=True,

        # Don't find unused (faster hooks)
        find_unused_parameters=False,
    )
```

### DeepSpeed Overlap Settings

```python
deepspeed_config = {
    "train_batch_size": 1024,
    "gradient_accumulation_steps": 4,

    "zero_optimization": {
        "stage": 2,

        # Overlap settings
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e7,  # 50MB

        # Prefetch
        "prefetch_bucket_size": 5e7,
    },

    "communication_data_type": "fp16",
}
```

### FSDP Overlap Configuration

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    BackwardPrefetch,
    ShardingStrategy,
)

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,

    # Prefetch during backward
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,

    # Forward prefetch
    forward_prefetch=True,

    # Limit concurrent AllGather
    limit_all_gathers=True,
)
```

## Overlap Debugging

### Common Issues

**1. No overlap observed**

```python
# Check if operations are on same stream (bad)
print(f"Compute stream: {torch.cuda.current_stream()}")
print(f"Comm stream: {dist.get_default_stream()}")

# Should be different for overlap
```

**2. Deadlock in overlap**

```python
# Ensure proper synchronization order
event = torch.cuda.Event()
event.record(compute_stream)
comm_stream.wait_event(event)  # Comm waits for compute
# Don't do: compute_stream.wait_stream(comm_stream) here
```

**3. Memory explosion with overlap**

```python
# Too many in-flight operations
# Limit concurrent buckets
max_concurrent_buckets = 2
if len(active_handles) >= max_concurrent_buckets:
    active_handles[0].wait()
    active_handles.pop(0)
```

### Profiling Overlap

```python
# Use PyTorch profiler with CUDA events
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
) as prof:
    train_step()

# Look for overlapping CUDA kernels in trace
prof.export_chrome_trace("overlap_trace.json")
```

## Exercises

1. **Overlap calculation**: A model has 10 layers, each taking 5ms for backward. Communication of each layer's gradients takes 3ms. With bucketing (2 layers per bucket), what's the maximum theoretical overlap? What's the total time?

2. **Bucket size optimization**: You have 1GB of gradients and 100 Gbps bandwidth. Communication latency is 100μs. Find the bucket size that minimizes per-bucket communication time. What's the optimal number of buckets?

3. **Stream scheduling**: Implement a simple two-stream scheduler that runs backward pass on one stream while performing AllReduce on another. Measure the overlap fraction achieved.

4. **Prefetch depth**: For ZeRO-3 with 24 layers, how many layers should you prefetch to hide all-gather latency if each all-gather takes 2ms and each layer compute takes 8ms?

5. **Communication bound analysis**: Your training step shows 80ms compute, 120ms communication, but total time is 140ms. Calculate the overlap fraction. What techniques could improve this?

6. **Double buffering**: Implement double-buffered gradient AllReduce. Measure memory overhead vs. overlap improvement.

## Key Takeaways

1. **Overlap hides latency**: Communication during compute approaches zero visible cost.

2. **Bucketing enables overlap**: Aggregate gradients to amortize latency and enable streaming.

3. **CUDA streams are fundamental**: Separate streams for compute and communication enable concurrency.

4. **Hooks trigger communication**: PyTorch hooks launch AllReduce as gradients become ready.

5. **Balance matters**: Overlap is limited by the slower of compute or communication per bucket.

6. **ZeRO needs prefetch**: Weight gathering must be scheduled ahead of compute.

7. **Profile to verify**: Assumed overlap often differs from actual—measure with profiler.

8. **Memory is the cost**: Overlap requires buffering, trading memory for latency.
