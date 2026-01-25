---
title: "Failures and Checkpointing"
subtitle: "Fault Tolerance as a Mathematical Invariant"
---

<div class="chapter-opener" markdown>
At scale, failures are not exceptions—they are expectations. A 10,000 GPU cluster with 4-hour MTBF per GPU will lose one GPU every 1.4 seconds on average. The mathematics of checkpointing determines whether training completes or collapses.
</div>

<div class="investigation-question" markdown>
**The Question**: How do you checkpoint a 1TB model distributed across 8192 GPUs such that any failure loses at most 5 minutes of work, while checkpoint overhead consumes less than 1% of training time? What mathematical invariants must the checkpoint maintain?
</div>

## The Scale of the Problem

### Failure Statistics at Scale

Individual component Mean Time Between Failures (MTBF):

| Component | MTBF (hours) | Failure Rate λ (per hour) |
|-----------|--------------|---------------------------|
| GPU | 10,000 - 50,000 | 0.0001 - 0.00002 |
| HBM | 100,000+ | 0.00001 |
| NVLink | 50,000+ | 0.00002 |
| Network Switch | 100,000+ | 0.00001 |
| Power Supply | 50,000+ | 0.00002 |
| Host Machine | 5,000 - 20,000 | 0.0002 - 0.00005 |

For a system with $N$ components each with MTBF $M$:

$$\text{MTBF}_{\text{system}} = \frac{M}{N}$$

### The 10,000 GPU Calculation

Assume 10,000 GPUs, 1,250 hosts (8 GPUs each), effective GPU MTBF of 20,000 hours, host MTBF of 10,000 hours:

$$\lambda_{\text{GPU}} = \frac{10,000}{20,000} = 0.5 \text{ failures/hour}$$

$$\lambda_{\text{host}} = \frac{1,250}{10,000} = 0.125 \text{ failures/hour}$$

$$\lambda_{\text{total}} \approx 0.625 \text{ failures/hour}$$

$$\text{MTBF}_{\text{system}} = \frac{1}{0.625} = 1.6 \text{ hours}$$

**Every 96 minutes on average, something fails.**

### The Training Time Equation

Without checkpointing, expected completion time for a job requiring $T$ hours:

$$\mathbb{E}[\text{completion time}] = \frac{e^{\lambda T} - 1}{\lambda}$$

For $T = 720$ hours (30 days) and $\lambda = 0.625$:

$$\mathbb{E}[\text{completion time}] = \frac{e^{450} - 1}{0.625} \approx \infty$$

The job will **never complete** without checkpointing.

## Checkpointing Fundamentals

### The Checkpoint Invariant

A valid checkpoint must satisfy the **consistency invariant**:

<div class="definition" markdown>
**Checkpoint Consistency**: A checkpoint $C$ is consistent if and only if resuming training from $C$ produces the same sequence of model states as uninterrupted training would have produced from the point $C$ was taken.
</div>

For distributed training, this requires:

1. **Model state consistency**: All parameter shards represent the same logical step
2. **Optimizer state consistency**: Moments, velocities align with parameters
3. **Data loader state**: Resume from correct position in epoch
4. **RNG state**: Random number generators reproducible

### Checkpoint Contents

A complete checkpoint contains:

```python
@dataclass
class DistributedCheckpoint:
    """Complete state for resumable distributed training."""

    # Training progress
    global_step: int
    tokens_seen: int
    epoch: int

    # Model state (sharded)
    model_state: Dict[str, ShardedTensor]

    # Optimizer state (sharded)
    optimizer_state: Dict[str, Dict[str, ShardedTensor]]

    # Learning rate scheduler
    scheduler_state: Dict[str, Any]

    # Data loading
    dataloader_state: DataLoaderState

    # Random number generators
    rng_states: RNGStates

    # Configuration for validation
    config: TrainingConfig

    # Metadata
    timestamp: float
    checkpoint_version: str
```

### Sharded Tensors

With model parallelism, each rank holds partial state:

```python
@dataclass
class ShardedTensor:
    """Tensor distributed across multiple ranks."""

    # Local shard data
    local_tensor: torch.Tensor

    # Global tensor metadata
    global_shape: Tuple[int, ...]
    global_dtype: torch.dtype

    # Sharding specification
    sharding_spec: ShardingSpec

    # Placement
    rank: int
    world_size: int

    def global_offset(self) -> Tuple[int, ...]:
        """Compute where this shard fits in global tensor."""
        return self.sharding_spec.offset_for_rank(self.rank)

@dataclass
class ShardingSpec:
    """How a tensor is sharded across ranks."""

    dim: int                    # Dimension sharded along
    num_shards: int             # Number of shards
    shard_sizes: List[int]      # Size of each shard

    def offset_for_rank(self, rank: int) -> Tuple[int, ...]:
        offset = [0] * len(self.shard_sizes)
        offset[self.dim] = sum(self.shard_sizes[:rank])
        return tuple(offset)
```

## Distributed Checkpointing Strategies

### Strategy 1: Gather-then-Write (Simple but Slow)

```
                 Gather to Rank 0
    ┌─────────────────────────────────┐
    │                                 │
    │    ┌───┐  ┌───┐  ┌───┐  ┌───┐  │    ┌───────────────┐
    │    │R0 │  │R1 │  │R2 │  │R3 │  ├───►│  Single File  │
    │    └───┘  └───┘  └───┘  └───┘  │    │    on Disk    │
    │       ▲      │      │      │   │    └───────────────┘
    │       └──────┴──────┴──────┘   │
    │           AllGather           │
    └─────────────────────────────────┘
```

**Pros**: Simple, single file, easy to load
**Cons**: Memory bottleneck at rank 0, serialized I/O, doesn't scale

```python
def gather_checkpoint(model: DistributedModel) -> Dict[str, torch.Tensor]:
    """Gather all shards to rank 0 for checkpointing."""

    full_state = {}

    for name, param in model.named_parameters():
        if dist.get_rank() == 0:
            gathered = [torch.zeros_like(param)
                       for _ in range(dist.get_world_size())]
        else:
            gathered = None

        dist.gather(param, gathered, dst=0)

        if dist.get_rank() == 0:
            # Concatenate shards
            full_state[name] = torch.cat(gathered, dim=param.shard_dim)

    return full_state
```

Memory requirement at rank 0: $O(P \cdot \text{model size})$ — infeasible for large models.

### Strategy 2: Parallel Write (Scalable)

```
           Each Rank Writes Independently
    ┌─────────────────────────────────────────┐
    │                                         │
    │   ┌───┐    ┌───┐    ┌───┐    ┌───┐     │
    │   │R0 │    │R1 │    │R2 │    │R3 │     │
    │   └─┬─┘    └─┬─┘    └─┬─┘    └─┬─┘     │
    │     │        │        │        │       │
    │     ▼        ▼        ▼        ▼       │
    │   ┌───┐    ┌───┐    ┌───┐    ┌───┐     │
    │   │.0 │    │.1 │    │.2 │    │.3 │     │  ◄── Parallel FS
    │   └───┘    └───┘    └───┘    └───┘     │
    │                                         │
    └─────────────────────────────────────────┘
```

**Pros**: Scales with ranks, no memory bottleneck
**Cons**: Many files, requires parallel filesystem

```python
class ParallelCheckpointer:
    """Write checkpoints in parallel across all ranks."""

    def __init__(self, checkpoint_dir: str, world_size: int):
        self.checkpoint_dir = checkpoint_dir
        self.world_size = world_size

    def save(self,
             model: DistributedModel,
             optimizer: DistributedOptimizer,
             step: int) -> None:
        """Save checkpoint with each rank writing its own shard."""

        rank = dist.get_rank()

        # Create step directory
        step_dir = os.path.join(self.checkpoint_dir, f"step_{step:08d}")
        os.makedirs(step_dir, exist_ok=True)

        # Each rank saves its shard
        shard_path = os.path.join(step_dir, f"shard_{rank:05d}.pt")

        shard_state = {
            'model': self._get_local_model_state(model),
            'optimizer': self._get_local_optimizer_state(optimizer),
            'step': step,
            'rank': rank,
            'world_size': self.world_size,
        }

        torch.save(shard_state, shard_path)

        # Barrier ensures all ranks complete
        dist.barrier()

        # Rank 0 writes metadata
        if rank == 0:
            self._write_metadata(step_dir, step)

    def _get_local_model_state(self,
                                model: DistributedModel) -> Dict[str, Any]:
        """Extract local shard state with sharding metadata."""

        state = {}
        for name, param in model.named_parameters():
            state[name] = {
                'data': param.data,
                'sharding_spec': param.sharding_spec,
                'global_shape': param.global_shape,
            }
        return state

    def _write_metadata(self, step_dir: str, step: int) -> None:
        """Write metadata file describing checkpoint structure."""

        metadata = {
            'step': step,
            'world_size': self.world_size,
            'timestamp': time.time(),
            'shards': [f"shard_{r:05d}.pt" for r in range(self.world_size)],
        }

        with open(os.path.join(step_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
```

### I/O Bandwidth Analysis

For a checkpoint of size $C$ bytes across $P$ ranks with filesystem bandwidth $B$:

**Gather-then-Write**:

$$T_{\text{gather}} = \frac{C \cdot (P-1)}{B_{\text{network}}} + \frac{C}{B_{\text{disk}}}$$

**Parallel Write** (with $P$ parallel paths):

$$T_{\text{parallel}} = \frac{C/P}{B_{\text{disk}}/P} = \frac{C}{B_{\text{disk}}}$$

Wait—parallel write with $P$ paths doesn't help if total bandwidth is fixed. But parallel filesystems like Lustre/GPFS provide **aggregate bandwidth scaling**:

$$B_{\text{aggregate}} = \min(P, N_{\text{OSTs}}) \cdot B_{\text{per-OST}}$$

With enough OSTs (Object Storage Targets):

$$T_{\text{parallel}} = \frac{C}{P \cdot B_{\text{per-OST}}}$$

This scales linearly with ranks!

## Asynchronous Checkpointing

Synchronous checkpointing blocks training. Asynchronous checkpointing overlaps I/O with computation.

### The Async Strategy

```
Training:    ═══════════════════════════════════════════►
                   │            │            │
                   ▼            ▼            ▼
             ┌──────────┐ ┌──────────┐ ┌──────────┐
Copy to      │  Buffer  │ │  Buffer  │ │  Buffer  │
pinned mem:  └────┬─────┘ └────┬─────┘ └────┬─────┘
                  │            │            │
                  ▼            ▼            ▼
             ┌──────────────────────────────────────┐
Async I/O:   │  Background thread pool writing      │
             └──────────────────────────────────────┘
```

```python
class AsyncCheckpointer:
    """Checkpoint asynchronously to minimize training disruption."""

    def __init__(self,
                 checkpoint_dir: str,
                 num_io_workers: int = 4,
                 pinned_buffer_size: int = 1 << 30):  # 1 GB

        self.checkpoint_dir = checkpoint_dir
        self.executor = ThreadPoolExecutor(max_workers=num_io_workers)

        # Pinned memory for fast GPU -> CPU transfer
        self.pinned_buffer = torch.empty(
            pinned_buffer_size, dtype=torch.uint8, pin_memory=True
        )

        # Track pending writes
        self.pending_futures: List[Future] = []

        # Double buffering
        self.buffer_a = {}
        self.buffer_b = {}
        self.active_buffer = 'a'

    def checkpoint_async(self,
                         model: DistributedModel,
                         optimizer: DistributedOptimizer,
                         step: int) -> Future:
        """Initiate asynchronous checkpoint."""

        # Wait for any pending checkpoint to complete
        self._wait_pending()

        # Swap buffers
        buffer = self.buffer_a if self.active_buffer == 'a' else self.buffer_b
        self.active_buffer = 'b' if self.active_buffer == 'a' else 'a'

        # Fast copy to CPU (async)
        self._copy_to_buffer(model, optimizer, step, buffer)

        # Submit write to thread pool
        future = self.executor.submit(
            self._write_buffer, buffer, step
        )
        self.pending_futures.append(future)

        return future

    def _copy_to_buffer(self,
                        model: DistributedModel,
                        optimizer: DistributedOptimizer,
                        step: int,
                        buffer: Dict) -> None:
        """Copy state to CPU buffer using CUDA streams."""

        copy_stream = torch.cuda.Stream()

        with torch.cuda.stream(copy_stream):
            buffer['step'] = step
            buffer['model'] = {}

            for name, param in model.named_parameters():
                # Non-blocking copy to pinned memory
                cpu_tensor = torch.empty_like(param, device='cpu',
                                              pin_memory=True)
                cpu_tensor.copy_(param, non_blocking=True)
                buffer['model'][name] = cpu_tensor

            buffer['optimizer'] = {}
            for key, state in optimizer.state.items():
                buffer['optimizer'][key] = {}
                for name, tensor in state.items():
                    if isinstance(tensor, torch.Tensor):
                        cpu_tensor = torch.empty_like(tensor, device='cpu',
                                                     pin_memory=True)
                        cpu_tensor.copy_(tensor, non_blocking=True)
                        buffer['optimizer'][key][name] = cpu_tensor

        # Synchronize copy stream
        copy_stream.synchronize()

    def _write_buffer(self, buffer: Dict, step: int) -> str:
        """Write buffer to disk (runs in thread pool)."""

        rank = dist.get_rank()
        step_dir = os.path.join(self.checkpoint_dir, f"step_{step:08d}")
        os.makedirs(step_dir, exist_ok=True)

        shard_path = os.path.join(step_dir, f"shard_{rank:05d}.pt")
        torch.save(buffer, shard_path)

        return shard_path

    def _wait_pending(self) -> None:
        """Wait for all pending checkpoints to complete."""

        for future in self.pending_futures:
            future.result()
        self.pending_futures.clear()
```

### Overhead Analysis

**Synchronous checkpoint time**:

$$T_{\text{sync}} = T_{\text{copy}} + T_{\text{write}}$$

**Asynchronous overhead** (time training is blocked):

$$T_{\text{async}} = T_{\text{copy}}$$

With pinned memory and CUDA streams:

$$T_{\text{copy}} = \frac{C}{B_{\text{PCIe}}} \approx \frac{C}{25 \text{ GB/s}}$$

For a 100GB checkpoint shard:

- $T_{\text{copy}} \approx 4$ seconds
- $T_{\text{write}} \approx 10$ seconds (100 GB/s parallel FS)

**Savings: 10 seconds per checkpoint!**

## Checkpoint Frequency Optimization

### The Cost Model

Let:

- $T_{\text{step}}$: time per training step
- $T_{\text{ckpt}}$: time to checkpoint (overhead)
- $\lambda$: failure rate (failures per hour)
- $f$: checkpoint frequency (checkpoints per step)

**Total time with checkpointing**:

$$T_{\text{total}} = N_{\text{steps}} \cdot T_{\text{step}} \cdot (1 + f \cdot T_{\text{ckpt}}/T_{\text{step}})$$

**Expected work lost per failure**:

$$W_{\text{lost}} = \frac{T_{\text{step}}}{2f}$$

(On average, failure occurs halfway between checkpoints.)

**Expected restarts**:

$$\mathbb{E}[\text{restarts}] = \lambda \cdot T_{\text{total}}$$

### Optimal Checkpoint Frequency

Minimize total training time including failures:

$$T_{\text{expected}} = N_{\text{steps}} \cdot T_{\text{step}} \cdot \left(1 + f \cdot \frac{T_{\text{ckpt}}}{T_{\text{step}}} + \lambda \cdot \frac{1}{2f}\right)$$

Taking derivative with respect to $f$ and setting to zero:

$$\frac{d T_{\text{expected}}}{df} = N \cdot T_{\text{step}} \cdot \left(\frac{T_{\text{ckpt}}}{T_{\text{step}}} - \frac{\lambda}{2f^2}\right) = 0$$

Solving:

$$f^* = \sqrt{\frac{\lambda \cdot T_{\text{step}}}{2 \cdot T_{\text{ckpt}}}}$$

### Numerical Example

Given:

- $T_{\text{step}} = 1$ second
- $T_{\text{ckpt}} = 60$ seconds (including overhead)
- $\lambda = 0.625$ failures/hour = $1.74 \times 10^{-4}$ failures/second

$$f^* = \sqrt{\frac{1.74 \times 10^{-4} \cdot 1}{2 \cdot 60}} = \sqrt{1.45 \times 10^{-6}} = 0.0012$$

Checkpoint every $1/f^* \approx 830$ steps.

At 1 second/step, checkpoint every **~14 minutes**.

## Checkpoint Consistency in Distributed Systems

### The Consistency Problem

With multiple parallelism dimensions, ensuring consistency is non-trivial:

```
              Time ──────────────────────────►

Rank 0:    ──────┬─────────────┬─────────────
                 │ step 100    │ step 101
                 ▼             ▼
Rank 1:    ────────┬───────────┬─────────────
                   │ step 100  │ step 101
                   ▼           ▼

DANGER: If Rank 0 saves at step 100
        but Rank 1 saves at step 101
        ──► Inconsistent checkpoint!
```

### Solution: Coordinated Checkpointing

```python
class CoordinatedCheckpointer:
    """Ensure all ranks checkpoint at the same logical step."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_step = None

    def maybe_checkpoint(self,
                         model: DistributedModel,
                         optimizer: DistributedOptimizer,
                         step: int,
                         checkpoint_interval: int) -> bool:
        """Checkpoint with distributed coordination."""

        should_checkpoint = (step % checkpoint_interval == 0)

        # All-reduce to ensure agreement
        should_checkpoint_tensor = torch.tensor(
            [1 if should_checkpoint else 0],
            device='cuda'
        )
        dist.all_reduce(should_checkpoint_tensor, op=dist.ReduceOp.MIN)

        if should_checkpoint_tensor.item() == 0:
            return False

        # Barrier before checkpoint
        dist.barrier()

        # Now all ranks are at the same step
        self._save_checkpoint(model, optimizer, step)

        # Barrier after checkpoint
        dist.barrier()

        return True

    def _save_checkpoint(self,
                         model: DistributedModel,
                         optimizer: DistributedOptimizer,
                         step: int) -> None:
        """Save local shard with step verification."""

        rank = dist.get_rank()
        step_dir = os.path.join(self.checkpoint_dir, f"step_{step:08d}")

        # Create directory only on rank 0
        if rank == 0:
            os.makedirs(step_dir, exist_ok=True)
        dist.barrier()

        # Each rank saves its shard
        shard_path = os.path.join(step_dir, f"shard_{rank:05d}.pt")

        state = {
            'model': {name: param.data.cpu()
                     for name, param in model.named_parameters()},
            'optimizer': optimizer.state_dict(),
            'step': step,
            'rank': rank,
        }

        torch.save(state, shard_path)
```

### Handling Pipeline Parallelism

With pipeline parallelism, different stages may be processing different micro-batches:

```
                  Micro-batch timeline
Stage 0:    [MB0][MB1][MB2][MB3]     ← Ahead
Stage 1:        [MB0][MB1][MB2][MB3] ← Behind
```

**Solution**: Checkpoint at pipeline bubble (when all stages are synchronized):

```python
class PipelineCheckpointer:
    """Checkpoint at pipeline synchronization points."""

    def __init__(self, pp_group: dist.ProcessGroup):
        self.pp_group = pp_group
        self.pp_rank = dist.get_rank(pp_group)
        self.pp_size = dist.get_world_size(pp_group)

    def checkpoint_at_bubble(self,
                             model: nn.Module,
                             optimizer: torch.optim.Optimizer,
                             step: int) -> None:
        """Checkpoint when pipeline is drained."""

        # Signal checkpoint intent from last stage
        if self.pp_rank == self.pp_size - 1:
            checkpoint_signal = torch.ones(1, device='cuda')
        else:
            checkpoint_signal = torch.zeros(1, device='cuda')

        # Broadcast from last stage
        dist.broadcast(checkpoint_signal,
                      src=self.pp_size - 1,
                      group=self.pp_group)

        # All stages wait for pipeline to drain
        # (This happens naturally at the end of each step in 1F1B)

        # Now all stages are synchronized
        self._save_stage_checkpoint(model, optimizer, step)

    def _save_stage_checkpoint(self,
                               model: nn.Module,
                               optimizer: torch.optim.Optimizer,
                               step: int) -> None:
        """Save this pipeline stage's checkpoint."""

        stage_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'pp_rank': self.pp_rank,
        }

        path = f"checkpoint_step{step}_stage{self.pp_rank}.pt"
        torch.save(stage_state, path)
```

## Checkpoint Loading and Recovery

### Resilient Loading

```python
class ResilientLoader:
    """Load checkpoints with validation and recovery."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the latest valid checkpoint."""

        # Find all checkpoint directories
        checkpoints = self._find_checkpoints()

        # Try from newest to oldest
        for ckpt_dir in reversed(checkpoints):
            try:
                state = self._load_checkpoint(ckpt_dir)
                if self._validate_checkpoint(state):
                    return state
                else:
                    print(f"Checkpoint {ckpt_dir} failed validation")
            except Exception as e:
                print(f"Failed to load {ckpt_dir}: {e}")

        return None

    def _find_checkpoints(self) -> List[str]:
        """Find checkpoint directories sorted by step."""

        pattern = os.path.join(self.checkpoint_dir, "step_*")
        dirs = glob.glob(pattern)

        # Sort by step number
        def extract_step(path):
            match = re.search(r'step_(\d+)', path)
            return int(match.group(1)) if match else 0

        return sorted(dirs, key=extract_step)

    def _load_checkpoint(self, ckpt_dir: str) -> Dict[str, Any]:
        """Load all shards of a checkpoint."""

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Read metadata
        with open(os.path.join(ckpt_dir, 'metadata.json')) as f:
            metadata = json.load(f)

        # Verify world size matches
        if metadata['world_size'] != world_size:
            raise ValueError(
                f"Checkpoint world_size {metadata['world_size']} "
                f"!= current {world_size}"
            )

        # Load this rank's shard
        shard_path = os.path.join(ckpt_dir, f"shard_{rank:05d}.pt")
        state = torch.load(shard_path, map_location='cuda')

        return state

    def _validate_checkpoint(self, state: Dict[str, Any]) -> bool:
        """Validate checkpoint integrity."""

        # Check required keys
        required = ['model', 'optimizer', 'step']
        for key in required:
            if key not in state:
                return False

        # Check model state
        for name, param in state['model'].items():
            if torch.isnan(param).any():
                return False
            if torch.isinf(param).any():
                return False

        # All ranks must agree checkpoint is valid
        valid_tensor = torch.ones(1, device='cuda')
        dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)

        return valid_tensor.item() == 1
```

### Resharding for Different Parallelism

When loading a checkpoint with different parallelism configuration:

```python
class ReshardingLoader:
    """Load checkpoint with different sharding configuration."""

    def __init__(self,
                 ckpt_world_size: int,
                 current_world_size: int,
                 ckpt_tp_size: int,
                 current_tp_size: int):
        self.ckpt_world_size = ckpt_world_size
        self.current_world_size = current_world_size
        self.ckpt_tp_size = ckpt_tp_size
        self.current_tp_size = current_tp_size

    def load_and_reshard(self,
                         ckpt_dir: str,
                         model: DistributedModel) -> None:
        """Load checkpoint and reshard for current configuration."""

        rank = dist.get_rank()

        if self.ckpt_tp_size == self.current_tp_size:
            # Simple case: same TP, just load
            self._load_direct(ckpt_dir, model, rank)
        elif self.current_tp_size > self.ckpt_tp_size:
            # More TP shards: split each checkpoint shard
            self._load_and_split(ckpt_dir, model, rank)
        else:
            # Fewer TP shards: merge checkpoint shards
            self._load_and_merge(ckpt_dir, model, rank)

    def _load_and_split(self,
                        ckpt_dir: str,
                        model: DistributedModel,
                        rank: int) -> None:
        """Split checkpoint shards for higher TP degree."""

        # Calculate which checkpoint shard this rank needs
        split_factor = self.current_tp_size // self.ckpt_tp_size
        ckpt_rank = rank // split_factor
        split_idx = rank % split_factor

        # Load the checkpoint shard
        shard_path = os.path.join(ckpt_dir, f"shard_{ckpt_rank:05d}.pt")
        state = torch.load(shard_path, map_location='cuda')

        # Split each parameter
        for name, param in model.named_parameters():
            ckpt_tensor = state['model'][name]['data']
            shard_dim = param.sharding_spec.dim

            # Split along shard dimension
            chunks = torch.chunk(ckpt_tensor, split_factor, dim=shard_dim)
            param.data.copy_(chunks[split_idx])

    def _load_and_merge(self,
                        ckpt_dir: str,
                        model: DistributedModel,
                        rank: int) -> None:
        """Merge checkpoint shards for lower TP degree."""

        merge_factor = self.ckpt_tp_size // self.current_tp_size

        # Load multiple checkpoint shards
        tensors_to_merge = []
        for i in range(merge_factor):
            ckpt_rank = rank * merge_factor + i
            shard_path = os.path.join(ckpt_dir, f"shard_{ckpt_rank:05d}.pt")
            state = torch.load(shard_path, map_location='cuda')
            tensors_to_merge.append(state)

        # Merge each parameter
        for name, param in model.named_parameters():
            shard_dim = param.sharding_spec.dim

            chunks = [t['model'][name]['data'] for t in tensors_to_merge]
            merged = torch.cat(chunks, dim=shard_dim)
            param.data.copy_(merged)
```

## Elastic Training

Elastic training allows the cluster size to change during training.

### The Elasticity Challenge

```
Initial: 64 GPUs, DP=64, TP=1

         ┌───┐ ┌───┐ ┌───┐ ... ┌───┐
         │G0 │ │G1 │ │G2 │     │G63│
         └───┘ └───┘ └───┘     └───┘
                    │
                    ▼  8 GPUs fail

After:   56 GPUs, DP=56, TP=1

         ┌───┐ ┌───┐ ┌───┐ ... ┌───┐
         │G0 │ │G1 │ │G2 │     │G55│
         └───┘ └───┘ └───┘     └───┘

         Batch size changes!
         Learning rate must adjust!
```

### Elastic Data Parallelism

```python
class ElasticDataParallel:
    """Data parallelism that handles changing cluster size."""

    def __init__(self,
                 model: nn.Module,
                 base_batch_size: int,
                 base_lr: float):

        self.model = model
        self.base_batch_size = base_batch_size
        self.base_lr = base_lr

        # Track cluster state
        self.initial_world_size = dist.get_world_size()
        self.current_world_size = self.initial_world_size

    def handle_membership_change(self,
                                  new_world_size: int,
                                  optimizer: torch.optim.Optimizer) -> None:
        """Adjust for new cluster size."""

        old_world_size = self.current_world_size
        self.current_world_size = new_world_size

        # Adjust learning rate (linear scaling)
        lr_scale = new_world_size / old_world_size
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_scale

        # Effective batch size changes automatically
        # (each rank still processes base_batch_size)

        print(f"Cluster resized: {old_world_size} → {new_world_size}")
        print(f"New effective batch: {new_world_size * self.base_batch_size}")
        print(f"New learning rate: {optimizer.param_groups[0]['lr']:.6f}")

    def forward_backward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward and backward with proper gradient scaling."""

        # Forward
        output = self.model(batch)
        loss = self.compute_loss(output)

        # Backward
        loss.backward()

        # All-reduce gradients
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self.current_world_size

        return loss
```

### Elastic Checkpointing

For elastic training, checkpoints must be restorable with any cluster size:

```python
class ElasticCheckpoint:
    """Checkpoint format that supports elastic loading."""

    def save(self,
             model: nn.Module,
             optimizer: torch.optim.Optimizer,
             step: int,
             path: str) -> None:
        """Save elastic-compatible checkpoint."""

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Each rank saves with its position info
        state = {
            'model_shard': model.state_dict(),
            'optimizer_shard': optimizer.state_dict(),
            'step': step,
            'rank': rank,
            'world_size': world_size,

            # For resharding
            'sharding_info': self._get_sharding_info(model),
        }

        shard_path = os.path.join(path, f"shard_{rank:05d}_of_{world_size}.pt")
        torch.save(state, shard_path)

        # Also save combined model for easy single-GPU loading
        if rank == 0:
            full_model = self._gather_full_model(model)
            torch.save(full_model, os.path.join(path, "model_full.pt"))

    def load_elastic(self,
                     model: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     path: str) -> int:
        """Load checkpoint into potentially different cluster size."""

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Find checkpoint metadata
        shards = glob.glob(os.path.join(path, "shard_*_of_*.pt"))
        ckpt_world_size = self._extract_world_size(shards[0])

        if world_size == ckpt_world_size:
            # Easy case: same size
            state = torch.load(
                os.path.join(path, f"shard_{rank:05d}_of_{world_size}.pt"),
                map_location='cuda'
            )
            model.load_state_dict(state['model_shard'])
            optimizer.load_state_dict(state['optimizer_shard'])

        elif world_size < ckpt_world_size:
            # Fewer GPUs: each rank loads multiple shards
            self._load_merged(path, model, optimizer,
                            ckpt_world_size, world_size, rank)

        else:
            # More GPUs: distribute shards across ranks
            self._load_split(path, model, optimizer,
                           ckpt_world_size, world_size, rank)

        return state['step']

    def _load_merged(self,
                     path: str,
                     model: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     ckpt_size: int,
                     current_size: int,
                     rank: int) -> None:
        """Load when current cluster is smaller than checkpoint."""

        # Calculate which shards this rank handles
        shards_per_rank = ckpt_size // current_size
        remainder = ckpt_size % current_size

        start_shard = rank * shards_per_rank + min(rank, remainder)
        end_shard = start_shard + shards_per_rank + (1 if rank < remainder else 0)

        # Load and merge shards
        merged_model_state = {}
        for shard_idx in range(start_shard, end_shard):
            shard_path = os.path.join(path,
                                      f"shard_{shard_idx:05d}_of_{ckpt_size}.pt")
            state = torch.load(shard_path, map_location='cuda')

            for name, param in state['model_shard'].items():
                if name not in merged_model_state:
                    merged_model_state[name] = []
                merged_model_state[name].append(param)

        # Concatenate sharded parameters
        for name, params in merged_model_state.items():
            merged_model_state[name] = torch.cat(params, dim=0)

        model.load_state_dict(merged_model_state)
```

## Incremental and Delta Checkpointing

For very large models, full checkpoints are expensive. Incremental checkpointing saves only changes.

### Delta Checkpoint

```python
class DeltaCheckpointer:
    """Save only parameter changes between checkpoints."""

    def __init__(self, threshold: float = 1e-6):
        self.threshold = threshold
        self.base_checkpoint = None
        self.base_step = None

    def checkpoint(self,
                   model: nn.Module,
                   step: int,
                   path: str) -> None:
        """Save delta or full checkpoint."""

        current_state = {name: param.data.clone()
                        for name, param in model.named_parameters()}

        if self.base_checkpoint is None:
            # First checkpoint: save full
            self._save_full(current_state, step, path)
            self.base_checkpoint = current_state
            self.base_step = step
        else:
            # Compute and save delta
            delta = self._compute_delta(current_state)
            self._save_delta(delta, step, self.base_step, path)

    def _compute_delta(self,
                       current: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compute sparse delta from base checkpoint."""

        delta = {}

        for name, current_param in current.items():
            base_param = self.base_checkpoint[name]
            diff = current_param - base_param

            # Sparsify: only keep significant changes
            mask = torch.abs(diff) > self.threshold

            if mask.any():
                delta[name] = {
                    'indices': mask.nonzero(as_tuple=True),
                    'values': diff[mask],
                    'shape': diff.shape,
                }

        return delta

    def _save_delta(self,
                    delta: Dict[str, Any],
                    step: int,
                    base_step: int,
                    path: str) -> None:
        """Save delta checkpoint."""

        state = {
            'type': 'delta',
            'step': step,
            'base_step': base_step,
            'delta': delta,
        }

        delta_path = os.path.join(path, f"delta_{base_step}_to_{step}.pt")
        torch.save(state, delta_path)

    def load_with_deltas(self,
                         base_path: str,
                         delta_paths: List[str],
                         model: nn.Module) -> None:
        """Load base checkpoint and apply deltas."""

        # Load base
        base_state = torch.load(base_path)
        model.load_state_dict(base_state['model'])

        # Apply deltas in order
        for delta_path in delta_paths:
            delta_state = torch.load(delta_path)
            self._apply_delta(model, delta_state['delta'])

    def _apply_delta(self,
                     model: nn.Module,
                     delta: Dict[str, Any]) -> None:
        """Apply delta to model parameters."""

        for name, param in model.named_parameters():
            if name in delta:
                d = delta[name]
                param.data[d['indices']] += d['values']
```

### Compression for Checkpoints

```python
class CompressedCheckpointer:
    """Checkpoint with compression for storage efficiency."""

    def __init__(self, compression: str = 'lz4'):
        self.compression = compression

        if compression == 'lz4':
            import lz4.frame
            self.compress = lz4.frame.compress
            self.decompress = lz4.frame.decompress
        elif compression == 'zstd':
            import zstandard
            self.compressor = zstandard.ZstdCompressor(level=3)
            self.decompressor = zstandard.ZstdDecompressor()
            self.compress = self.compressor.compress
            self.decompress = self.decompressor.decompress

    def save(self, state: Dict[str, Any], path: str) -> None:
        """Save compressed checkpoint."""

        # Serialize to bytes
        buffer = io.BytesIO()
        torch.save(state, buffer)
        data = buffer.getvalue()

        # Compress
        compressed = self.compress(data)

        # Write
        with open(path, 'wb') as f:
            f.write(compressed)

        ratio = len(data) / len(compressed)
        print(f"Compression ratio: {ratio:.2f}x")

    def load(self, path: str) -> Dict[str, Any]:
        """Load compressed checkpoint."""

        with open(path, 'rb') as f:
            compressed = f.read()

        data = self.decompress(compressed)
        buffer = io.BytesIO(data)

        return torch.load(buffer, map_location='cuda')
```

## Complete Fault-Tolerant Training Loop

```python
class FaultTolerantTrainer:
    """Complete training loop with fault tolerance."""

    def __init__(self,
                 model: DistributedModel,
                 optimizer: torch.optim.Optimizer,
                 dataloader: DataLoader,
                 checkpoint_dir: str,
                 checkpoint_interval: int = 1000,
                 max_failures: int = 100):

        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.max_failures = max_failures

        self.checkpointer = AsyncCheckpointer(checkpoint_dir)
        self.loader = ResilientLoader(checkpoint_dir)

        # Failure tracking
        self.failure_count = 0
        self.last_checkpoint_step = 0

    def train(self, total_steps: int) -> None:
        """Main training loop with fault tolerance."""

        # Try to resume from checkpoint
        start_step = self._maybe_resume()

        step = start_step

        try:
            for batch in self.dataloader:
                if step >= total_steps:
                    break

                # Training step
                loss = self._train_step(batch)

                step += 1

                # Checkpoint
                if step % self.checkpoint_interval == 0:
                    self._checkpoint(step)

                # Logging
                if step % 100 == 0:
                    print(f"Step {step}, Loss: {loss:.4f}")

        except Exception as e:
            self._handle_failure(e, step)

            if self.failure_count < self.max_failures:
                # Retry from last checkpoint
                self.train(total_steps)
            else:
                raise RuntimeError(f"Exceeded max failures: {self.max_failures}")

        # Final checkpoint
        self._checkpoint(step, force=True)

    def _maybe_resume(self) -> int:
        """Resume from checkpoint if available."""

        state = self.loader.load_latest()

        if state is None:
            print("No checkpoint found, starting fresh")
            return 0

        # Restore state
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])

        step = state['step']
        self.last_checkpoint_step = step

        print(f"Resumed from checkpoint at step {step}")

        return step

    def _train_step(self, batch: torch.Tensor) -> float:
        """Single training step."""

        self.optimizer.zero_grad()

        output = self.model(batch)
        loss = output.loss

        loss.backward()

        # Gradient synchronization
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= dist.get_world_size()

        self.optimizer.step()

        return loss.item()

    def _checkpoint(self, step: int, force: bool = False) -> None:
        """Save checkpoint asynchronously."""

        # Coordinate across ranks
        should_checkpoint = torch.tensor([1], device='cuda')
        dist.all_reduce(should_checkpoint, op=dist.ReduceOp.MIN)

        if should_checkpoint.item() == 0 and not force:
            return

        dist.barrier()

        self.checkpointer.checkpoint_async(
            self.model, self.optimizer, step
        )

        self.last_checkpoint_step = step

    def _handle_failure(self, error: Exception, step: int) -> None:
        """Handle training failure."""

        self.failure_count += 1

        work_lost = step - self.last_checkpoint_step

        print(f"Failure at step {step}: {error}")
        print(f"Work lost: {work_lost} steps")
        print(f"Total failures: {self.failure_count}")

        # Wait for any pending checkpoints
        self.checkpointer._wait_pending()

        # Reset distributed state
        dist.barrier()
```

## Exercises

1. **MTBF calculation**: A cluster has 4,096 GPUs (MTBF 25,000 hours), 512 hosts (MTBF 8,000 hours), and 32 network switches (MTBF 100,000 hours). Calculate the system MTBF.

??? success "Solution"
    **System MTBF formula:**

    For independent components, failure rates add:

    $$\lambda_{system} = \sum_i n_i \cdot \lambda_i = \sum_i \frac{n_i}{MTBF_i}$$

    $$MTBF_{system} = \frac{1}{\lambda_{system}}$$

    **Component failure rates:**

    | Component | Count | MTBF (hrs) | Failure Rate (per hr) |
    |-----------|-------|------------|----------------------|
    | GPUs | 4,096 | 25,000 | $\frac{4096}{25000} = 0.164$ |
    | Hosts | 512 | 8,000 | $\frac{512}{8000} = 0.064$ |
    | Switches | 32 | 100,000 | $\frac{32}{100000} = 0.00032$ |

    **System failure rate:**

    $$\lambda_{system} = 0.164 + 0.064 + 0.00032 = 0.228 \text{ failures/hour}$$

    **System MTBF:**

    $$MTBF_{system} = \frac{1}{0.228} = \boxed{4.38 \text{ hours}}$$

    **Analysis:**

    | Component | Contribution to Failure Rate |
    |-----------|------------------------------|
    | GPUs | 72% |
    | Hosts | 28% |
    | Switches | 0.14% |

    GPUs dominate the failure rate due to their quantity, even with better individual reliability than hosts.

    **Implications:**

    - Expect a failure every ~4.4 hours
    - Checkpoint interval should be << 4.4 hours
    - Investment in GPU reliability has highest ROI

2. **Optimal checkpoint interval**: Given MTBF of 2 hours, checkpoint time of 30 seconds, and step time of 500ms, what's the optimal checkpoint interval?

??? success "Solution"
    **Given:**

    - MTBF = 2 hours = 7,200 seconds
    - Checkpoint time: $C$ = 30 seconds
    - Step time: $T_{step}$ = 0.5 seconds

    **Optimal checkpoint interval formula (Young/Daly):**

    $$I^* = \sqrt{2 \cdot C \cdot MTBF}$$

    **Calculation:**

    $$I^* = \sqrt{2 \times 30 \times 7200} = \sqrt{432,000} = \boxed{657 \text{ seconds}} \approx 11 \text{ minutes}$$

    **Convert to steps:**

    $$\text{Steps between checkpoints} = \frac{657}{0.5} = \boxed{1,314 \text{ steps}}$$

    **Verify optimality:**

    Total time with checkpointing = training time + checkpoint overhead + expected wasted work

    $$T_{total} = T_{train} \times \left(1 + \frac{C}{I} + \frac{I}{2 \times MTBF}\right)$$

    | Interval | Checkpoint Overhead | Expected Waste | Total Overhead |
    |----------|--------------------:|---------------:|---------------:|
    | 200s | 15.0% | 1.4% | 16.4% |
    | 400s | 7.5% | 2.8% | 10.3% |
    | **657s** | 4.6% | 4.6% | **9.1%** |
    | 1000s | 3.0% | 6.9% | 9.9% |
    | 2000s | 1.5% | 13.9% | 15.4% |

    At the optimal interval, checkpoint overhead equals expected wasted work (both ~4.6%).

    **Practical considerations:**

    - Round to nice step count: 1,300 or 1,500 steps
    - Account for async checkpointing reducing effective $C$
    - Monitor actual failure rate and adjust

    **Summary:**

    | Metric | Value |
    |--------|-------|
    | Optimal interval | 657 seconds |
    | Steps between checkpoints | 1,314 |
    | Total overhead at optimal | 9.1% |

3. **Async overhead**: Model size is 50GB per rank. PCIe bandwidth is 32 GB/s. Parallel FS bandwidth per rank is 2 GB/s. Calculate sync vs async checkpoint overhead.

??? success "Solution"
    **Given:**

    - Model size per rank: 50 GB
    - PCIe bandwidth (GPU→CPU): 32 GB/s
    - Parallel FS bandwidth (CPU→storage): 2 GB/s

    **Synchronous checkpointing:**

    All operations are on the critical path:

    $$T_{sync} = T_{GPU \to CPU} + T_{CPU \to storage}$$
    $$T_{sync} = \frac{50}{32} + \frac{50}{2} = 1.56 + 25 = \boxed{26.56 \text{ seconds}}$$

    Training is blocked for the entire duration.

    **Asynchronous checkpointing:**

    Only GPU→CPU copy is on critical path (training resumes after):

    $$T_{async}^{blocking} = T_{GPU \to CPU} = \frac{50}{32} = \boxed{1.56 \text{ seconds}}$$

    Storage write happens in background.

    **Overhead comparison:**

    | Approach | Blocking Time | Speedup |
    |----------|---------------|---------|
    | Synchronous | 26.56s | 1× |
    | Asynchronous | 1.56s | **17×** |

    **Effective checkpoint time reduction:**

    $$\text{Reduction} = \frac{26.56 - 1.56}{26.56} = \boxed{94\%}$$

    **Background write considerations:**

    The 25-second background write must complete before the next checkpoint:

    $$\text{Min checkpoint interval} > 25 \text{ seconds}$$

    For optimal interval of ~657s (from Exercise 2), this is easily satisfied.

    **Impact on optimal checkpointing:**

    Using async, effective checkpoint time becomes 1.56s instead of 26.56s:

    $$I_{async}^* = \sqrt{2 \times 1.56 \times 7200} = 150 \text{ seconds}$$

    More frequent checkpoints reduce wasted work!

    **Memory overhead:**

    Async requires double-buffering in CPU memory:

    $$M_{overhead} = 50 \text{ GB/rank}$$

    | Metric | Sync | Async |
    |--------|------|-------|
    | Blocking time | 26.56s | 1.56s |
    | Optimal interval | 657s | 150s |
    | Expected wasted work | 4.6% | 1.0% |
    | CPU memory overhead | 0 | 50 GB |

4. **Resharding math**: A checkpoint was saved with TP=8. Loading with TP=4. Each parameter was sharded along dimension 0. Write the resharding formula.

??? success "Solution"
    **Scenario:**

    - Saved: TP=8 (8 shards along dimension 0)
    - Loading: TP=4 (need 4 shards along dimension 0)

    **Original sharding (TP=8):**

    For a parameter with shape $[D_0, D_1, ...]$:

    Each rank $r \in [0,7]$ holds:

    $$\text{shard}_r = \text{param}\left[\frac{r \cdot D_0}{8} : \frac{(r+1) \cdot D_0}{8}, :, ...\right]$$

    **Target sharding (TP=4):**

    Each rank $r' \in [0,3]$ needs:

    $$\text{shard}_{r'} = \text{param}\left[\frac{r' \cdot D_0}{4} : \frac{(r'+1) \cdot D_0}{4}, :, ...\right]$$

    **Resharding formula:**

    Since $8/4 = 2$, each new shard is formed by concatenating 2 old shards:

    $$\boxed{\text{shard}_{r'}^{new} = \text{concat}\left(\text{shard}_{2r'}^{old}, \text{shard}_{2r'+1}^{old}\right)}$$

    **Mapping:**

    | New Rank (TP=4) | Old Ranks (TP=8) | Formula |
    |-----------------|------------------|---------|
    | 0 | 0, 1 | concat(shard_0, shard_1) |
    | 1 | 2, 3 | concat(shard_2, shard_3) |
    | 2 | 4, 5 | concat(shard_4, shard_5) |
    | 3 | 6, 7 | concat(shard_6, shard_7) |

    **General resharding formula:**

    For TP_old → TP_new where TP_old > TP_new (merging shards):

    $$k = \frac{TP_{old}}{TP_{new}}$$

    $$\text{shard}_{r'}^{new} = \text{concat}\left(\text{shard}_{k \cdot r'}^{old}, \text{shard}_{k \cdot r'+1}^{old}, ..., \text{shard}_{k \cdot r'+(k-1)}^{old}\right)$$

    For TP_old < TP_new (splitting shards):

    $$k = \frac{TP_{new}}{TP_{old}}$$

    $$\text{shard}_{r'}^{new} = \text{split}_k\left(\text{shard}_{\lfloor r'/k \rfloor}^{old}\right)[r' \mod k]$$

    **Python implementation:**

    ```python
    def reshard_tp(old_shards, old_tp, new_tp, dim=0):
        """Reshard from old_tp to new_tp along specified dimension."""
        # Reconstruct full tensor
        full = torch.cat(old_shards, dim=dim)

        # Create new shards
        chunk_size = full.shape[dim] // new_tp
        new_shards = torch.split(full, chunk_size, dim=dim)

        return list(new_shards)

    # Example: TP=8 → TP=4
    # old_shards: list of 8 tensors
    # new_shards: list of 4 tensors (each 2× larger in dim 0)
    new_shards = reshard_tp(old_shards, old_tp=8, new_tp=4)
    ```

5. **Compression trade-off**: Checkpoint size 100GB. LZ4 compression ratio 2.5x at 4 GB/s. Uncompressed write at 10 GB/s. Which is faster for save? For load?

??? success "Solution"
    **Given:**

    - Checkpoint size: 100 GB (uncompressed)
    - LZ4 compression ratio: 2.5× → compressed size = 40 GB
    - LZ4 throughput: 4 GB/s (compression and decompression)
    - Uncompressed I/O: 10 GB/s

    **Save time analysis:**

    **Without compression:**
    $$T_{save}^{uncomp} = \frac{100}{10} = 10 \text{ seconds}$$

    **With compression:**
    $$T_{compress} = \frac{100}{4} = 25 \text{ seconds}$$
    $$T_{write} = \frac{40}{10} = 4 \text{ seconds}$$

    If sequential: $T_{save}^{comp} = 25 + 4 = 29$ seconds

    If pipelined (compress while writing): $T_{save}^{comp} = \max(25, 4) = 25$ seconds

    **Save verdict:**

    $$\boxed{\text{Uncompressed is faster for save: 10s vs 25s}}$$

    **Load time analysis:**

    **Without compression:**
    $$T_{load}^{uncomp} = \frac{100}{10} = 10 \text{ seconds}$$

    **With compression:**
    $$T_{read} = \frac{40}{10} = 4 \text{ seconds}$$
    $$T_{decompress} = \frac{100}{4} = 25 \text{ seconds}$$

    If sequential: $T_{load}^{comp} = 4 + 25 = 29$ seconds

    If pipelined: $T_{load}^{comp} = \max(4, 25) = 25$ seconds

    **Load verdict:**

    $$\boxed{\text{Uncompressed is faster for load: 10s vs 25s}}$$

    **Summary:**

    | Operation | Uncompressed | Compressed | Winner |
    |-----------|--------------|------------|--------|
    | Save | 10s | 25s | Uncompressed |
    | Load | 10s | 25s | Uncompressed |
    | Storage | 100 GB | 40 GB | Compressed (2.5×) |

    **When compression wins:**

    Compression becomes faster when I/O is the bottleneck:

    $$T_{uncomp} > T_{comp}$$
    $$\frac{S}{BW_{io}} > \frac{S}{BW_{comp}}$$

    Solve: $BW_{io} < \frac{BW_{comp}}{ratio} = \frac{4}{2.5} = 1.6$ GB/s

    **If storage bandwidth were 1.5 GB/s:**

    | Operation | Uncompressed | Compressed |
    |-----------|--------------|------------|
    | Save | 66.7s | max(25, 26.7) = 26.7s |
    | Load | 66.7s | max(26.7, 25) = 26.7s |

    Compression wins at low I/O bandwidth!

    **Recommendation:** Use compression when storage bandwidth < 1.6 GB/s.

6. **Elastic batch sizing**: Training at 64 GPUs with batch 2048 and LR 0.001. Cluster shrinks to 48 GPUs. What should the new batch size and LR be under linear scaling?

??? success "Solution"
    **Given:**

    - Original: 64 GPUs, batch=2048, LR=0.001
    - New: 48 GPUs

    **Per-GPU batch size:**

    $$B_{gpu} = \frac{2048}{64} = 32 \text{ samples/GPU}$$

    **Option 1: Keep per-GPU batch constant**

    New global batch:

    $$B_{new} = 32 \times 48 = \boxed{1536}$$

    **Linear scaling rule for LR:**

    $$\frac{LR_{new}}{LR_{old}} = \frac{B_{new}}{B_{old}}$$

    $$LR_{new} = 0.001 \times \frac{1536}{2048} = \boxed{0.00075}$$

    **Summary (Option 1):**

    | Parameter | Original (64 GPU) | New (48 GPU) |
    |-----------|-------------------|--------------|
    | GPUs | 64 | 48 |
    | Batch/GPU | 32 | 32 |
    | Global batch | 2048 | 1536 |
    | Learning rate | 0.001 | 0.00075 |

    **Option 2: Maintain global batch with gradient accumulation**

    Keep B=2048 by accumulating gradients:

    $$\text{accum steps} = \frac{2048}{48 \times 32} = \frac{2048}{1536} = 1.33$$

    Not an integer! Adjust per-GPU batch:

    - Option A: $B_{gpu}=43$, accum=1 → global batch = 2064 (close)
    - Option B: $B_{gpu}=32$, accum=2 → global batch = 3072 (too high)
    - Option C: $B_{gpu}=21$, accum=2 → global batch = 2016 (close)

    **Option 2A configuration:**

    | Parameter | Value |
    |-----------|-------|
    | GPUs | 48 |
    | Batch/GPU | 43 |
    | Global batch | 2064 |
    | Learning rate | 0.001 × (2064/2048) = 0.001008 |

    **Practical recommendation:**

    Option 1 (reduced batch with scaled LR) is cleaner and maintains training dynamics.

    **Final answer:**

    $$\boxed{B = 1536, \quad LR = 0.00075}$$

    **Throughput impact:**

    Tokens per step: $1536 \times S$ vs $2048 \times S$ (75% of original)

    Steps needed: 1.33× more steps to process same data

    Time per step: similar (compute bound)

    **Total training time increase:** ~33% slower

## Key Takeaways

1. **Failures are statistical certainties at scale**: Plan for them, don't hope to avoid them.

2. **Checkpoint frequency is optimizable**: There's a mathematical optimum balancing overhead and work loss.

3. **Async checkpointing is essential**: Overlap I/O with compute to minimize training disruption.

4. **Consistency requires coordination**: Distributed checkpoints need barriers and verification.

5. **Resharding enables elasticity**: Checkpoints should be loadable with different parallelism configs.

6. **Compression and delta methods reduce storage**: But consider I/O bandwidth trade-offs.

7. **Fault tolerance is a system property**: Requires checkpointing, detection, and recovery working together.

