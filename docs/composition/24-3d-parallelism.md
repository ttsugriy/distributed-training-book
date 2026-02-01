---
title: "3D Parallelism"
subtitle: "The Canonical Composition: DP × TP × PP"
---

<div class="chapter-opener" markdown>
No single parallelism strategy scales to thousands of GPUs. Data parallelism wastes memory. Tensor parallelism drowns in communication. Pipeline parallelism creates bubbles. The solution: compose all three, each operating at its optimal scale.
</div>

<div class="investigation-question" markdown>
**The Question**: You have 1024 GPUs and a 175B parameter model. DP alone: each GPU needs the full model—impossible. TP alone: 1024-way splits create 1023 communication barriers per layer. PP alone: 1023/1024 = 99.9% bubble overhead. How do you combine them to train efficiently?
</div>

!!! abstract "Chapter Map"
    **Prerequisites**: Chapters 14–16 (DP, TP, PP fundamentals), Chapter 23 (device mesh abstraction)

    **Key insight**: Each parallelism strategy has an optimal scale. TP works best within nodes (fast NVLink). PP spans nodes with minimal communication. DP scales the outer dimension. The canonical 3D composition: TP within nodes, PP across nodes, DP across replicas.

## The Limits of Single Strategies

Each parallelism strategy has a natural scale where it excels:

### Data Parallelism

**Strength**: Perfect scaling of computation.

**Weakness**: Memory redundancy—every GPU stores the full model.

$$\text{Memory per GPU} = M_{\text{model}} + M_{\text{optimizer}} + M_{\text{activation}}$$

For a 175B model with Adam optimizer in mixed precision:

$$M_{\text{model}} + M_{\text{optimizer}} = 175\text{B} \times (2 + 8) = 1.75\text{ TB}$$

No GPU can hold this.

### Tensor Parallelism

**Strength**: Divides model memory linearly.

**Weakness**: Communication scales with hidden dimension.

For $P$ GPUs, each linear layer requires:

$$\text{Communication} = O(P \times H \times \text{batch}) \text{ per forward and backward}$$

With $P = 1024$, AllReduce latency dominates:

$$T_{\text{allreduce}} = 2(P-1) \alpha + \frac{2(P-1)}{P\beta} \cdot n$$

The $2046\alpha$ latency term destroys throughput.

### Pipeline Parallelism

**Strength**: Minimal communication (only at stage boundaries).

**Weakness**: Pipeline bubbles.

With $P$ stages and micro-batch count $M$:

$$\text{Bubble fraction} = \frac{P - 1}{P + M - 1}$$

With $P = 1024$ and $M = 4$:

$$\text{Bubble fraction} = \frac{1023}{1027} = 99.6\%$$

## The 3D Composition

The insight: different strategies have different optimal scales. Compose them hierarchically.

### The Three Dimensions

```
                        Global (1024 GPUs)
                              │
              ┌───────────────┴───────────────┐
              │           DP = 32             │
              │    (replication groups)       │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              │           PP = 8              │
              │     (pipeline stages)         │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              │           TP = 4              │
              │    (tensor sharding)          │
              └───────────────┘
```

**Total GPUs**: $N = \text{DP} \times \text{PP} \times \text{TP} = 32 \times 8 \times 4 = 1024$

### Why This Ordering?

**TP innermost**: Requires lowest latency (NVLink within node).

**PP middle**: Moderate bandwidth, can span nodes.

**DP outermost**: Most bandwidth-tolerant (large gradient tensors).

```
Node 0 (8 GPUs):                Node 1 (8 GPUs):
┌─────────┬─────────┐           ┌─────────┬─────────┐
│ Stage 0 │ Stage 1 │           │ Stage 0 │ Stage 1 │
│ TP=0-3  │ TP=0-3  │           │ TP=0-3  │ TP=0-3  │
└─────────┴─────────┘           └─────────┴─────────┘
     │         │                     │         │
     └─────────┼─────────────────────┘         │
               │        DP sync                │
               └───────────────────────────────┘
```

## Mapping to Device Mesh

With the device mesh abstraction from Chapter 23, 3D parallelism becomes a mesh configuration:

```python
from typing import Tuple, Optional
import torch
import torch.distributed as dist
import numpy as np

class ThreeDMesh:
    """Device mesh configured for 3D parallelism."""

    def __init__(
        self,
        dp_size: int,
        pp_size: int,
        tp_size: int,
        device_ids: Optional[np.ndarray] = None
    ):
        self.dp_size = dp_size
        self.pp_size = pp_size
        self.tp_size = tp_size

        world_size = dp_size * pp_size * tp_size

        if device_ids is None:
            device_ids = np.arange(world_size)

        # Shape: (DP, PP, TP)
        self.mesh = device_ids.reshape(dp_size, pp_size, tp_size)

        # Create process groups
        self._create_groups()

    def _create_groups(self):
        """Create DP, PP, and TP process groups."""
        rank = dist.get_rank()

        # Find my coordinates
        coords = np.argwhere(self.mesh == rank)[0]
        self.dp_rank = coords[0]
        self.pp_rank = coords[1]
        self.tp_rank = coords[2]

        # TP group: same DP, same PP, vary TP
        tp_ranks = self.mesh[self.dp_rank, self.pp_rank, :].tolist()
        self.tp_group = dist.new_group(tp_ranks)

        # PP group: same DP, vary PP, same TP
        pp_ranks = self.mesh[self.dp_rank, :, self.tp_rank].tolist()
        self.pp_group = dist.new_group(pp_ranks)

        # DP group: vary DP, same PP, same TP
        dp_ranks = self.mesh[:, self.pp_rank, self.tp_rank].tolist()
        self.dp_group = dist.new_group(dp_ranks)

    def get_coordinates(self) -> Tuple[int, int, int]:
        """Return (dp_rank, pp_rank, tp_rank)."""
        return (self.dp_rank, self.pp_rank, self.tp_rank)

    def get_pipeline_neighbors(self) -> Tuple[Optional[int], Optional[int]]:
        """Return (prev_rank, next_rank) in pipeline."""
        prev_rank = None
        next_rank = None

        if self.pp_rank > 0:
            prev_rank = self.mesh[self.dp_rank, self.pp_rank - 1, self.tp_rank]

        if self.pp_rank < self.pp_size - 1:
            next_rank = self.mesh[self.dp_rank, self.pp_rank + 1, self.tp_rank]

        return (prev_rank, next_rank)
```

### The Group Correspondence Theorem (3D Case)

**Theorem**: In a 3D mesh of shape $(D, P, T)$, the process groups satisfy:

1. Each TP group has $T$ members
2. Each PP group has $P$ members
3. Each DP group has $D$ members
4. Total groups: $D \cdot P$ TP groups, $D \cdot T$ PP groups, $P \cdot T$ DP groups
5. Each device belongs to exactly one group of each type

**Proof**: Each TP group is determined by fixing $(d, p)$, giving $D \cdot P$ groups. Similarly for PP and DP. A device at $(d, p, t)$ belongs to:

- TP group $(d, p, *)$
- PP group $(d, *, t)$
- DP group $(*, p, t)$

Since coordinates are unique, each device is in exactly one group of each type. $\square$

## Communication Patterns

### Tensor Parallelism Communication

Within each TP group, every layer requires AllReduce or ReduceScatter/AllGather:

```
TP Group (ranks 0-3):
┌─────┬─────┬─────┬─────┐
│ G0  │ G1  │ G2  │ G3  │  Partial activations
└──┬──┴──┬──┴──┬──┴──┬──┘
   └─────┴─────┴─────┘
         AllReduce           High bandwidth via NVLink
   ┌─────┬─────┬─────┐
┌──┴──┬──┴──┬──┴──┬──┴──┐
│ Sum │ Sum │ Sum │ Sum │  Full activations
└─────┴─────┴─────┴─────┘
```

**Bandwidth**: $\frac{2(T-1)}{T} \cdot H \cdot B$ per layer.

### Pipeline Parallelism Communication

Between PP stages, point-to-point sends:

```
PP Group (ranks 0-7):
Stage 0 → Stage 1 → Stage 2 → ... → Stage 7
         ↑           ↑           ↑
       Send        Send        Send
    activations  activations  activations
```

**Bandwidth**: $H \cdot B$ per micro-batch boundary (much less than TP).

### Data Parallelism Communication

After backward pass, gradient synchronization:

```
DP Group (ranks across nodes):
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Grads 0 │     │ Grads 1 │ ... │ Grads 31│
└────┬────┘     └────┬────┘     └────┬────┘
     └───────────────┴───────────────┘
              AllReduce
     ┌───────────────┬───────────────┐
┌────┴────┐     ┌────┴────┐     ┌────┴────┐
│ Avg Grad│     │ Avg Grad│ ... │ Avg Grad│
└─────────┘     └─────────┘     └─────────┘
```

**Bandwidth**: $\frac{2(D-1)}{D} \cdot \frac{\text{Parameters}}{T \cdot P}$ per step.

## Memory Analysis

### Per-GPU Memory Breakdown

With 3D parallelism, memory is distributed:

**Model Parameters** (sharded by TP and PP):

$$M_{\text{params}} = \frac{\Psi \times 2}{T \times P}$$

**Optimizer States** (sharded by TP and PP):

$$M_{\text{optimizer}} = \frac{\Psi \times 8}{T \times P}$$

**Activations** (sharded by TP, multiplied by pipeline depth):

$$M_{\text{activations}} = \frac{B \times L_{\text{stage}} \times H}{T} \times k_{\text{buffer}}$$

Where $k_{\text{buffer}}$ accounts for in-flight micro-batches.

### Example: 175B Model on 1024 GPUs

Configuration: DP=32, PP=8, TP=4.

**Parameters per GPU**:

$$\frac{175\text{B} \times 2}{4 \times 8} = \frac{350\text{GB}}{32} = 10.9\text{ GB}$$

**Optimizer per GPU**:

$$\frac{175\text{B} \times 8}{4 \times 8} = \frac{1.4\text{TB}}{32} = 43.8\text{ GB}$$

**Activations per GPU** (with $B=32$ micro-batches in flight):

$$\frac{32 \times 12 \times 12288 \times 2}{4} \times 32 = \text{~24 GB}$$

**Total**: 10.9 + 43.8 + 24 ≈ **79 GB** (fits in 80GB A100).

## Performance Model

### Compute Time

Forward and backward pass time (per micro-batch):

$$T_{\text{compute}} = \frac{6 \times \Psi \times B_{\mu}}{P \times \text{FLOPs}_{\text{GPU}}}$$

Where $B_{\mu}$ is micro-batch size.

### Communication Time

**TP Communication** (per layer, both forward and backward):

$$T_{\text{TP}} = 4 \times L_{\text{stage}} \times \left(\alpha + \frac{2(T-1)}{T} \times \frac{H \times B_{\mu}}{\beta_{\text{NVLink}}}\right)$$

**PP Communication** (per micro-batch):

$$T_{\text{PP}} = 2 \times \left(\alpha + \frac{H \times B_{\mu}}{\beta_{\text{IB}}}\right)$$

**DP Communication** (once per step):

$$T_{\text{DP}} = \frac{2(D-1)}{D} \times \frac{\Psi/(T \times P)}{\beta_{\text{IB}}}$$

### Pipeline Efficiency

With 1F1B schedule and $M$ micro-batches:

$$\eta_{\text{pipeline}} = \frac{M}{M + P - 1}$$

### Total Step Time

$$T_{\text{step}} = \frac{M \times T_{\text{compute}} + M \times T_{\text{TP}}}{\eta_{\text{pipeline}}} + T_{\text{DP}}$$

The PP communication overlaps with compute in 1F1B, so doesn't add to critical path.

## Implementation

### 3D Parallel Trainer

```python
from dataclasses import dataclass
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.distributed as dist

@dataclass
class ParallelConfig:
    """Configuration for 3D parallelism."""
    dp_size: int
    pp_size: int
    tp_size: int
    num_microbatches: int
    gradient_accumulation_steps: int = 1

    @property
    def world_size(self) -> int:
        return self.dp_size * self.pp_size * self.tp_size

class ThreeDParallelTrainer:
    """Trainer implementing 3D parallelism."""

    def __init__(
        self,
        model_fn,  # Function to create model
        config: ParallelConfig,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] = None
    ):
        self.config = config

        # Initialize mesh
        self.mesh = ThreeDMesh(
            config.dp_size,
            config.pp_size,
            config.tp_size
        )

        # Create model shard for this rank
        self.model = self._create_model_shard(model_fn)

        # Create optimizer
        optimizer_kwargs = optimizer_kwargs or {'lr': 1e-4}
        self.optimizer = optimizer_class(
            self.model.parameters(),
            **optimizer_kwargs
        )

        # Setup communication buffers
        self._setup_buffers()

    def _create_model_shard(self, model_fn):
        """Create the local model shard based on PP and TP rank."""
        # Get which layers this rank handles
        pp_rank = self.mesh.pp_rank
        tp_rank = self.mesh.tp_rank

        # model_fn should return the appropriate shard
        model = model_fn(
            stage_id=pp_rank,
            num_stages=self.config.pp_size,
            tp_rank=tp_rank,
            tp_size=self.config.tp_size
        )

        return model.cuda()

    def _setup_buffers(self):
        """Allocate communication buffers."""
        # Activation buffers for pipeline (double-buffered)
        self.activation_buffers = [None, None]

        # Gradient buffers for DP sync
        self.gradient_buffer = None

    def train_step(self, data_iterator) -> float:
        """Execute one training step with 3D parallelism."""
        self.optimizer.zero_grad()

        num_microbatches = self.config.num_microbatches
        losses = []

        # 1F1B Schedule
        # Warmup: forward passes only
        for i in range(min(self.mesh.pp_size - 1, num_microbatches)):
            loss = self._forward_step(data_iterator, i)
            losses.append(loss)

        # Steady state: 1F1B
        for i in range(num_microbatches - self.mesh.pp_size + 1):
            # Forward
            loss = self._forward_step(
                data_iterator,
                i + self.mesh.pp_size - 1
            )
            losses.append(loss)

            # Backward for earlier microbatch
            self._backward_step(i)

        # Cooldown: backward passes only
        for i in range(self.mesh.pp_size - 1):
            self._backward_step(
                num_microbatches - self.mesh.pp_size + 1 + i
            )

        # DP gradient sync
        self._sync_gradients()

        # Optimizer step
        self.optimizer.step()

        return sum(losses) / len(losses)

    def _forward_step(self, data_iterator, micro_idx: int) -> float:
        """Forward pass for one microbatch."""
        # Get input
        if self.mesh.pp_rank == 0:
            # First stage: get from data
            batch = next(data_iterator)
            input_tensor = batch['input'].cuda()
        else:
            # Receive from previous stage
            input_tensor = self._recv_forward()

        # Forward through local layers (with TP)
        output = self.model(input_tensor)

        # Save for backward
        self._save_activation(micro_idx, input_tensor, output)

        if self.mesh.pp_rank == self.mesh.pp_size - 1:
            # Last stage: compute loss
            target = next(data_iterator)['target'].cuda()
            loss = self._compute_loss(output, target)
            return loss.item()
        else:
            # Send to next stage
            self._send_forward(output)
            return 0.0

    def _backward_step(self, micro_idx: int):
        """Backward pass for one microbatch."""
        # Get saved activation
        input_tensor, output = self._get_saved_activation(micro_idx)

        if self.mesh.pp_rank == self.mesh.pp_size - 1:
            # Last stage: gradient from loss
            output.backward(self._saved_loss_grad[micro_idx])
        else:
            # Receive gradient from next stage
            grad = self._recv_backward()
            output.backward(grad)

        if self.mesh.pp_rank > 0:
            # Send gradient to previous stage
            self._send_backward(input_tensor.grad)

    def _sync_gradients(self):
        """AllReduce gradients across DP group."""
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(
                    param.grad,
                    op=dist.ReduceOp.SUM,
                    group=self.mesh.dp_group
                )
                param.grad /= self.mesh.dp_size

    def _send_forward(self, tensor: torch.Tensor):
        """Send activation to next pipeline stage."""
        _, next_rank = self.mesh.get_pipeline_neighbors()
        dist.send(tensor, dst=next_rank, group=self.mesh.pp_group)

    def _recv_forward(self) -> torch.Tensor:
        """Receive activation from previous pipeline stage."""
        prev_rank, _ = self.mesh.get_pipeline_neighbors()
        tensor = torch.empty(self._get_activation_shape()).cuda()
        dist.recv(tensor, src=prev_rank, group=self.mesh.pp_group)
        return tensor

    def _send_backward(self, tensor: torch.Tensor):
        """Send gradient to previous pipeline stage."""
        prev_rank, _ = self.mesh.get_pipeline_neighbors()
        dist.send(tensor, dst=prev_rank, group=self.mesh.pp_group)

    def _recv_backward(self) -> torch.Tensor:
        """Receive gradient from next pipeline stage."""
        _, next_rank = self.mesh.get_pipeline_neighbors()
        tensor = torch.empty(self._get_activation_shape()).cuda()
        dist.recv(tensor, src=next_rank, group=self.mesh.pp_group)
        return tensor
```

### Tensor Parallel Layer Integration

Each layer must implement TP-aware forward:

```python
class TPLinear(nn.Module):
    """Linear layer with tensor parallelism."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_group: dist.ProcessGroup,
        tp_size: int,
        split_input: bool = False  # Column vs row parallel
    ):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.split_input = split_input

        if split_input:
            # Row parallel: split input dimension
            assert in_features % tp_size == 0
            local_in = in_features // tp_size
            local_out = out_features
        else:
            # Column parallel: split output dimension
            assert out_features % tp_size == 0
            local_in = in_features
            local_out = out_features // tp_size

        self.weight = nn.Parameter(
            torch.empty(local_out, local_in)
        )
        self.bias = nn.Parameter(torch.empty(local_out))

        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.nn.functional.linear(x, self.weight, self.bias)

        if self.split_input:
            # Row parallel: AllReduce output
            dist.all_reduce(output, op=dist.ReduceOp.SUM, group=self.tp_group)

        return output
```

## Choosing Dimensions

The art of 3D parallelism is choosing $(D, P, T)$ given:

- Total GPUs $N$
- Model size $M$
- Per-GPU memory $G$
- Network topology

### Dimension Selection Algorithm

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class HardwareSpec:
    """Hardware specifications."""
    total_gpus: int
    gpus_per_node: int
    gpu_memory_gb: float
    nvlink_bandwidth_gbps: float
    ib_bandwidth_gbps: float
    gpu_flops: float  # TFLOPs

@dataclass
class ModelSpec:
    """Model specifications."""
    num_params: int  # in billions
    hidden_dim: int
    num_layers: int
    vocab_size: int

def choose_3d_config(
    hw: HardwareSpec,
    model: ModelSpec,
    target_batch_size: int
) -> ParallelConfig:
    """Choose optimal 3D parallelism configuration."""

    # Calculate memory requirements
    param_memory_gb = model.num_params * 2  # FP16
    optimizer_memory_gb = model.num_params * 8  # Adam FP32

    # TP should fit in a node for NVLink
    max_tp = hw.gpus_per_node

    # Start with minimum TP that allows model to fit
    for tp in [1, 2, 4, 8]:
        if tp > max_tp:
            break

        # Try PP sizes
        for pp in [1, 2, 4, 8, 16, 32]:
            model_mem = (param_memory_gb + optimizer_memory_gb) / (tp * pp)

            # Leave room for activations (~40% of memory)
            if model_mem < hw.gpu_memory_gb * 0.6:
                # This (tp, pp) might work
                dp = hw.total_gpus // (tp * pp)

                if dp >= 1 and tp * pp * dp == hw.total_gpus:
                    # Validate activation memory
                    activation_mem = estimate_activation_memory(
                        model, target_batch_size // dp, tp, pp
                    )

                    total_mem = model_mem + activation_mem
                    if total_mem < hw.gpu_memory_gb * 0.9:
                        # Calculate efficiency
                        efficiency = calculate_efficiency(
                            hw, model, dp, pp, tp, target_batch_size
                        )

                        return ParallelConfig(
                            dp_size=dp,
                            pp_size=pp,
                            tp_size=tp,
                            num_microbatches=max(pp * 2, 8)
                        )

    raise ValueError("Cannot fit model with available resources")

def estimate_activation_memory(
    model: ModelSpec,
    batch_size: int,
    tp: int,
    pp: int
) -> float:
    """Estimate activation memory in GB."""
    layers_per_stage = model.num_layers // pp
    hidden = model.hidden_dim // tp

    # Per layer: hidden * batch * 2 (input + output) * 2 (bytes)
    per_layer = hidden * batch_size * 4 / 1e9

    # Pipeline buffers
    pipeline_factor = min(pp, 8)  # Microbatches in flight

    return per_layer * layers_per_stage * pipeline_factor

def calculate_efficiency(
    hw: HardwareSpec,
    model: ModelSpec,
    dp: int, pp: int, tp: int,
    batch_size: int
) -> float:
    """Estimate training efficiency (0-1)."""
    # Pipeline efficiency
    microbatches = max(pp * 2, 8)
    pipeline_eff = microbatches / (microbatches + pp - 1)

    # TP overhead (communication / compute ratio)
    layers_per_stage = model.num_layers // pp
    compute_per_layer = 6 * model.num_params * 1e9 / model.num_layers * batch_size / dp
    comm_per_layer = 4 * model.hidden_dim * batch_size / dp * 2  # AllReduce

    if tp > 1:
        # NVLink communication
        tp_time = comm_per_layer / (hw.nvlink_bandwidth_gbps * 1e9)
        compute_time = compute_per_layer / (hw.gpu_flops * 1e12)
        tp_eff = compute_time / (compute_time + tp_time)
    else:
        tp_eff = 1.0

    # DP overhead (gradient sync / step time)
    grad_sync = 2 * model.num_params * 1e9 / (tp * pp) / (hw.ib_bandwidth_gbps * 1e9)
    step_time = compute_time * layers_per_stage * microbatches / pipeline_eff
    dp_eff = step_time / (step_time + grad_sync)

    return pipeline_eff * tp_eff * dp_eff
```

### Rules of Thumb

1. **TP ≤ 8**: Keep within NVLink domain
2. **PP = layers / min_layers_per_stage**: Usually 4-16
3. **DP = remaining GPUs**: Maximize for throughput
4. **Microbatches ≥ 2 × PP**: Reduce bubble overhead

## Case Study: Megatron-LM Configuration

Megatron-LM trains large models with 3D parallelism:

### 175B GPT-3 Scale

| Configuration | Value |
|--------------|-------|
| Total GPUs | 1024 |
| TP | 8 |
| PP | 16 |
| DP | 8 |
| Microbatches | 32 |

**Memory per GPU**:

- Parameters: 175B × 2 / (8 × 16) = 2.7 GB
- Optimizer: 175B × 8 / (8 × 16) = 10.9 GB
- Activations: ~40 GB
- Total: ~54 GB (fits 80GB A100)

**Pipeline efficiency**:

$$\eta = \frac{32}{32 + 16 - 1} = \frac{32}{47} = 68\%$$

**TP communication** (per layer):

$$T_{\text{TP}} = 2 \times \left(\alpha + \frac{7}{8} \times \frac{12288 \times 2048 \times 2}{600 \times 10^9}\right) \approx 0.07\text{ ms}$$

**Overall MFU**: ~45-50% on A100 cluster.

### 530B MT-NLG Configuration

| Configuration | Value |
|--------------|-------|
| Total GPUs | 2240 |
| TP | 8 |
| PP | 35 |
| DP | 8 |
| Microbatches | 70 |

**Pipeline efficiency**:

$$\eta = \frac{70}{70 + 35 - 1} = \frac{70}{104} = 67\%$$

## Interleaved Pipeline Stages

Advanced optimization: each PP rank handles multiple non-contiguous stages.

### Standard vs Interleaved

**Standard** (virtual stages = 1):
```
Rank 0: [Layer 0-11]
Rank 1: [Layer 12-23]
Rank 2: [Layer 24-35]
Rank 3: [Layer 36-47]
```

**Interleaved** (virtual stages = 2):
```
Rank 0: [Layer 0-5] + [Layer 24-29]
Rank 1: [Layer 6-11] + [Layer 30-35]
Rank 2: [Layer 12-17] + [Layer 36-41]
Rank 3: [Layer 18-23] + [Layer 42-47]
```

### Interleaved Benefits

**Reduced bubble**:

$$\text{Bubble} = \frac{P - 1}{P + M - 1} \to \frac{P/v - 1}{P/v + M - 1}$$

With $v$ virtual stages per rank.

**Example**: PP=8, M=16
- Standard: bubble = 7/23 = 30%
- Interleaved (v=4): bubble = 1/17 = 6%

**Cost**: More communication (but overlapped with compute).

```python
class InterleavedPipelineSchedule:
    """Interleaved 1F1B schedule for reduced bubble."""

    def __init__(
        self,
        num_stages: int,
        virtual_stages: int,
        num_microbatches: int
    ):
        self.num_stages = num_stages
        self.virtual_stages = virtual_stages
        self.num_microbatches = num_microbatches

        # Each rank has multiple virtual stages
        self.stages_per_rank = virtual_stages
        self.total_virtual_stages = num_stages * virtual_stages

    def get_schedule(self, rank: int) -> List[Tuple[str, int, int]]:
        """
        Return schedule as list of (op_type, microbatch_id, virtual_stage_id).
        """
        schedule = []

        # Warmup phases
        warmup_steps = (self.num_stages - rank - 1) * self.virtual_stages

        for step in range(warmup_steps):
            virtual_stage = step % self.virtual_stages
            microbatch = step // self.virtual_stages
            schedule.append(('forward', microbatch, virtual_stage))

        # Steady state
        forward_mb = warmup_steps // self.virtual_stages
        backward_mb = 0

        steady_steps = self.num_microbatches - warmup_steps // self.virtual_stages

        for _ in range(steady_steps * self.virtual_stages):
            # Alternate forward and backward across virtual stages
            schedule.append(('forward', forward_mb, forward_mb % self.virtual_stages))
            forward_mb += 1

            schedule.append(('backward', backward_mb, backward_mb % self.virtual_stages))
            backward_mb += 1

        # Cooldown
        for step in range(warmup_steps):
            virtual_stage = step % self.virtual_stages
            schedule.append(('backward', backward_mb, virtual_stage))
            if (step + 1) % self.virtual_stages == 0:
                backward_mb += 1

        return schedule
```

## ZeRO Integration

Combine 3D parallelism with ZeRO for further memory reduction:

### ZeRO-1 with 3D Parallelism

Shard optimizer states across DP dimension:

```
Before ZeRO-1:
Each DP replica: full optimizer states (8 bytes/param)

After ZeRO-1:
Each DP replica: 1/D of optimizer states

Memory savings: (D-1)/D of optimizer memory
```

**Communication change**: Optimizer step requires AllGather of parameters.

### Memory Stack

```
┌────────────────────────────────────────┐
│           Original Model               │
│          175B params × 10B             │
│              = 1.75 TB                 │
├────────────────────────────────────────┤
│     After TP/PP (8×16 = 128-way)       │
│           1.75TB / 128                 │
│              = 13.7 GB                 │
├────────────────────────────────────────┤
│     After ZeRO-1 (DP=8)                │
│      Params: 2.7 GB (unchanged)        │
│      Optim: 10.9GB / 8 = 1.4 GB        │
│           Total: 4.1 GB                │
└────────────────────────────────────────┘
```

## Exercises

1. **Configuration design**: You have 256 A100 GPUs (32 nodes × 8 GPUs) and want to train a 40B parameter model. Design a 3D parallelism configuration. Calculate memory per GPU and expected pipeline efficiency.

??? success "Solution"
    **Model characteristics (40B):**

    $$M_{total} = 16\Psi = 16 \times 40 \times 10^9 = 640 \text{ GB}$$

    **Configuration design:**

    | Strategy | Value | Rationale |
    |----------|-------|-----------|
    | TP | 8 | Full intra-node (NVLink) |
    | PP | 8 | 4 nodes per pipeline |
    | DP | 4 | 256 / (8 × 8) = 4 replicas |

    **Verify:** $8 \times 8 \times 4 = 256$ ✓

    **Memory per GPU:**

    | Component | Per-GPU Memory |
    |-----------|----------------|
    | Parameters | $\frac{2 \times 40B}{TP \times PP} = \frac{80}{64} = 1.25$ GB |
    | Gradients | 1.25 GB |
    | Optimizer | $\frac{12 \times 40B}{TP \times PP} = 7.5$ GB |
    | **Subtotal static** | **10 GB** |

    With ZeRO-1 on DP=4:

    | Component | Per-GPU Memory |
    |-----------|----------------|
    | Parameters | 1.25 GB |
    | Gradients | 1.25 GB |
    | Optimizer | $\frac{7.5}{4} = 1.875$ GB |
    | **Subtotal static** | **4.375 GB** |

    **Activation memory estimate:**

    Assuming B=4 per DP, S=4096, H=8192, 40 layers per stage:

    $$M_{act} \approx \frac{40}{8} \times B \times S \times H \times 34 / TP$$

    $$M_{act} \approx 5 \times 4 \times 4096 \times 8192 \times 34 / 8 \approx 35 \text{ GB (with checkpointing)}$$

    **Total per GPU:** ~40 GB — fits in A100 80GB ✓

    **Pipeline efficiency:**

    With 32 micro-batches (M=32) and PP=8:

    $$\eta_{PP} = \frac{M}{M + P - 1} = \frac{32}{32 + 8 - 1} = \frac{32}{39} = \boxed{82\%}$$

    | Metric | Value |
    |--------|-------|
    | Bubble fraction | 18% |
    | Expected MFU | ~40-45% (accounting for all overheads) |

2. **Communication analysis**: For configuration DP=8, PP=4, TP=4 training a model with hidden_dim=8192 and batch_size=512:

   - Calculate TP communication volume per layer
   - Calculate PP communication volume per micro-batch
   - Calculate DP communication volume per step
   - Which is the bottleneck?

??? success "Solution"
    **Configuration:** DP=8, PP=4, TP=4 → Total GPUs = 128

    **Given:** H=8192, batch_size=512, assume S=2048

    **1. TP Communication (per layer):**

    Each layer has 2 AllReduce operations (after column-parallel and row-parallel):

    $$V_{TP}^{layer} = 2 \times B_{micro} \times S \times H \times 2 \text{ bytes}$$

    Micro-batch size: $B_{micro} = \frac{512}{DP \times M} = \frac{512}{8 \times 32} = 2$ (assuming M=32)

    $$V_{TP}^{layer} = 2 \times 2 \times 2048 \times 8192 \times 2 = 134 \text{ MB}$$

    Effective volume (ring AllReduce): $\frac{TP-1}{TP} \times 134 = 100.5$ MB

    **2. PP Communication (per micro-batch):**

    Activation transfer between stages:

    $$V_{PP}^{micro} = B_{micro} \times S \times H \times 2 = 2 \times 2048 \times 8192 \times 2 = 67 \text{ MB}$$

    Point-to-point, so full volume counts.

    **3. DP Communication (per step):**

    AllReduce gradients across DP replicas:

    Assume 30B model split across PP×TP:

    $$V_{DP}^{step} = 2\Psi = 2 \times 30 \times 10^9 / (PP \times TP) = \frac{60 \text{ GB}}{16} = 3.75 \text{ GB}$$

    Effective: $\frac{DP-1}{DP} \times 3.75 = 3.28$ GB

    **4. Time analysis (assuming bandwidths):**

    | Communication | Volume | Bandwidth | Time |
    |---------------|--------|-----------|------|
    | TP (per layer) | 100 MB | 450 GB/s (NVLink) | 0.22 ms |
    | TP (total, 80 layers) | 8 GB | 450 GB/s | 18 ms |
    | PP (total, 32 micro-batches) | 2.1 GB | 50 GB/s (IB) | 43 ms |
    | DP (per step) | 3.28 GB | 50 GB/s | 66 ms |

    $$\boxed{\text{DP AllReduce is the bottleneck (66 ms)}}$$

    **Mitigation:** Overlap DP AllReduce with backward pass computation.

3. **Interleaving trade-off**: Compare bubble fraction for PP=16, M=32 with:

   - No interleaving (v=1)
   - v=2 interleaving
   - v=4 interleaving
   What's the communication overhead for each?

??? success "Solution"
    **Given:** PP=16 stages, M=32 micro-batches

    **Bubble fraction formula:**

    Without interleaving (1F1B):

    $$\text{Bubble} = \frac{P - 1}{M + P - 1}$$

    With interleaving factor $v$:

    $$\text{Bubble}_{interleaved} = \frac{P - 1}{v \times M + P - 1}$$

    **Calculations:**

    | Interleaving | Bubble Formula | Bubble Fraction |
    |--------------|----------------|-----------------|
    | v=1 | $\frac{15}{32+15}$ | $\frac{15}{47} = 31.9\%$ |
    | v=2 | $\frac{15}{64+15}$ | $\frac{15}{79} = 19.0\%$ |
    | v=4 | $\frac{15}{128+15}$ | $\frac{15}{143} = \boxed{10.5\%}$ |

    **Communication overhead:**

    With interleaving factor $v$, each micro-batch crosses $v$ times more stage boundaries:

    $$\text{Comm}_{interleaved} = v \times \text{Comm}_{base}$$

    | Interleaving | Comm Multiplier | Bubble Reduction | Net Benefit? |
    |--------------|-----------------|------------------|--------------|
    | v=1 | 1× | baseline | baseline |
    | v=2 | 2× | 12.9% less bubble | Yes, if comm < bubble |
    | v=4 | 4× | 21.4% less bubble | Diminishing returns |

    **Break-even analysis:**

    Let $T_c$ = compute time, $T_{comm}$ = base PP communication time.

    Interleaving benefits when:

    $$\Delta\text{Bubble} \times T_c > (v-1) \times T_{comm}$$

    For v=2:

    $$0.129 \times T_c > 1 \times T_{comm}$$

    If $T_c = 1000$ ms and $T_{comm} = 50$ ms:

    $$129 \text{ ms} > 50 \text{ ms} \checkmark$$

    **Summary:**

    | v | Bubble | Comm Overhead | Recommended When |
    |---|--------|---------------|------------------|
    | 1 | 31.9% | 1× | Limited bandwidth |
    | 2 | 19.0% | 2× | Typical (balanced) |
    | 4 | 10.5% | 4× | High bandwidth links |

4. **Scaling efficiency**: A 3D parallel configuration achieves 50% MFU on 512 GPUs. When scaling to 2048 GPUs (4× DP), predict the new MFU. What are the bottlenecks?

??? success "Solution"
    **Baseline:** 50% MFU at 512 GPUs

    **Scaling scenario:** Only increase DP from D to 4D (keep TP, PP constant)

    **Analysis of overheads:**

    1. **Compute efficiency** (unchanged):
       - Same batch per GPU → same compute density
       - Kernel efficiency unchanged

    2. **TP communication** (unchanged):
       - TP groups unchanged
       - Same overhead per forward/backward

    3. **PP efficiency** (unchanged):
       - Same pipeline depth
       - Same bubble fraction

    4. **DP communication** (increases):
       - AllReduce now across 4× more GPUs
       - Ring AllReduce: $T_{DP} = \frac{D-1}{D} \times \frac{2\Psi}{B}$

    **DP scaling impact:**

    At 512 GPUs with DP=D:

    $$T_{DP}^{512} = \frac{D-1}{D} \times \frac{2\Psi}{B}$$

    At 2048 GPUs with DP=4D:

    $$T_{DP}^{2048} = \frac{4D-1}{4D} \times \frac{2\Psi}{B}$$

    Ratio: $\frac{T_{DP}^{2048}}{T_{DP}^{512}} = \frac{(4D-1)/4D}{(D-1)/D} = \frac{4D-1}{4(D-1)} \approx 1.0$ for large D

    **But** the DP AllReduce time stays the same per GPU, while compute per GPU stays the same.

    **Real bottleneck: Batch size scaling**

    To maintain efficiency, global batch must scale 4×:

    $$B_{global}^{new} = 4 \times B_{global}^{old}$$

    If we **don't** scale batch:
    - Each GPU does 1/4 the work
    - Communication stays same
    - MFU drops significantly

    **Predicted MFU:**

    | Scenario | Global Batch | Per-GPU Work | MFU |
    |----------|--------------|--------------|-----|
    | Keep batch constant | 1× | 0.25× | ~20% |
    | Scale batch 2× | 2× | 0.5× | ~35% |
    | Scale batch 4× | 4× | 1× | $\boxed{\sim 48\%}$ |

    **Bottlenecks at 2048 GPUs:**

    1. **DP AllReduce latency** — more hops in larger rings
    2. **Network congestion** — 4× more inter-node traffic
    3. **Batch size limits** — may hit learning rate stability issues
    4. **Memory bandwidth** — activation reloading at larger scale

5. **ZeRO integration**: For DP=32, PP=8, TP=4 with 175B parameters:

   - Calculate optimizer memory per GPU without ZeRO
   - Calculate optimizer memory per GPU with ZeRO-1
   - What's the activation memory budget freed up?

??? success "Solution"
    **Configuration:** DP=32, PP=8, TP=4 → Total GPUs = 1024

    **Model partitioning:**

    Parameters per GPU (before ZeRO):

    $$\Psi_{GPU} = \frac{175B}{PP \times TP} = \frac{175B}{32} = 5.47B \text{ params}$$

    **Without ZeRO:**

    | Component | Size | Memory |
    |-----------|------|--------|
    | Parameters (fp16) | $5.47B \times 2$ | 10.9 GB |
    | Gradients (fp16) | $5.47B \times 2$ | 10.9 GB |
    | Optimizer states (fp32) | $5.47B \times 12$ | 65.6 GB |
    | **Total** | | **87.4 GB** |

    $$M_{opt}^{no\_ZeRO} = \boxed{65.6 \text{ GB}}$$

    **With ZeRO-1 (optimizer state sharding across DP):**

    ZeRO-1 shards optimizer states across DP=32:

    $$M_{opt}^{ZeRO1} = \frac{65.6 \text{ GB}}{32} = \boxed{2.05 \text{ GB}}$$

    | Component | Without ZeRO | With ZeRO-1 |
    |-----------|--------------|-------------|
    | Parameters | 10.9 GB | 10.9 GB |
    | Gradients | 10.9 GB | 10.9 GB |
    | Optimizer | 65.6 GB | 2.05 GB |
    | **Total Static** | **87.4 GB** | **23.85 GB** |

    **Activation memory budget freed:**

    Memory saved:

    $$\Delta M = 87.4 - 23.85 = 63.55 \text{ GB}$$

    On an 80GB A100:

    | Metric | Without ZeRO | With ZeRO-1 |
    |--------|--------------|-------------|
    | Static memory | 87.4 GB | 23.85 GB |
    | Available for activations | -7.4 GB (OOM!) | 56.15 GB |
    | **Feasible?** | No | Yes |

    $$\boxed{\text{ZeRO-1 frees } \sim 64 \text{ GB for activations}}$$

    **Note:** ZeRO-1 adds AllGather for optimizer states during the update step, but this is a one-time cost per step and can be overlapped.

6. **Alternative ordering**: The standard ordering is (DP, PP, TP). What happens if you use (DP, TP, PP)? Analyze the communication pattern changes.

??? success "Solution"
    **Standard ordering (DP, PP, TP):**

    ```
    Mesh shape: (DP, PP, TP)
    Innermost (TP): Consecutive ranks → intra-node (NVLink)
    Middle (PP): Strides by TP → may cross nodes
    Outermost (DP): Strides by PP×TP → crosses nodes
    ```

    **Alternative ordering (DP, TP, PP):**

    ```
    Mesh shape: (DP, TP, PP)
    Innermost (PP): Consecutive ranks
    Middle (TP): Strides by PP
    Outermost (DP): Strides by TP×PP
    ```

    **Example with 64 GPUs (DP=2, TP=4, PP=8):**

    | Ordering | TP Group (rank 0) | PP Chain (TP=0, DP=0) |
    |----------|-------------------|------------------------|
    | (DP, PP, TP) | {0,1,2,3} | {0,4,8,12,16,20,24,28} |
    | (DP, TP, PP) | {0,8,16,24} | {0,1,2,3,4,5,6,7} |

    **Communication impact:**

    | Collective | Standard (DP,PP,TP) | Alternative (DP,TP,PP) |
    |------------|---------------------|------------------------|
    | TP AllReduce | Consecutive → NVLink | Strided → may cross nodes |
    | PP Send/Recv | Strided → cross-node | Consecutive → may be NVLink |
    | DP AllReduce | Large stride → cross-node | Large stride → cross-node |

    **Analysis:**

    **Standard ordering is preferred because:**

    1. **TP frequency**: 2× AllReduce per layer × 80+ layers = 160+ collectives/step
    2. **PP frequency**: 1× Send/Recv per micro-batch × 32 = 32 transfers/step
    3. **DP frequency**: 1× AllReduce per step

    TP needs highest bandwidth → must be innermost (NVLink).

    **Alternative ordering consequences:**

    | Issue | Impact |
    |-------|--------|
    | TP over cross-node | 18× bandwidth reduction (900 → 50 GB/s) |
    | TP latency | Adds network hops |
    | PP consecutive | Slight improvement for PP |

    **Quantitative example:**

    TP AllReduce time (100MB per operation):

    | Ordering | Bandwidth | Time per AllReduce |
    |----------|-----------|-------------------|
    | Standard (NVLink) | 900 GB/s | 0.11 ms |
    | Alternative (IB) | 50 GB/s | 2.0 ms |

    Per step (160 AllReduces):

    | Ordering | TP Time |
    |----------|---------|
    | Standard | 18 ms |
    | Alternative | 320 ms |

    $$\boxed{\text{Alternative ordering: 18× slower TP communication}}$$

    **When might alternative work?**

    - Very deep pipelines where PP dominates
    - Extremely small TP groups
    - Custom topologies where NVLink spans differently

## Key Takeaways

1. **No single strategy scales**: DP wastes memory, TP has communication overhead, PP has bubbles.

2. **3D = DP × PP × TP**: Compose strategies at their optimal scales.

3. **TP innermost**: Use NVLink bandwidth within nodes.

4. **Pipeline efficiency**: Use many micro-batches and interleaving.

5. **Configuration is an optimization problem**: Balance memory, compute, and communication.

6. **ZeRO compounds benefits**: Shard optimizer states across DP for additional memory savings.

7. **Megatron-LM patterns work**: TP=8, PP=8-16, DP for remaining GPUs is proven at 100B+ scale.
