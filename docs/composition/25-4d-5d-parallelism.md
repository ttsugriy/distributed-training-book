---
title: "4D and 5D Parallelism"
subtitle: "Context Parallelism and Expert Parallelism in the Mix"
---

<div class="chapter-opener" markdown>
As context windows grow to millions of tokens and sparse Mixture-of-Experts models reach trillions of parameters, 3D parallelism hits new walls. The fourth dimension—context parallelism—handles long sequences. The fifth—expert parallelism—handles sparse capacity. Together, they enable the largest models in existence.
</div>

<div class="investigation-question" markdown>
**The Question**: You're training a 1T parameter MoE model with 128K context length on 16,384 GPUs. 3D parallelism can handle neither the sequence memory nor the expert routing. What additional dimensions of parallelism do you need, and how do they compose with the existing three?
</div>

## Beyond 3D: New Constraints

### The Context Length Problem

With 3D parallelism (DP × TP × PP), activation memory scales as:

$$M_{\text{act}} = \frac{B \times S \times H}{T} \times L_{\text{stage}}$$

Where $S$ is sequence length. As $S \to 128K$ or beyond:

| Sequence Length | Activation Memory (per layer, TP=8) |
|-----------------|-------------------------------------|
| 2K | 0.4 GB |
| 8K | 1.6 GB |
| 32K | 6.4 GB |
| 128K | 25.6 GB |
| 1M | 200 GB |

Even with TP=8 and PP=16, a 128K sequence exhausts GPU memory on activations alone.

### The Expert Scaling Problem

Mixture-of-Experts models have different memory characteristics:

**Dense model**: All parameters active for all tokens.

**MoE model**: Only $k$ of $E$ experts active per token, but all must be stored.

$$M_{\text{experts}} = \frac{E \times \text{ExpertSize}}{T \times P}$$

For a 1T MoE with 128 experts:
$$M_{\text{experts}} = \frac{128 \times 8\text{B} \times 2}{8 \times 16} = 16\text{ GB just for expert parameters}$$

Plus routing creates dynamic load imbalance.

## The Fourth Dimension: Context Parallelism

Context Parallelism (CP) partitions the sequence dimension across devices.

### CP Operation

```
Without CP:
GPU 0: [Token 0, Token 1, ..., Token 128K]  ← OOM

With CP=4:
GPU 0: [Token 0, ..., Token 32K]
GPU 1: [Token 32K, ..., Token 64K]
GPU 2: [Token 64K, ..., Token 96K]
GPU 3: [Token 96K, ..., Token 128K]
```

Each GPU processes 1/CP of the sequence.

### CP Communication Pattern

Attention requires all tokens to attend to all tokens:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

With sequence partitioning, this requires AllGather of K and V:

```
Forward pass:
┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│ Q0  │  │ Q1  │  │ Q2  │  │ Q3  │  Local queries
└──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
   │        │        │        │
   └────────┴────────┴────────┘
          AllGather K,V
   ┌────────┬────────┬────────┐
   │        │        │        │
┌──┴──┐  ┌──┴──┐  ┌──┴──┐  ┌──┴──┐
│K0-3 │  │K0-3 │  │K0-3 │  │K0-3 │  Full K,V on each
│V0-3 │  │V0-3 │  │V0-3 │  │V0-3 │
└──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
   │        │        │        │
┌──┴──┐  ┌──┴──┐  ┌──┴──┐  ┌──┴──┐
│Attn │  │Attn │  │Attn │  │Attn │  Local attention
│ O0  │  │ O1  │  │ O2  │  │ O3  │
└─────┘  └─────┘  └─────┘  └─────┘
```

### Ring Attention Optimization

Instead of AllGather (memory-intensive), use ring attention:

```python
from typing import Optional
import torch
import torch.distributed as dist

class RingAttention:
    """Memory-efficient ring attention for context parallelism."""

    def __init__(
        self,
        cp_group: dist.ProcessGroup,
        cp_size: int,
        causal: bool = True
    ):
        self.cp_group = cp_group
        self.cp_size = cp_size
        self.causal = causal
        self.cp_rank = dist.get_rank(cp_group)

    def forward(
        self,
        q: torch.Tensor,  # [batch, seq_local, heads, dim]
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """Ring attention forward pass."""
        batch, seq_local, heads, dim = q.shape

        # Initialize output accumulator
        output = torch.zeros_like(q)
        normalizer = torch.zeros(batch, seq_local, heads, 1, device=q.device)

        # Ring buffers
        k_recv = torch.empty_like(k)
        v_recv = torch.empty_like(v)

        k_send = k.contiguous()
        v_send = v.contiguous()

        for step in range(self.cp_size):
            # Compute which KV chunk we have
            kv_rank = (self.cp_rank - step) % self.cp_size

            # Compute attention scores for this KV chunk
            scores = torch.einsum('bshd,bShd->bshS', q, k_send)
            scores = scores / (dim ** 0.5)

            # Apply causal mask if needed
            if self.causal:
                scores = self._apply_causal_mask(
                    scores, step, kv_rank
                )

            # Online softmax update
            max_scores = scores.max(dim=-1, keepdim=True).values
            exp_scores = torch.exp(scores - max_scores)
            sum_exp = exp_scores.sum(dim=-1, keepdim=True)

            # Update output with this chunk's contribution
            chunk_output = torch.einsum('bshS,bShd->bshd', exp_scores, v_send)

            # Numerically stable accumulation
            output, normalizer = self._online_softmax_update(
                output, normalizer, chunk_output, sum_exp, max_scores
            )

            # Ring communication (except last step)
            if step < self.cp_size - 1:
                # Send to next, receive from prev
                next_rank = (self.cp_rank + 1) % self.cp_size
                prev_rank = (self.cp_rank - 1) % self.cp_size

                send_ops = [
                    dist.P2POp(dist.isend, k_send, next_rank, self.cp_group),
                    dist.P2POp(dist.isend, v_send, next_rank, self.cp_group),
                ]
                recv_ops = [
                    dist.P2POp(dist.irecv, k_recv, prev_rank, self.cp_group),
                    dist.P2POp(dist.irecv, v_recv, prev_rank, self.cp_group),
                ]

                reqs = dist.batch_isend_irecv(send_ops + recv_ops)
                for req in reqs:
                    req.wait()

                # Swap buffers
                k_send, k_recv = k_recv, k_send
                v_send, v_recv = v_recv, v_send

        # Final normalization
        output = output / normalizer

        return output

    def _apply_causal_mask(
        self,
        scores: torch.Tensor,
        step: int,
        kv_rank: int
    ) -> torch.Tensor:
        """Apply causal masking for ring attention."""
        batch, seq_q, heads, seq_kv = scores.shape

        # Query positions: [cp_rank * seq_local, (cp_rank+1) * seq_local)
        # KV positions: [kv_rank * seq_local, (kv_rank+1) * seq_local)

        if kv_rank > self.cp_rank:
            # All KV positions are in the future - mask all
            return torch.full_like(scores, float('-inf'))
        elif kv_rank < self.cp_rank:
            # All KV positions are in the past - no masking
            return scores
        else:
            # Same chunk - standard causal mask
            mask = torch.triu(
                torch.ones(seq_q, seq_kv, device=scores.device),
                diagonal=1
            ).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            return scores

    def _online_softmax_update(
        self,
        output: torch.Tensor,
        normalizer: torch.Tensor,
        chunk_output: torch.Tensor,
        chunk_sum: torch.Tensor,
        chunk_max: torch.Tensor
    ) -> tuple:
        """Online softmax accumulation for numerical stability."""
        # This implements the online softmax algorithm
        # for accumulating attention across chunks

        # First chunk
        if normalizer.sum() == 0:
            return chunk_output, chunk_sum

        # Update normalizer and output with proper scaling
        new_normalizer = normalizer + chunk_sum
        scale_old = normalizer / new_normalizer
        scale_new = chunk_sum / new_normalizer

        new_output = output * scale_old + chunk_output * scale_new

        return new_output, new_normalizer
```

### CP Memory Savings

Memory reduction from context parallelism:

| Component | Without CP | With CP=C |
|-----------|-----------|-----------|
| Activations | $B \times S \times H$ | $B \times S/C \times H$ |
| KV Cache | $2 \times L \times B \times S \times H$ | $2 \times L \times B \times S \times H$ (ring) |
| Peak Memory | $O(S^2)$ for attention | $O(S \times S/C)$ with ring |

**Key insight**: Ring attention avoids the $O(S^2)$ memory for attention matrices.

## 4D Parallelism: DP × TP × PP × CP

### Mesh Configuration

```
4D Mesh: shape = (D, P, T, C)

Example: 1024 GPUs for 100B model with 128K context
- DP = 4   (data parallel replicas)
- PP = 8   (pipeline stages)
- TP = 8   (tensor parallel)
- CP = 4   (context parallel)

Total: 4 × 8 × 8 × 4 = 1024
```

### Process Group Structure

```python
from dataclasses import dataclass
from typing import Tuple, Optional
import torch.distributed as dist
import numpy as np

@dataclass
class FourDConfig:
    """Configuration for 4D parallelism."""
    dp_size: int
    pp_size: int
    tp_size: int
    cp_size: int

    @property
    def world_size(self) -> int:
        return self.dp_size * self.pp_size * self.tp_size * self.cp_size

class FourDMesh:
    """Device mesh for 4D parallelism: DP × PP × TP × CP."""

    def __init__(self, config: FourDConfig):
        self.config = config

        world_size = config.world_size
        devices = np.arange(world_size).reshape(
            config.dp_size,
            config.pp_size,
            config.tp_size,
            config.cp_size
        )

        self.mesh = devices

        # Get my coordinates
        rank = dist.get_rank()
        coords = np.argwhere(devices == rank)[0]
        self.dp_rank = coords[0]
        self.pp_rank = coords[1]
        self.tp_rank = coords[2]
        self.cp_rank = coords[3]

        # Create process groups
        self._create_groups()

    def _create_groups(self):
        """Create all necessary process groups."""
        # DP group: vary DP, fix others
        dp_ranks = self.mesh[:, self.pp_rank, self.tp_rank, self.cp_rank].tolist()
        self.dp_group = dist.new_group(dp_ranks)

        # PP group: fix DP, vary PP, fix TP, fix CP
        pp_ranks = self.mesh[self.dp_rank, :, self.tp_rank, self.cp_rank].tolist()
        self.pp_group = dist.new_group(pp_ranks)

        # TP group: fix DP, fix PP, vary TP, fix CP
        tp_ranks = self.mesh[self.dp_rank, self.pp_rank, :, self.cp_rank].tolist()
        self.tp_group = dist.new_group(tp_ranks)

        # CP group: fix DP, fix PP, fix TP, vary CP
        cp_ranks = self.mesh[self.dp_rank, self.pp_rank, self.tp_rank, :].tolist()
        self.cp_group = dist.new_group(cp_ranks)

        # TP-CP group (for some fused operations): vary both TP and CP
        tp_cp_ranks = self.mesh[self.dp_rank, self.pp_rank, :, :].flatten().tolist()
        self.tp_cp_group = dist.new_group(tp_cp_ranks)

    def get_ranks(self) -> Tuple[int, int, int, int]:
        """Return (dp_rank, pp_rank, tp_rank, cp_rank)."""
        return (self.dp_rank, self.pp_rank, self.tp_rank, self.cp_rank)
```

### 4D Communication Analysis

Communication patterns in 4D parallelism:

| Dimension | Operation | Frequency | Volume |
|-----------|-----------|-----------|--------|
| TP | AllReduce | Every layer | $2 \times \frac{T-1}{T} \times H \times B \times S/C$ |
| CP | Ring P2P | Every attention | $2 \times B \times S/C \times H$ (K+V) |
| PP | P2P Send/Recv | Every micro-batch | $B \times S/C \times H$ |
| DP | AllReduce | Every step | $2 \times \frac{D-1}{D} \times \text{Params}/(T \times P)$ |

### Dimension Ordering for Hardware

Map dimensions to hardware topology:

```
Optimal for 8-GPU DGX nodes:
┌─────────────────────────────────┐
│         Within Node             │
│  TP (NVLink) + CP (NVLink)      │
│         8 GPUs                  │
├─────────────────────────────────┤
│         Across Nodes            │
│  PP (IB) + DP (IB)              │
│        Many nodes               │
└─────────────────────────────────┘

Recommended: TP × CP ≤ 8 (within NVLink domain)
```

## The Fifth Dimension: Expert Parallelism

Expert Parallelism (EP) distributes MoE experts across devices.

### MoE Memory Breakdown

For a Mixture-of-Experts layer:

```
Standard MoE Layer:

- Router: H → E weights (small)
- Experts: E × (H → 4H → H) (large)
- Typically E = 64-256 experts
```

Memory per expert:
$$M_{\text{expert}} = 2 \times H \times 4H \times 2 \text{ bytes (FP16)} = 16H^2$$

For H=12288, E=128:
$$M_{\text{all\_experts}} = 128 \times 16 \times 12288^2 \approx 310\text{ GB}$$

### EP Operation

```
Without EP (all experts on each GPU):
GPU 0: [Expert 0-127]  ← 310 GB just for experts

With EP=8:
GPU 0: [Expert 0-15]    ← 39 GB
GPU 1: [Expert 16-31]   ← 39 GB
...
GPU 7: [Expert 112-127] ← 39 GB
```

### EP Communication: AlltoAll

MoE routing requires AlltoAll communication:

```
Token Routing with EP=4:
┌──────────────────────────────────────────┐
│ GPU 0: Tokens [T0, T1, T2, T3]           │
│ Router decides:                          │
│   T0 → Expert 2 (GPU 0)                  │
│   T1 → Expert 5 (GPU 1)                  │
│   T2 → Expert 3 (GPU 0)                  │
│   T3 → Expert 9 (GPU 2)                  │
└──────────────────────────────────────────┘
              │
              ▼ AlltoAll
┌──────────────────────────────────────────┐
│ GPU 0: [T0, T2] → Experts 0-3            │
│ GPU 1: [T1]     → Experts 4-7            │
│ GPU 2: [T3]     → Experts 8-11           │
│ GPU 3: []       → Experts 12-15          │
└──────────────────────────────────────────┘
              │
              ▼ Expert Computation
              │
              ▼ AlltoAll (reverse)
┌──────────────────────────────────────────┐
│ GPU 0: [T0, T1, T2, T3] (results)        │
└──────────────────────────────────────────┘
```

### EP Implementation

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Tuple

class ExpertParallelMoE(nn.Module):
    """MoE layer with expert parallelism."""

    def __init__(
        self,
        hidden_dim: int,
        expert_dim: int,
        num_experts: int,
        top_k: int,
        ep_group: dist.ProcessGroup,
        ep_size: int
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_group = ep_group
        self.ep_size = ep_size
        self.ep_rank = dist.get_rank(ep_group)

        # Number of local experts
        assert num_experts % ep_size == 0
        self.num_local_experts = num_experts // ep_size

        # Router
        self.router = nn.Linear(hidden_dim, num_experts)

        # Local experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, hidden_dim)
            )
            for _ in range(self.num_local_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with expert parallelism.

        Args:
            x: [batch * seq, hidden_dim]

        Returns:
            [batch * seq, hidden_dim]
        """
        batch_seq = x.shape[0]

        # Route tokens
        router_logits = self.router(x)  # [batch*seq, num_experts]
        routing_weights, selected_experts = self._route(router_logits)

        # Prepare for AlltoAll
        # Group tokens by destination EP rank
        tokens_per_ep, send_counts = self._prepare_alltoall(
            x, selected_experts
        )

        # AlltoAll: send tokens to their expert's EP rank
        received_tokens, recv_counts = self._alltoall(
            tokens_per_ep, send_counts
        )

        # Process through local experts
        expert_outputs = self._process_local_experts(
            received_tokens, recv_counts
        )

        # AlltoAll: return results
        final_outputs, _ = self._alltoall_reverse(
            expert_outputs, recv_counts, send_counts
        )

        # Combine with routing weights
        output = self._combine_outputs(
            final_outputs, routing_weights, selected_experts
        )

        return output

    def _route(
        self,
        logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k routing."""
        routing_weights = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(routing_weights, self.top_k, dim=-1)

        # Normalize weights
        weights = weights / weights.sum(dim=-1, keepdim=True)

        return weights, indices

    def _prepare_alltoall(
        self,
        tokens: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> Tuple[torch.Tensor, list]:
        """Prepare tokens for AlltoAll dispatch."""
        batch_seq = tokens.shape[0]

        # Count tokens going to each EP rank
        send_counts = [0] * self.ep_size

        # Flatten expert selection
        flat_experts = selected_experts.flatten()

        for expert_id in flat_experts:
            ep_rank = expert_id.item() // self.num_local_experts
            send_counts[ep_rank] += 1

        # Sort tokens by destination EP rank
        # (In practice, use more efficient GPU-based sorting)
        sorted_indices = torch.argsort(
            flat_experts // self.num_local_experts
        )

        # Expand tokens for top-k
        expanded_tokens = tokens.unsqueeze(1).expand(-1, self.top_k, -1)
        expanded_tokens = expanded_tokens.reshape(-1, self.hidden_dim)

        sorted_tokens = expanded_tokens[sorted_indices]

        return sorted_tokens, send_counts

    def _alltoall(
        self,
        send_data: torch.Tensor,
        send_counts: list
    ) -> Tuple[torch.Tensor, list]:
        """AlltoAll communication."""
        # Exchange counts
        recv_counts = [0] * self.ep_size
        dist.all_to_all_single(
            torch.tensor(recv_counts, device=send_data.device),
            torch.tensor(send_counts, device=send_data.device),
            group=self.ep_group
        )
        recv_counts = recv_counts

        # Exchange data
        total_recv = sum(recv_counts)
        recv_data = torch.empty(
            total_recv, self.hidden_dim,
            device=send_data.device, dtype=send_data.dtype
        )

        dist.all_to_all_single(
            recv_data, send_data,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=self.ep_group
        )

        return recv_data, recv_counts

    def _process_local_experts(
        self,
        tokens: torch.Tensor,
        counts_per_source: list
    ) -> torch.Tensor:
        """Process tokens through local experts."""
        # For each local expert, process its assigned tokens
        outputs = []

        # In practice, need to track which tokens go to which expert
        # This is simplified - real implementation tracks expert assignments
        for expert_idx, expert in enumerate(self.experts):
            # Get tokens for this expert
            expert_tokens = self._get_tokens_for_expert(
                tokens, expert_idx, counts_per_source
            )

            if expert_tokens.shape[0] > 0:
                expert_output = expert(expert_tokens)
                outputs.append(expert_output)

        return torch.cat(outputs, dim=0) if outputs else torch.empty(0, self.hidden_dim)

    def _get_tokens_for_expert(
        self,
        tokens: torch.Tensor,
        expert_idx: int,
        counts: list
    ) -> torch.Tensor:
        """Get tokens assigned to a specific local expert."""
        # Simplified - in practice, track indices during routing
        return tokens  # Placeholder
```

## 5D Parallelism: DP × TP × PP × CP × EP

### The Complete Picture

```
5D Mesh: shape = (D, P, T, C, E)

Example: 16,384 GPUs for 1T MoE with 128K context
- DP = 8    (data parallel replicas)
- PP = 16   (pipeline stages)
- TP = 8    (tensor parallel)
- CP = 4    (context parallel)
- EP = 4    (expert parallel)

Total: 8 × 16 × 8 × 4 × 4 = 16,384
```

### 5D Mesh Implementation

```python
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import torch.distributed as dist
import numpy as np

@dataclass
class FiveDConfig:
    """Configuration for 5D parallelism."""
    dp_size: int
    pp_size: int
    tp_size: int
    cp_size: int
    ep_size: int

    @property
    def world_size(self) -> int:
        return (self.dp_size * self.pp_size * self.tp_size *
                self.cp_size * self.ep_size)

    def validate(self) -> None:
        """Validate configuration constraints."""
        # TP × CP should fit within node for NVLink
        if self.tp_size * self.cp_size > 8:
            print("Warning: TP × CP > 8 may cross NVLink boundary")

        # EP should not be too large (load imbalance)
        if self.ep_size > 32:
            print("Warning: Large EP may cause load imbalance")

class FiveDMesh:
    """Device mesh for 5D parallelism."""

    def __init__(self, config: FiveDConfig):
        self.config = config
        config.validate()

        world_size = config.world_size
        devices = np.arange(world_size).reshape(
            config.dp_size,
            config.pp_size,
            config.tp_size,
            config.cp_size,
            config.ep_size
        )

        self.mesh = devices

        # Get my coordinates
        rank = dist.get_rank()
        coords = np.argwhere(devices == rank)[0]
        self.dp_rank = coords[0]
        self.pp_rank = coords[1]
        self.tp_rank = coords[2]
        self.cp_rank = coords[3]
        self.ep_rank = coords[4]

        # Create all process groups
        self._create_groups()

    def _create_groups(self):
        """Create all necessary process groups."""
        # Single-dimension groups
        self.dp_group = self._create_group_along(0)
        self.pp_group = self._create_group_along(1)
        self.tp_group = self._create_group_along(2)
        self.cp_group = self._create_group_along(3)
        self.ep_group = self._create_group_along(4)

        # Composite groups for fused operations
        self.tp_cp_group = self._create_group_along([2, 3])
        self.tp_ep_group = self._create_group_along([2, 4])

        # DP group that spans EP (for gradient sync in MoE)
        self.dp_ep_group = self._create_group_along([0, 4])

    def _create_group_along(self, dims) -> dist.ProcessGroup:
        """Create process group varying along specified dimension(s)."""
        if isinstance(dims, int):
            dims = [dims]

        # Build slice that fixes all dims except those specified
        slices = []
        for i, size in enumerate([
            self.config.dp_size,
            self.config.pp_size,
            self.config.tp_size,
            self.config.cp_size,
            self.config.ep_size
        ]):
            if i in dims:
                slices.append(slice(None))
            else:
                my_coords = [self.dp_rank, self.pp_rank, self.tp_rank,
                            self.cp_rank, self.ep_rank]
                slices.append(my_coords[i])

        ranks = self.mesh[tuple(slices)].flatten().tolist()
        return dist.new_group(ranks)

    def get_all_ranks(self) -> Dict[str, int]:
        """Return all ranks in a dictionary."""
        return {
            'dp': self.dp_rank,
            'pp': self.pp_rank,
            'tp': self.tp_rank,
            'cp': self.cp_rank,
            'ep': self.ep_rank
        }
```

### 5D Communication Matrix

| From \ To | TP | CP | PP | DP | EP |
|-----------|----|----|----|----|-----|
| TP | AllReduce | - | - | - | - |
| CP | - | Ring P2P | - | - | - |
| PP | - | - | P2P | - | - |
| DP | - | - | - | AllReduce | AllReduce (MoE grads) |
| EP | - | - | - | - | AlltoAll |

### Memory Analysis for 5D

Per-GPU memory with 5D parallelism:

**Parameters**:
$$M_{\text{params}} = \frac{N_{\text{dense}}}{T \times P} + \frac{N_{\text{expert}}}{T \times P \times E}$$

**Optimizer States**:
$$M_{\text{optimizer}} = 4 \times M_{\text{params}}$$

**Activations**:
$$M_{\text{activations}} = \frac{B \times S \times H}{T \times C} \times L_{\text{stage}} \times k_{\text{buffer}}$$

**Example**: 1T MoE (200B dense + 800B experts), 128K context, 16K GPUs

Configuration: DP=8, PP=16, TP=8, CP=4, EP=4

- Dense params: $\frac{200\text{B} \times 2}{8 \times 16} = 3.1\text{ GB}$
- Expert params: $\frac{800\text{B} \times 2}{8 \times 16 \times 4} = 3.1\text{ GB}$
- Optimizer: $4 \times 6.2 = 24.8\text{ GB}$
- Activations: $\frac{4096 \times 128\text{K} \times 8192 \times 2}{8 \times 4} \times 4 = 34\text{ GB}$
- **Total**: ~65 GB (fits 80GB A100)

## Practical Considerations

### Load Balancing in EP

Expert parallelism introduces load imbalance:

```python
class LoadBalancer:
    """Auxiliary loss for expert load balancing."""

    def __init__(
        self,
        num_experts: int,
        capacity_factor: float = 1.25,
        balance_loss_weight: float = 0.01
    ):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.balance_loss_weight = balance_loss_weight

    def compute_balance_loss(
        self,
        router_logits: torch.Tensor,
        expert_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.

        Args:
            router_logits: [batch * seq, num_experts]
            expert_mask: [batch * seq, num_experts] binary

        Returns:
            Scalar balance loss
        """
        # Fraction of tokens routed to each expert
        tokens_per_expert = expert_mask.float().mean(dim=0)

        # Average routing probability to each expert
        router_prob = torch.softmax(router_logits, dim=-1)
        router_prob_per_expert = router_prob.mean(dim=0)

        # Balance loss: minimize product
        # (encourages uniform distribution)
        balance_loss = (
            self.num_experts *
            (tokens_per_expert * router_prob_per_expert).sum()
        )

        return self.balance_loss_weight * balance_loss

    def get_capacity(self, num_tokens: int) -> int:
        """Get capacity per expert (max tokens it can handle)."""
        avg_tokens_per_expert = num_tokens / self.num_experts
        return int(avg_tokens_per_expert * self.capacity_factor)
```

### Overlapping Communication

In 5D parallelism, overlap opportunities:

```
Timeline for 5D training step:

│ GPU │ Op Type           │ Overlap With │
├─────┼───────────────────┼──────────────┤
│ All │ Forward compute   │              │
│ CP  │ Ring attention    │ FFN compute  │
│ EP  │ AlltoAll dispatch │ Self-attn    │
│ TP  │ AllReduce         │ Next layer   │
│ EP  │ AlltoAll combine  │ Next layer   │
│ All │ Backward compute  │              │
│ PP  │ Send activation   │ Next micro   │
│ DP  │ AllReduce grads   │ Next step    │
```

### Choosing Dimensions

Algorithm for 5D configuration:

```python
def choose_5d_config(
    total_gpus: int,
    model_params_dense: int,
    model_params_expert: int,
    num_experts: int,
    sequence_length: int,
    gpu_memory_gb: float = 80,
    gpus_per_node: int = 8
) -> FiveDConfig:
    """
    Choose optimal 5D parallelism configuration.

    Priority:
    1. TP × CP ≤ gpus_per_node (NVLink)
    2. EP divides num_experts evenly
    3. PP minimizes bubble
    4. DP maximizes throughput
    """

    best_config = None
    best_efficiency = 0

    # TP options: powers of 2 up to gpus_per_node
    for tp in [1, 2, 4, 8]:
        if tp > gpus_per_node:
            continue

        # CP options: fill remaining NVLink bandwidth
        for cp in [1, 2, 4, 8]:
            if tp * cp > gpus_per_node:
                continue

            # EP options: must divide num_experts
            for ep in [1, 2, 4, 8, 16, 32]:
                if num_experts % ep != 0:
                    continue

                # PP options
                for pp in [1, 2, 4, 8, 16, 32]:
                    # Check if DP is valid
                    dp = total_gpus // (tp * cp * ep * pp)
                    if dp < 1:
                        continue
                    if dp * tp * cp * ep * pp != total_gpus:
                        continue

                    # Check memory
                    config = FiveDConfig(dp, pp, tp, cp, ep)
                    mem = estimate_5d_memory(
                        config, model_params_dense, model_params_expert,
                        num_experts, sequence_length
                    )

                    if mem > gpu_memory_gb * 0.9:
                        continue

                    # Estimate efficiency
                    eff = estimate_5d_efficiency(
                        config, model_params_dense, model_params_expert,
                        num_experts, sequence_length
                    )

                    if eff > best_efficiency:
                        best_efficiency = eff
                        best_config = config

    if best_config is None:
        raise ValueError("No valid configuration found")

    return best_config
```

## Case Studies

### LLaMA 3 405B (Hypothetical 4D)

Training a dense 405B model with 128K context:

| Dimension | Value | Rationale |
|-----------|-------|-----------|
| DP | 32 | Maximize throughput |
| PP | 16 | 96 layers / 6 per stage |
| TP | 8 | Within NVLink |
| CP | 4 | Handle 128K context |

**Total**: 32 × 16 × 8 × 4 = 16,384 GPUs

### Mixtral 8x22B (4D + EP)

Training Mixtral with 8 experts:

| Dimension | Value | Rationale |
|-----------|-------|-----------|
| DP | 64 | High throughput |
| PP | 4 | Shallow MoE |
| TP | 4 | Modest hidden dim |
| EP | 8 | One expert per EP rank |

**Total**: 64 × 4 × 4 × 8 = 8,192 GPUs

### GPT-4 Scale MoE (Hypothetical 5D)

1T+ parameter MoE with 128 experts:

| Dimension | Value | Rationale |
|-----------|-------|-----------|
| DP | 16 | Data parallelism |
| PP | 32 | Deep model |
| TP | 8 | Within node |
| CP | 2 | 128K context |
| EP | 16 | 128/16 = 8 experts per rank |

**Total**: 16 × 32 × 8 × 2 × 16 = 131,072 GPUs

## Exercises

1. **4D design**: You have 4,096 GPUs and need to train a 70B model with 256K context. Design a 4D configuration. What's the memory per GPU?

2. **EP communication**: For a MoE with 64 experts, EP=8, and batch×seq=16384 tokens with top-2 routing:

   - How many tokens does each EP rank send in AlltoAll?
   - What's the AlltoAll volume?

3. **Ring attention analysis**: For CP=8 and sequence length 128K:

   - How many ring steps are needed?
   - What's the memory for K,V buffers?
   - Compare to AllGather memory requirement

4. **Load imbalance**: An MoE with 8 experts shows routing: [30%, 5%, 25%, 10%, 5%, 10%, 10%, 5%]. With capacity_factor=1.25, which experts will drop tokens?

5. **5D scaling**: A 5D configuration achieves 40% MFU on 16K GPUs. When scaling to 64K GPUs (4× DP), predict the new MFU and identify bottlenecks.

6. **Dimension ordering**: Propose an alternative ordering to (DP, PP, TP, CP, EP). Justify when it would be better.

## Key Takeaways

1. **4D adds Context Parallelism**: Sequence dimension partitioning for long contexts.

2. **5D adds Expert Parallelism**: MoE expert distribution across devices.

3. **Ring attention avoids O(S²) memory**: Streaming KV through the ring.

4. **EP requires AlltoAll**: Token routing is a permutation, not a reduction.

5. **TP × CP ≤ node size**: Keep highest-bandwidth communication on NVLink.

6. **Load balancing is critical for EP**: Auxiliary losses prevent expert collapse.

7. **Composite groups enable fusion**: TP-CP groups, DP-EP groups for combined operations.

8. **Configuration is multi-objective optimization**: Balance memory, compute, communication, and load balance.
