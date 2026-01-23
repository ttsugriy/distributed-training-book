---
title: "Expert Parallelism from Sparsity"
subtitle: "Routing Tokens to Distributed Experts"
---

<div class="chapter-opener" markdown>
Mixture of Experts models activate only a subset of parameters per token. This sparsity enables massive parameter counts without proportional compute costs—but requires careful routing to balance load across distributed experts.
</div>

<div class="investigation-question" markdown>
**The Question**: A model has 8 experts distributed across 8 GPUs. A token is routed to experts on GPUs 3 and 7. How does the token get there and back? What if all tokens want the same expert?
</div>

## The Sparsity Property

Dense neural networks have a fundamental property: every parameter participates in every computation. For a dense feedforward layer:

$$y = W_2(\sigma(W_1 x))$$

Every element of $W_1$ and $W_2$ contributes to every output.

**Definition (Conditional Computation)**: A computation is *sparse* if only a subset of parameters are activated for each input:

$$y = \sum_{i \in S(x)} w_i \cdot f_i(x)$$

where $S(x) \subseteq \{1, \ldots, E\}$ is an input-dependent *selection function* and $|S(x)| \ll E$.

This enables a crucial property.

**Theorem (Sparsity Scaling)**: A Mixture of Experts layer with $E$ experts, each of size $d_{\text{model}} \times d_{\text{ff}}$, has:

- **Parameters**: $E \cdot d_{\text{model}} \cdot d_{\text{ff}}$ (scales with $E$)
- **FLOPs per token**: $k \cdot d_{\text{model}} \cdot d_{\text{ff}}$ (independent of $E$)

where $k$ is the number of experts activated per token.

*Proof*: Each token only flows through $k$ selected experts, regardless of total expert count. The parameter count grows linearly with $E$, but compute per token remains constant at $O(k \cdot d_{\text{model}} \cdot d_{\text{ff}})$. □

This asymmetry is powerful: we can scale parameters (capacity) without scaling compute (cost).

## Mixture of Experts Architecture

### The MoE Layer

A Mixture of Experts layer replaces the standard feedforward network (FFN) in a transformer block:

```
Standard Transformer:      MoE Transformer:
┌─────────────────┐        ┌─────────────────┐
│   Attention     │        │   Attention     │
├─────────────────┤        ├─────────────────┤
│   Add & Norm    │        │   Add & Norm    │
├─────────────────┤        ├─────────────────┤
│      FFN        │   →    │   Router (Gate) │
│                 │        │    ↙  ↓  ↘      │
│                 │        │   E₀ E₁ ... E_n │
├─────────────────┤        ├─────────────────┤
│   Add & Norm    │        │   Add & Norm    │
└─────────────────┘        └─────────────────┘
```

Each expert $E_i$ is itself a complete FFN:

$$E_i(x) = W_2^{(i)} \cdot \text{GeLU}(W_1^{(i)} x)$$

The router (gating network) determines which experts process each token.

### Mathematical Formulation

For input token $x \in \mathbb{R}^{d}$:

**Step 1: Compute routing logits**
$$h = W_g x$$

where $W_g \in \mathbb{R}^{E \times d}$ is the gating weight matrix.

**Step 2: Apply gating function**
$$g = \text{Softmax}(h)$$

**Step 3: Select top-k experts**
$$S = \text{TopK}(g, k)$$

**Step 4: Compute weighted output**
$$y = \sum_{i \in S} \frac{g_i}{\sum_{j \in S} g_j} \cdot E_i(x)$$

The normalization ensures weights sum to 1 over selected experts.

### Why Sparsity Works

**Intuition**: Not all parameters need to be active for all inputs. Different experts can specialize:

- Expert 1: Handles syntax-related computations
- Expert 2: Handles factual knowledge
- Expert 3: Handles reasoning chains
- ...

The router learns to direct tokens to appropriate specialists.

**Empirically observed**: With sufficient capacity, experts do develop distinct specializations, though these are often difficult to interpret.

## Gating Mechanisms

The router is crucial: it determines the selection function $S(x)$.

### Softmax Gating (Original MoE)

The simplest approach:

$$g_i = \frac{e^{h_i}}{\sum_{j=1}^E e^{h_j}}, \quad h = W_g x$$

**Problem**: Softmax is "soft"—all experts get some weight. For true sparsity, we need hard selection.

### Top-k Gating

Select the $k$ experts with highest routing scores:

$$S = \{i : g_i \text{ is among top-}k\text{ values}\}$$

Renormalize weights:
$$\tilde{g}_i = \frac{g_i}{\sum_{j \in S} g_j} \text{ for } i \in S$$

**Typical values**: $k = 1$ (Switch Transformer) or $k = 2$ (GShard, original MoE).

### Noisy Top-k Gating

Add noise during training to encourage exploration:

$$h_i = (W_g x)_i + \epsilon_i \cdot \text{Softplus}((W_{\text{noise}} x)_i)$$

where $\epsilon_i \sim \mathcal{N}(0, 1)$.

The learned noise allows the model to:
1. Explore different expert assignments
2. Escape poor local optima in routing
3. Develop more balanced load distribution

### Expert Choice Routing

Instead of tokens choosing experts, experts choose tokens:

**Standard (Token Choice)**:

- Each token picks its top-k experts
- Leads to load imbalance

**Expert Choice** (Zhou et al., 2022):

- Each expert picks its top-k tokens
- Guarantees perfect load balance
- Each expert processes exactly $\text{capacity} = k \cdot T / E$ tokens

```python
def expert_choice_routing(tokens, router_logits, capacity):
    """
    Expert choice: experts select their top tokens.

    Args:
        tokens: [batch, seq, dim] - input tokens
        router_logits: [batch, seq, num_experts] - routing scores
        capacity: int - tokens per expert
    """
    batch, seq, dim = tokens.shape
    num_tokens = batch * seq

    # Reshape for routing
    tokens_flat = tokens.view(num_tokens, dim)
    logits_flat = router_logits.view(num_tokens, -1)

    # Transpose: [num_experts, num_tokens]
    expert_scores = logits_flat.t()

    # Each expert selects top-capacity tokens
    _, indices = expert_scores.topk(capacity, dim=1)

    # Gather selected tokens for each expert
    expert_inputs = []
    for e in range(num_experts):
        selected = tokens_flat[indices[e]]  # [capacity, dim]
        expert_inputs.append(selected)

    return expert_inputs, indices
```

## The AlltoAll Communication Pattern

Expert parallelism requires a unique communication pattern: AlltoAll.

### Why AlltoAll?

Consider 4 GPUs, each with 1 expert, processing a batch of tokens:

```
Before routing:
GPU 0: [t0, t1, t2, t3] - tokens 0-3 need various experts
GPU 1: [t4, t5, t6, t7] - tokens 4-7 need various experts
GPU 2: [t8, t9, t10, t11]
GPU 3: [t12, t13, t14, t15]

After routing decision:
t0 → E2    t4 → E0    t8 → E1     t12 → E3
t1 → E0    t5 → E3    t9 → E0     t13 → E1
t2 → E1    t6 → E2    t10 → E2    t14 → E0
t3 → E3    t7 → E1    t11 → E3    t15 → E2
```

Each GPU needs to:
1. Send its tokens to the correct expert's GPU
2. Receive tokens destined for its local expert

This is exactly the AlltoAll pattern:

```
AlltoAll:
GPU 0 sends:  [to_E0, to_E1, to_E2, to_E3]
GPU 1 sends:  [to_E0, to_E1, to_E2, to_E3]
GPU 2 sends:  [to_E0, to_E1, to_E2, to_E3]
GPU 3 sends:  [to_E0, to_E1, to_E2, to_E3]

After AlltoAll:
GPU 0 receives: [from_GPU0, from_GPU1, from_GPU2, from_GPU3] (all for E0)
GPU 1 receives: [from_GPU0, from_GPU1, from_GPU2, from_GPU3] (all for E1)
GPU 2 receives: [from_GPU0, from_GPU1, from_GPU2, from_GPU3] (all for E2)
GPU 3 receives: [from_GPU0, from_GPU1, from_GPU2, from_GPU3] (all for E3)
```

### Communication Volume

For AlltoAll with $P$ participants, each holding $n$ bytes total:

$$\text{Volume per GPU} = \frac{(P-1) \cdot n}{P}$$

Each GPU sends $(P-1)/P$ of its data to other GPUs.

**Total volume** (sum across all GPUs):

$$\text{Total} = P \cdot \frac{(P-1) \cdot n}{P} = (P-1) \cdot n$$

### The AlltoAll-AlltoAll Pattern

An MoE layer requires two AlltoAll operations:

```
┌────────────────┐
│ Token inputs   │  Local tokens on each GPU
├────────────────┤
│ Routing (Gate) │  Compute expert assignments
├────────────────┤
│ AlltoAll       │  ← Dispatch: tokens → expert GPUs
├────────────────┤
│ Expert FFN     │  Each GPU runs its local expert(s)
├────────────────┤
│ AlltoAll       │  ← Combine: outputs → original GPUs
├────────────────┤
│ Weighted sum   │  Combine expert outputs
└────────────────┘
```

**Implementation**:

```python
class MoELayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        num_experts_per_tok: int,
        expert_group: dist.ProcessGroup
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_group = expert_group
        self.world_size = dist.get_world_size(expert_group)
        self.rank = dist.get_rank(expert_group)

        # Router
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Local experts (this GPU's experts)
        experts_per_rank = num_experts // self.world_size
        self.local_experts = nn.ModuleList([
            FFN(hidden_dim, ffn_dim)
            for _ in range(experts_per_rank)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq, hidden] input tokens
        Returns:
            [batch, seq, hidden] output tokens
        """
        batch, seq, hidden = x.shape
        x_flat = x.view(-1, hidden)  # [batch*seq, hidden]
        num_tokens = x_flat.shape[0]

        # Step 1: Compute routing
        router_logits = self.gate(x_flat)  # [num_tokens, num_experts]
        routing_weights, selected_experts = self._top_k_gating(router_logits)

        # Step 2: Prepare for AlltoAll dispatch
        # Group tokens by destination expert
        dispatch_data, dispatch_indices = self._prepare_dispatch(
            x_flat, selected_experts, routing_weights
        )

        # Step 3: AlltoAll dispatch
        # Each GPU sends tokens to expert-owning GPUs
        received_data = self._all_to_all(dispatch_data)

        # Step 4: Process through local experts
        expert_outputs = self._run_local_experts(received_data)

        # Step 5: AlltoAll combine
        # Send outputs back to original GPUs
        combined_outputs = self._all_to_all(expert_outputs)

        # Step 6: Weighted combination
        output = self._combine_outputs(
            combined_outputs, dispatch_indices, routing_weights, num_tokens, hidden
        )

        return output.view(batch, seq, hidden)

    def _top_k_gating(self, logits: torch.Tensor):
        """Select top-k experts per token."""
        weights = F.softmax(logits, dim=-1)
        top_weights, top_indices = weights.topk(
            self.num_experts_per_tok, dim=-1
        )
        # Renormalize
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        return top_weights, top_indices

    def _all_to_all(self, x: torch.Tensor) -> torch.Tensor:
        """Perform AlltoAll communication."""
        return all_to_all_single(
            x,
            output_split_sizes=None,
            input_split_sizes=None,
            group=self.expert_group
        )
```

## Load Balancing

The critical challenge: what if all tokens want the same expert?

### The Problem

Without balancing, expert load can be highly skewed:

```
Unbalanced:
Expert 0: ████████████████████ (80% of tokens)
Expert 1: ██ (5%)
Expert 2: ██ (5%)
Expert 3: ███ (10%)

→ Expert 0's GPU is the bottleneck
→ Other GPUs mostly idle
→ Training throughput collapses
```

### Solution 1: Auxiliary Load Balancing Loss

Add a loss term that penalizes imbalanced routing:

$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot p_i$$

where:

- $f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}[\text{token } t \text{ routes to expert } i]$ (fraction of tokens to expert $i$)
- $p_i = \frac{1}{T} \sum_{t=1}^{T} g_i^{(t)}$ (average routing probability for expert $i$)
- $\alpha$ is a balancing coefficient (typically 0.01-0.1)

**Why this works**: The product $f_i \cdot p_i$ is minimized when load is uniform:

- If $f_i$ is high (many tokens), $p_i$ must be low to minimize loss
- The gradient pushes the router toward balanced assignments

**Derivation**: For uniform routing, $f_i = p_i = 1/E$ for all $i$:

$$\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} \frac{1}{E} \cdot \frac{1}{E} = \alpha \cdot E \cdot E \cdot \frac{1}{E^2} = \alpha$$

Any deviation from uniform increases $\mathcal{L}_{\text{aux}}$ above $\alpha$.

### Solution 2: Capacity Factor

Limit how many tokens each expert can process:

$$C = \left\lceil \text{capacity\_factor} \times \frac{T \times k}{E} \right\rceil$$

where:

- $T$ = total tokens
- $k$ = experts per token
- $E$ = total experts
- $\text{capacity\_factor} \geq 1.0$ (typically 1.25-2.0)

Tokens exceeding capacity are:

- **Dropped**: Set output to zero or skip (Switch Transformer)
- **Overflowed**: Route to next-best expert (GShard)

```python
def apply_capacity_limit(
    tokens: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    capacity_factor: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply capacity limit to expert assignments.

    Returns:
        tokens, indices, weights with capacity limits applied
    """
    num_tokens, k = expert_indices.shape
    num_experts = expert_indices.max().item() + 1

    # Compute capacity per expert
    capacity = int(math.ceil(capacity_factor * num_tokens * k / num_experts))

    # Count assignments per expert
    expert_counts = torch.zeros(num_experts, dtype=torch.long, device=tokens.device)

    # Mask for valid assignments
    valid_mask = torch.zeros_like(expert_indices, dtype=torch.bool)

    for tok_idx in range(num_tokens):
        for k_idx in range(k):
            expert_id = expert_indices[tok_idx, k_idx].item()
            if expert_counts[expert_id] < capacity:
                valid_mask[tok_idx, k_idx] = True
                expert_counts[expert_id] += 1

    # Zero out weights for dropped assignments
    expert_weights = expert_weights * valid_mask.float()

    # Renormalize weights
    weight_sum = expert_weights.sum(dim=-1, keepdim=True)
    weight_sum = torch.where(weight_sum > 0, weight_sum, torch.ones_like(weight_sum))
    expert_weights = expert_weights / weight_sum

    return tokens, expert_indices, expert_weights
```

### Solution 3: Z-Loss

Regularize the router logits to prevent over-confident routing:

$$\mathcal{L}_z = \frac{1}{T} \sum_{t=1}^{T} \log^2 \left( \sum_{i=1}^{E} e^{h_i^{(t)}} \right)$$

This penalizes large logits, encouraging softer routing distributions.

### Comparison of Balancing Strategies

| Strategy | Pros | Cons |
|----------|------|------|
| Auxiliary loss | Differentiable, end-to-end | May interfere with main loss |
| Capacity factor | Hard guarantee on balance | Drops tokens, non-differentiable |
| Expert choice | Perfect balance | Requires architectural changes |
| Z-loss | Stabilizes training | Doesn't directly balance |

**Best practice**: Combine auxiliary loss + capacity factor + z-loss.

## Token Dropping Analysis

What happens when tokens are dropped?

### Dropped Token Impact

Let $d$ be the drop rate. The effective batch size becomes:

$$B_{\text{eff}} = B \times (1 - d)$$

**Gradient estimation**: Dropped tokens don't contribute gradients:

$$\nabla_{\text{observed}} = \frac{1}{B_{\text{eff}}} \sum_{i \in \text{not dropped}} \nabla L_i$$

This is an unbiased estimator of the gradient if dropping is random, but routing-based dropping is not random.

### The Bias Problem

Tokens routed to popular experts are more likely to be dropped. These are often:

- More common patterns
- Important for generalization
- Tokens the model "wants" to process similarly

Systematic dropping of these tokens can hurt model quality.

### Mitigation: No-Token-Left-Behind

Alternative strategies:
1. **Overflow routing**: Dropped tokens go to next-best expert
2. **Auxiliary buffer**: Store dropped tokens, process in next batch
3. **Increased capacity**: Set capacity_factor = 2.0 or higher

## Expert Parallelism Implementation

### Placing Experts Across GPUs

With $E$ experts and $P$ GPUs:

**Case 1: $E = P$ (one expert per GPU)**
```
GPU 0: Expert 0
GPU 1: Expert 1
...
GPU P-1: Expert P-1
```

This is the simplest case: each AlltoAll participant is exactly one expert.

**Case 2: $E > P$ (multiple experts per GPU)**
```
GPU 0: Expert 0, Expert 1
GPU 1: Expert 2, Expert 3
...
GPU P-1: Expert 2(P-1), Expert 2P-1
```

Local routing happens without communication; only cross-GPU routing uses AlltoAll.

**Case 3: $E < P$ (expert sharded across GPUs)**

Tensor parallelism within each expert:
```
Expert 0: GPU 0, GPU 1 (TP=2)
Expert 1: GPU 2, GPU 3 (TP=2)
...
```

### Full Implementation

```python
class DistributedMoE(nn.Module):
    """
    Distributed Mixture of Experts layer with AlltoAll communication.
    """
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        expert_group: Optional[dist.ProcessGroup] = None
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_coef = aux_loss_coef
        self.z_loss_coef = z_loss_coef

        # Expert parallel group
        if expert_group is None:
            expert_group = dist.distributed_c10d._get_default_group()
        self.expert_group = expert_group
        self.world_size = dist.get_world_size(expert_group)
        self.rank = dist.get_rank(expert_group)

        assert num_experts % self.world_size == 0
        self.experts_per_rank = num_experts // self.world_size

        # Router (on all ranks)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Local experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim, bias=False),
                nn.GELU(),
                nn.Linear(ffn_dim, hidden_dim, bias=False)
            )
            for _ in range(self.experts_per_rank)
        ])

        # Auxiliary loss storage
        self.aux_loss = 0.0
        self.z_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with AlltoAll communication.

        Args:
            x: [batch, seq, hidden]
        Returns:
            [batch, seq, hidden]
        """
        batch, seq, hidden = x.shape
        num_tokens = batch * seq
        x_flat = x.view(num_tokens, hidden)

        # ===== ROUTING =====
        router_logits = self.gate(x_flat)  # [num_tokens, num_experts]

        # Z-loss for stability
        self.z_loss = self.z_loss_coef * torch.logsumexp(
            router_logits, dim=-1
        ).square().mean()

        # Softmax routing
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        top_weights, top_indices = router_probs.topk(self.top_k, dim=-1)

        # Renormalize
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # Auxiliary load balancing loss
        self.aux_loss = self._compute_aux_loss(router_probs, top_indices)

        # ===== CAPACITY LIMITING =====
        capacity = int(math.ceil(
            self.capacity_factor * num_tokens * self.top_k / self.num_experts
        ))

        # Build dispatch mask and positions
        dispatch_mask, combine_weights, tokens_per_expert = self._build_dispatch(
            x_flat, top_indices, top_weights, capacity
        )

        # ===== ALLTOALL DISPATCH =====
        # Prepare data: group tokens by target expert
        dispatch_tokens = self._gather_for_dispatch(x_flat, dispatch_mask, capacity)

        # AlltoAll: send tokens to expert-owning ranks
        # Shape: [experts_per_rank * capacity, hidden] per rank
        recv_tokens = self._dispatch_all_to_all(dispatch_tokens, tokens_per_expert)

        # ===== EXPERT COMPUTATION =====
        expert_outputs = self._run_experts(recv_tokens, capacity)

        # ===== ALLTOALL COMBINE =====
        # AlltoAll: send outputs back
        combined = self._combine_all_to_all(expert_outputs, tokens_per_expert)

        # ===== WEIGHTED SUM =====
        output = self._weighted_combine(combined, combine_weights, dispatch_mask)

        return output.view(batch, seq, hidden)

    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary load balancing loss."""
        num_tokens = router_probs.shape[0]

        # f_i: fraction of tokens routed to expert i
        # Use one-hot encoding of selections
        one_hot = F.one_hot(
            selected_experts.view(-1),
            num_classes=self.num_experts
        ).float()
        tokens_per_expert = one_hot.sum(dim=0)
        f = tokens_per_expert / (num_tokens * self.top_k)

        # p_i: average routing probability for expert i
        p = router_probs.mean(dim=0)

        # Aux loss: encourages uniform f and p
        aux_loss = self.aux_loss_coef * self.num_experts * (f * p).sum()

        return aux_loss

    def _dispatch_all_to_all(
        self,
        tokens: torch.Tensor,
        tokens_per_expert: torch.Tensor
    ) -> torch.Tensor:
        """
        AlltoAll to dispatch tokens to expert-owning ranks.
        """
        # Compute send/recv sizes
        send_sizes = tokens_per_expert.tolist()
        recv_sizes = [0] * self.world_size

        # Exchange sizes
        dist.all_to_all_single(
            torch.tensor(recv_sizes, device=tokens.device),
            torch.tensor(send_sizes, device=tokens.device),
            group=self.expert_group
        )

        # Allocate receive buffer
        total_recv = sum(recv_sizes)
        recv_buffer = torch.empty(
            total_recv, self.hidden_dim,
            dtype=tokens.dtype, device=tokens.device
        )

        # AlltoAll data
        dist.all_to_all_single(
            recv_buffer,
            tokens,
            output_split_sizes=recv_sizes,
            input_split_sizes=send_sizes,
            group=self.expert_group
        )

        return recv_buffer

    def _run_experts(
        self,
        tokens: torch.Tensor,
        capacity: int
    ) -> torch.Tensor:
        """Run tokens through local experts."""
        outputs = []

        # Tokens are grouped by expert
        offset = 0
        for expert_idx, expert in enumerate(self.experts):
            # Get tokens for this expert
            expert_tokens = tokens[offset:offset + capacity]

            # Forward through expert
            expert_out = expert(expert_tokens)
            outputs.append(expert_out)

            offset += capacity

        return torch.cat(outputs, dim=0)

    def get_aux_loss(self) -> torch.Tensor:
        """Get total auxiliary loss (add to main loss)."""
        return self.aux_loss + self.z_loss
```

## Gradient Flow Through Sparse Routing

How do gradients flow through the discrete expert selection?

### The Differentiability Problem

Top-k selection is non-differentiable:
$$\frac{\partial \text{TopK}(g)}{\partial g} = \text{undefined}$$

We can't backpropagate through the selection operation.

### Straight-Through Estimator

Use the selected weights, which are differentiable:

**Forward**:
$$y = \sum_{i \in S} \tilde{g}_i \cdot E_i(x)$$

**Backward**:
$$\frac{\partial L}{\partial g_i} = \tilde{g}_i \cdot \frac{\partial L}{\partial E_i(x)} \cdot E_i(x)$$

The gradient flows through the routing weights $\tilde{g}_i$, not the selection.

### Router Gradient

The router receives gradient from:
1. **Selected expert outputs**: How well did selected experts perform?
2. **Auxiliary losses**: Push toward load balance

This means the router learns to select experts that:

- Produce good outputs for the token
- Don't overload any single expert

## Composition with Other Parallelism Dimensions

Expert parallelism combines naturally with other forms:

### EP + DP (Expert Parallel + Data Parallel)

Most common combination:

```
                    Data Parallel Group (replicas)
                    ┌───────────────────────────┐
                    │                           │
     ┌──────────────┼─────────────┬─────────────┼──────────────┐
     │              │             │             │              │
     ▼              ▼             ▼             ▼              │
 ┌───────┐     ┌───────┐     ┌───────┐     ┌───────┐          │
 │Replica│     │Replica│     │Replica│     │Replica│          │
 │   0   │     │   1   │     │   2   │     │   3   │          │
 └───┬───┘     └───┬───┘     └───┬───┘     └───┬───┘          │
     │             │             │             │              │
     └─────────────┴──────┬──────┴─────────────┘              │
                          │                                    │
                 Expert Parallel Group                         │
                 ┌────────┴────────┐                          │
                 │                 │                          │
            ┌────┼────┐       ┌────┼────┐                     │
            ▼    ▼    ▼       ▼    ▼    ▼                     │
           E0   E1   E2      E3   E4   E5                     │
           │    │    │       │    │    │                      │
           GPU0 GPU1 GPU2   GPU3 GPU4 GPU5                    │
```

**Communication pattern**:

- AlltoAll within EP group (expert routing)
- AllReduce across DP replicas (gradient sync)

### EP + TP (Expert Parallel + Tensor Parallel)

For very large experts, shard each expert:

```
Expert 0:        Expert 1:
┌────┬────┐      ┌────┬────┐
│TP0 │TP1 │      │TP0 │TP1 │
│GPU0│GPU1│      │GPU2│GPU3│
└────┴────┘      └────┴────┘
     │                │
     └────────────────┘
          AlltoAll
```

Each expert uses tensor parallelism internally.

### Full 3D: EP + DP + TP

```python
def create_3d_moe_groups(
    world_size: int,
    tp_size: int,
    ep_size: int
) -> Tuple[ProcessGroup, ProcessGroup, ProcessGroup]:
    """
    Create process groups for 3D parallelism with MoE.

    world_size = tp_size * ep_size * dp_size
    """
    dp_size = world_size // (tp_size * ep_size)

    rank = dist.get_rank()

    # Tensor parallel: consecutive ranks
    tp_group_id = rank // tp_size
    tp_ranks = list(range(tp_group_id * tp_size, (tp_group_id + 1) * tp_size))
    tp_group = dist.new_group(tp_ranks)

    # Expert parallel: strided by TP size
    ep_group_id = (rank % (tp_size * ep_size)) // tp_size
    ep_ranks = [
        (rank // (tp_size * ep_size)) * (tp_size * ep_size) +
        i * tp_size + rank % tp_size
        for i in range(ep_size)
    ]
    ep_group = dist.new_group(ep_ranks)

    # Data parallel: strided by TP * EP size
    dp_ranks = [
        rank % (tp_size * ep_size) + i * (tp_size * ep_size)
        for i in range(dp_size)
    ]
    dp_group = dist.new_group(dp_ranks)

    return tp_group, ep_group, dp_group
```

### Communication Costs

For a model with:

- $H$ = hidden dimension
- $E$ = number of experts
- $T$ = tokens per batch
- $k$ = experts per token

| Parallelism | Operation | Volume (per GPU) |
|-------------|-----------|------------------|
| EP only | 2 × AlltoAll | $2 \times \frac{(P-1)}{P} \times T \times H$ |
| EP + DP | AlltoAll + AllReduce | AlltoAll + $2 \times \frac{(P_{DP}-1)}{P_{DP}} \times \text{params}$ |
| EP + TP | AlltoAll + AllReduce | AlltoAll + $2 \times \frac{(P_{TP}-1)}{P_{TP}} \times T \times H$ |

## Practical Considerations

### When to Use MoE

**Good fit for MoE**:

- Very large models where dense is prohibitively expensive
- High throughput requirements (more params, same compute)
- Tasks benefiting from specialization

**Poor fit for MoE**:

- Small models (routing overhead dominates)
- Low-latency inference (routing adds latency)
- Limited GPU interconnect (AlltoAll is bandwidth-intensive)

### Hyperparameter Guidelines

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| num_experts | 8-128 | Powers of 2 for easy sharding |
| top_k | 1-2 | k=2 more stable, k=1 more efficient |
| capacity_factor | 1.25-2.0 | Higher = fewer drops, more memory |
| aux_loss_coef | 0.01-0.1 | Too high hurts main task |
| z_loss_coef | 0.001-0.01 | Stabilizes training |

### Common Pitfalls

**Pitfall 1: Routing collapse**

All tokens route to one expert. Signs:

- One expert sees 90%+ of tokens
- Auxiliary loss stays high
- Model quality degrades

Fix: Increase aux_loss_coef, add jitter noise, check initialization.

**Pitfall 2: Capacity overflow**

Too many tokens dropped. Signs:

- High drop rate (>10%)
- Training loss unstable
- Gradient variance increases

Fix: Increase capacity_factor, reduce batch size, more experts.

**Pitfall 3: AlltoAll bottleneck**

Communication dominates compute. Signs:

- Low GPU utilization
- Training much slower than expected
- AlltoAll takes >50% of step time

Fix: Increase tokens per batch, reduce EP size, improve network.

**Pitfall 4: Expert underutilization**

Some experts rarely used. Signs:

- Load imbalance metrics show skew
- Some expert parameters barely update
- Model capacity wasted

Fix: Expert choice routing, larger aux_loss_coef, random routing regularization.

## Complete MoE Transformer Block

```python
class MoETransformerBlock(nn.Module):
    """
    Transformer block with Mixture of Experts FFN.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        expert_group: Optional[ProcessGroup] = None,
        use_moe: bool = True  # Some layers can be dense
    ):
        super().__init__()

        # Attention
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

        # FFN (MoE or dense)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        if use_moe:
            self.ffn = DistributedMoE(
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                num_experts=num_experts,
                top_k=top_k,
                capacity_factor=capacity_factor,
                expert_group=expert_group
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, hidden_dim)
            )

        self.use_moe = use_moe

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            output, aux_loss
        """
        # Attention block
        residual = x
        x = self.attn_norm(x)
        x, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = residual + x

        # FFN block
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        # Get auxiliary loss if MoE
        aux_loss = self.ffn.get_aux_loss() if self.use_moe else 0.0

        return x, aux_loss


class MoETransformer(nn.Module):
    """
    Full MoE Transformer with alternating dense and MoE layers.
    """
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int,
        moe_frequency: int = 2,  # Every Nth layer is MoE
        expert_group: Optional[ProcessGroup] = None,
        **moe_kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            MoETransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                num_experts=num_experts,
                expert_group=expert_group,
                use_moe=(i % moe_frequency == moe_frequency - 1),
                **moe_kwargs
            )
            for i in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning output and total auxiliary loss.
        """
        total_aux_loss = 0.0

        for layer in self.layers:
            x, aux_loss = layer(x, attention_mask)
            total_aux_loss = total_aux_loss + aux_loss

        return x, total_aux_loss
```

## Exercises

1. **AlltoAll volume analysis**: With 8 experts on 8 GPUs, 1024 tokens per GPU, hidden dimension 4096, calculate the AlltoAll dispatch volume (a) assuming uniform distribution, and (b) if 50% of tokens go to expert 0.

2. **Capacity factor selection**: You have 64 tokens, 8 experts, top_k=2, and observe 10% token drop rate with capacity_factor=1.25. What capacity_factor would reduce drops to <1%?

3. **Load balancing loss**: Derive the gradient of the auxiliary load balancing loss with respect to the router logits. Show that the gradient pushes toward uniform expert selection.

4. **Expert choice implementation**: Implement expert choice routing where each expert selects its top-C tokens. Prove that this guarantees perfect load balance.

5. **Communication overlap**: Design a scheme to overlap AlltoAll dispatch with attention computation in the previous layer. What are the constraints?

6. **3D parallelism groups**: With world_size=64, TP=4, EP=8, DP=2, enumerate all process groups for rank 13.

7. **Routing collapse detection**: Design a monitoring system to detect routing collapse early in training. What metrics would you track, and what thresholds would trigger alerts?

8. **Gradient analysis**: In a model with top_k=2, expert 0 sees 60% of tokens and expert 1 sees 40%. Analyze how the auxiliary loss gradient affects these proportions.

## Key Takeaways

1. **Sparsity enables scale**: MoE models can have 10-100× more parameters with constant compute per token.

2. **AlltoAll is the communication pattern**: Tokens go to experts, outputs come back—requiring two AlltoAll operations per MoE layer.

3. **Load balancing is critical**: Without balancing, routing collapse makes sparsity useless. Use auxiliary loss + capacity limiting.

4. **Expert choice guarantees balance**: Letting experts choose tokens, rather than tokens choosing experts, ensures perfect load distribution.

5. **Capacity factor trades off drops vs. memory**: Higher capacity = fewer dropped tokens but more memory per expert.

6. **Composition is natural**: EP combines cleanly with DP and TP through orthogonal process groups.

7. **Gradients flow through weights**: The top-k selection is non-differentiable, but routing weights provide gradient signal to the router.

8. **Routing overhead matters**: For small models or low-latency requirements, the routing computation and AlltoAll communication may dominate.
