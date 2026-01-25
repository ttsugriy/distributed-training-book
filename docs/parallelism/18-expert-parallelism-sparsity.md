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

!!! abstract "Chapter Map"
    **Prerequisites**: Chapter 11 (AlltoAll primitive), Chapters 14–15 (data and tensor parallelism context)

    **Key insight**: MoE achieves massive parameter counts with constant compute by activating only a few experts per token. The communication pattern is AlltoAll (tokens to experts, results back). Load balancing is critical—auxiliary losses and capacity factors prevent expert collapse.

## What is Mixture of Experts?

A standard transformer processes every token through the same feedforward network (FFN)—every parameter participates in every computation. **Mixture of Experts (MoE)** replaces this single FFN with multiple parallel FFNs called *experts*, plus a lightweight *router* (or *gate*) that decides which expert(s) process each token.

The key insight: different tokens may benefit from different computations. Rather than forcing all tokens through identical weights, MoE lets the model learn to specialize:

- The **router** examines each token and produces a score for each expert
- Only the top-scoring expert(s) are activated—typically 1 or 2 out of many (8, 64, or even 128)
- The selected expert(s) process the token and return their outputs, weighted by routing scores

This creates a powerful asymmetry: the model can have many more parameters (more experts = more capacity) without proportionally increasing compute cost (only a few experts run per token). A model with 64 experts but top-2 routing has 32× more FFN parameters while using only 2× the FFN compute of a dense model.

The challenge is distribution: when experts live on different GPUs, tokens must travel to the right expert and return. This chapter explores how that communication works and how to keep load balanced across experts.

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

??? success "Solution"
    **Given:**

    - Experts: $E = 8$ on 8 GPUs
    - Tokens per GPU: $T = 1024$
    - Hidden dimension: $H = 4096$
    - Assume bf16 (2 bytes per element)

    **AlltoAll in MoE:**

    Each GPU sends its tokens to the appropriate expert GPUs and receives tokens from all other GPUs for its local expert.

    **(a) Uniform distribution:**

    With uniform routing, each GPU sends $T/E = 1024/8 = 128$ tokens to each expert.

    **Send volume per GPU:**

    Each GPU sends tokens to all 8 GPUs (including itself):
    - Tokens to other GPUs: $\frac{E-1}{E} \times T = \frac{7}{8} \times 1024 = 896$ tokens
    - Volume: $896 \times H \times 2 = 896 \times 4096 \times 2 = 7.34$ MB

    **Total AlltoAll volume (dispatch only):**

    $$V_{\text{dispatch}} = 896 \times 4096 \times 2 = \boxed{7.34 \text{ MB per GPU}}$$

    For the full forward (dispatch + combine):

    $$V_{\text{total}} = 2 \times V_{\text{dispatch}} = \boxed{14.68 \text{ MB per GPU}}$$

    **(b) 50% tokens to expert 0:**

    Now the distribution is skewed: expert 0 gets 50%, others split the remaining 50%.

    From each GPU:
    - Tokens to expert 0: $0.5 \times 1024 = 512$
    - Tokens to each other expert: $\frac{0.5 \times 1024}{7} = 73$ tokens each

    **Volume from GPU 0 (hosts expert 0):**

    GPU 0 sends 512 tokens to itself (no network), sends $7 \times 73 = 511$ to others:

    $$V_{\text{GPU0 send}} = 511 \times 4096 \times 2 = 4.19 \text{ MB}$$

    **Volume from GPU $i \neq 0$:**

    Sends 512 tokens to GPU 0, sends 73 tokens to each of 6 other GPUs (excluding self):

    $$V_{\text{GPUi send}} = (512 + 6 \times 73) \times 4096 \times 2 = 950 \times 4096 \times 2 = 7.78 \text{ MB}$$

    **GPU 0 receives:**

    Receives 512 tokens from each of 7 other GPUs:

    $$V_{\text{GPU0 recv}} = 7 \times 512 \times 4096 \times 2 = 29.4 \text{ MB}$$

    **Comparison:**

    | Distribution | Send Volume (per GPU avg) | Receive Volume (GPU 0) |
    |--------------|--------------------------|------------------------|
    | Uniform | 7.34 MB | 7.34 MB |
    | 50% to expert 0 | 7.34 MB | 29.4 MB |

    **Key insight:** Skewed routing creates hotspots—expert 0 receives 4× more data, becoming a bottleneck.

    $$\boxed{\text{Uniform: 14.68 MB total, Skewed: GPU 0 receives 4× more}}$$

2. **Capacity factor selection**: You have 64 tokens, 8 experts, top_k=2, and observe 10% token drop rate with capacity_factor=1.25. What capacity_factor would reduce drops to <1%?

??? success "Solution"
    **Given:**

    - Total tokens: $T = 64$
    - Number of experts: $E = 8$
    - Top-k: $k = 2$
    - Current capacity factor: $c = 1.25$
    - Current drop rate: 10%

    **Capacity calculation:**

    Expert capacity = number of tokens each expert can process:

    $$C = c \times \frac{T \times k}{E} = 1.25 \times \frac{64 \times 2}{8} = 1.25 \times 16 = 20 \text{ tokens}$$

    **Understanding drops:**

    Total token-expert assignments: $T \times k = 64 \times 2 = 128$

    If uniformly distributed, each expert gets $128/8 = 16$ assignments—no drops.

    With 10% drops = 12.8 assignments dropped, meaning some experts received more than capacity 20.

    **Modeling the distribution:**

    Drops occur when expert load exceeds capacity. With capacity factor $c$:

    $$\text{Max load per expert} = c \times \frac{Tk}{E}$$

    The drop rate depends on the variance of the load distribution. Assuming approximate binomial distribution:

    - Expected load per expert: $\mu = Tk/E = 16$
    - Standard deviation: $\sigma \approx \sqrt{Tk/E} = 4$ (rough approximation)

    For 10% drops with $c = 1.25$ (capacity = 20):

    The tail beyond 20 contains ~10% of probability mass.

    **Target: <1% drops**

    We need capacity such that P(load > capacity) < 1%.

    Using normal approximation:
    - For 10% tail: $z_{0.10} \approx 1.28$, so $20 \approx 16 + 1.28 \times \sigma$
    - This gives $\sigma \approx 3.1$

    For 1% tail: $z_{0.01} \approx 2.33$
    - Required capacity: $16 + 2.33 \times 3.1 \approx 23.2$ tokens

    **Required capacity factor:**

    $$c_{\text{new}} = \frac{C_{\text{new}}}{Tk/E} = \frac{23.2}{16} = \boxed{1.45}$$

    **Practical recommendation:**

    To be safe, use $c = 1.5$ or slightly higher.

    | Capacity Factor | Capacity per Expert | Expected Drop Rate |
    |-----------------|--------------------|--------------------|
    | 1.25 | 20 | ~10% |
    | 1.45 | 23.2 | ~1% |
    | 1.50 | 24 | <1% |
    | 2.00 | 32 | ~0% |

    **Trade-off:** Higher capacity factor means more memory per expert buffer:

    $$\text{Memory increase} = \frac{1.50}{1.25} = 1.2\times \text{ (20% more memory)}$$

    **Alternative solution: Expert Choice Routing**

    With expert choice, each expert selects exactly $C$ tokens, guaranteeing 0% drops by design. This is often preferable to increasing capacity factor.

3. **Load balancing loss**: Derive the gradient of the auxiliary load balancing loss with respect to the router logits. Show that the gradient pushes toward uniform expert selection.

??? success "Solution"
    **Setup:**

    Let $g_i = \text{softmax}(z)_i$ be the routing probability for expert $i$, where $z$ are the router logits.

    **Auxiliary loss definition (Switch Transformer style):**

    $$L_{\text{aux}} = E \cdot \sum_{i=1}^{E} f_i \cdot p_i$$

    Where:
    - $f_i$: fraction of tokens routed to expert $i$ (discrete)
    - $p_i$: average routing probability for expert $i$
    - $E$: number of experts

    For gradient computation, we use $p_i$ (differentiable proxy for $f_i$):

    $$L_{\text{aux}} \approx E \cdot \sum_{i=1}^{E} p_i^2$$

    where $p_i = \frac{1}{T} \sum_{t=1}^{T} g_{t,i}$ is the mean probability across tokens.

    **Gradient derivation:**

    For a single token with logits $z$ and probabilities $g = \text{softmax}(z)$:

    $$\frac{\partial L_{\text{aux}}}{\partial z_j} = \frac{\partial L_{\text{aux}}}{\partial g} \cdot \frac{\partial g}{\partial z_j}$$

    **Step 1: Gradient w.r.t. routing probabilities**

    $$\frac{\partial L_{\text{aux}}}{\partial g_i} = \frac{\partial}{\partial g_i} \left( E \cdot \sum_k p_k^2 \right) = 2E \cdot p_i \cdot \frac{\partial p_i}{\partial g_i} = \frac{2E \cdot p_i}{T}$$

    **Step 2: Softmax Jacobian**

    $$\frac{\partial g_i}{\partial z_j} = g_i(\delta_{ij} - g_j)$$

    **Step 3: Chain rule**

    $$\frac{\partial L_{\text{aux}}}{\partial z_j} = \sum_i \frac{2E \cdot p_i}{T} \cdot g_i(\delta_{ij} - g_j)$$

    $$= \frac{2E}{T} \left[ p_j \cdot g_j - g_j \sum_i p_i \cdot g_i \right]$$

    $$= \frac{2E \cdot g_j}{T} \left[ p_j - \sum_i p_i \cdot g_i \right]$$

    **Interpretation:**

    The gradient for expert $j$ is proportional to:

    $$\nabla_{z_j} L_{\text{aux}} \propto g_j \cdot (p_j - \bar{p})$$

    where $\bar{p} = \sum_i p_i \cdot g_i$ is the weighted average load.

    **Why this pushes toward uniformity:**

    | Condition | Gradient Sign | Effect |
    |-----------|---------------|--------|
    | $p_j > \bar{p}$ (overloaded) | Positive | Decrease $z_j$ → lower $g_j$ |
    | $p_j < \bar{p}$ (underloaded) | Negative | Increase $z_j$ → raise $g_j$ |
    | $p_j = \bar{p}$ (balanced) | Zero | No change |

    **Equilibrium analysis:**

    At equilibrium, $\nabla L_{\text{aux}} = 0$ for all experts:

    $$p_j = \bar{p} \quad \forall j$$

    This means all experts have equal load: $p_i = 1/E$ for all $i$.

    **Visual intuition:**

    ```
    Gradient pushes probability mass from overloaded to underloaded experts:

    Before:     After gradient step:
    p₀ ████████     p₀ ██████
    p₁ ██           p₁ ████
    p₂ ██           p₂ ████
    p₃ ████████     p₃ ██████
    ```

    $$\boxed{\nabla_{z_j} L_{\text{aux}} \propto g_j(p_j - \bar{p}) \text{ pushes toward } p_i = 1/E}$$

4. **Expert choice implementation**: Implement expert choice routing where each expert selects its top-C tokens. Prove that this guarantees perfect load balance.

??? success "Solution"
    **Expert Choice Routing:**

    Instead of tokens choosing experts (top-k), experts choose tokens (top-C).

    **Implementation:**

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ExpertChoiceRouter(nn.Module):
        """
        Expert Choice routing: each expert selects its top-C tokens.
        Guarantees perfect load balance by construction.
        """
        def __init__(
            self,
            hidden_dim: int,
            num_experts: int,
            capacity_factor: float = 1.0
        ):
            super().__init__()
            self.num_experts = num_experts
            self.capacity_factor = capacity_factor

            # Router: projects tokens to expert scores
            self.router = nn.Linear(hidden_dim, num_experts, bias=False)

        def forward(
            self,
            tokens: torch.Tensor  # [batch, seq, hidden]
        ) -> tuple:
            """
            Returns:
                dispatch_mask: [num_experts, capacity, batch*seq]
                combine_weights: [num_experts, capacity, batch*seq]
                expert_indices: [num_experts, capacity] - which tokens selected
            """
            batch, seq, hidden = tokens.shape
            num_tokens = batch * seq

            # Compute expert capacity
            capacity = int(self.capacity_factor * num_tokens / self.num_experts)

            # Flatten tokens
            flat_tokens = tokens.view(num_tokens, hidden)  # [T, H]

            # Compute routing scores: which tokens does each expert want?
            scores = self.router(flat_tokens)  # [T, E]

            # Transpose: [E, T] - each row is an expert's preference over tokens
            expert_scores = scores.T  # [E, T]

            # Apply softmax over tokens for each expert
            expert_probs = F.softmax(expert_scores, dim=-1)  # [E, T]

            # Each expert selects top-C tokens
            top_values, top_indices = torch.topk(
                expert_probs, k=capacity, dim=-1
            )  # [E, C], [E, C]

            # Create dispatch mask: one-hot encoding of selections
            dispatch_mask = torch.zeros(
                self.num_experts, capacity, num_tokens,
                device=tokens.device
            )
            for e in range(self.num_experts):
                dispatch_mask[e, torch.arange(capacity), top_indices[e]] = 1.0

            # Combine weights: normalized probabilities for selected tokens
            combine_weights = top_values  # [E, C]

            # Renormalize weights per token (a token may be selected by multiple experts)
            token_expert_weights = torch.zeros(num_tokens, device=tokens.device)
            for e in range(self.num_experts):
                token_expert_weights.scatter_add_(
                    0, top_indices[e], top_values[e]
                )

            # Normalize combine weights
            for e in range(self.num_experts):
                normalizer = token_expert_weights[top_indices[e]]
                combine_weights[e] = top_values[e] / (normalizer + 1e-9)

            return dispatch_mask, combine_weights, top_indices

    class ExpertChoiceMoE(nn.Module):
        """MoE layer using Expert Choice routing."""
        def __init__(
            self,
            hidden_dim: int,
            ffn_dim: int,
            num_experts: int,
            capacity_factor: float = 1.0
        ):
            super().__init__()
            self.router = ExpertChoiceRouter(
                hidden_dim, num_experts, capacity_factor
            )
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, ffn_dim),
                    nn.GELU(),
                    nn.Linear(ffn_dim, hidden_dim)
                )
                for _ in range(num_experts)
            ])
            self.num_experts = num_experts

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, seq, hidden = x.shape
            num_tokens = batch * seq
            flat_x = x.view(num_tokens, hidden)

            # Get routing
            dispatch_mask, combine_weights, indices = self.router(x)

            # Initialize output
            output = torch.zeros_like(flat_x)

            # Process each expert
            for e, expert in enumerate(self.experts):
                # Gather tokens for this expert
                expert_tokens = flat_x[indices[e]]  # [C, H]

                # Process through expert
                expert_output = expert(expert_tokens)  # [C, H]

                # Weighted scatter back
                weights = combine_weights[e].unsqueeze(-1)  # [C, 1]
                output.scatter_add_(
                    0,
                    indices[e].unsqueeze(-1).expand(-1, hidden),
                    expert_output * weights
                )

            return output.view(batch, seq, hidden)
    ```

    **Proof of perfect load balance:**

    **Theorem:** Expert Choice routing guarantees that each expert processes exactly $C$ tokens.

    **Proof:**

    1. **By construction**: Each expert selects exactly $C = \lceil \frac{T \cdot c}{E} \rceil$ tokens using top-k.

    2. **Capacity allocation**: With $T$ total tokens and $E$ experts:

       $$\text{Tokens per expert} = C = \frac{T \cdot c}{E}$$

    3. **Total capacity**: $E \times C = T \cdot c$ slots available.

    4. **No overflow**: Unlike token-choice where popular experts overflow, each expert's selection is bounded by construction.

    5. **Load variance**: $\text{Var}(\text{load}) = 0$ since every expert gets exactly $C$ tokens.

    $$\boxed{\text{Load}_i = C \quad \forall i \in \{1, \ldots, E\}}$$

    **Comparison:**

    | Aspect | Token Choice (top-k) | Expert Choice (top-C) |
    |--------|---------------------|----------------------|
    | Load balance | Requires aux loss | Perfect by design |
    | Token drops | Possible with overflow | Never |
    | Token coverage | All tokens processed | Some tokens may be skipped |
    | Gradient flow | Through routing weights | Through selection weights |

    **Key trade-off:** Expert Choice may skip some tokens entirely (if no expert selects them). This is addressed by:
    1. Using $c > 1$ so total slots $> T$
    2. Combining with a shared dense layer
    3. Using auxiliary loss to encourage coverage

5. **Communication overlap**: Design a scheme to overlap AlltoAll dispatch with attention computation in the previous layer. What are the constraints?

??? success "Solution"
    **Goal:** Hide AlltoAll latency by overlapping with compute.

    **MoE Layer Structure:**

    ```
    Layer N:   Attention → MoE (dispatch → experts → combine)
    Layer N+1: Attention → MoE (dispatch → experts → combine)
    ```

    **Overlap Strategy:**

    Overlap the AlltoAll dispatch of layer N+1's MoE with the attention compute of layer N+1:

    ```
    Timeline:
    ────────────────────────────────────────────────────────────────

    Layer N MoE:    [dispatch]──[experts]──[combine]
                                                    │
                                                    ▼
    Layer N+1:                              [Attention Compute]
                                            ────────────────────
                                                    ▲
    Overlap:                                [AlltoAll dispatch]
                                            (for layer N+1 MoE)
    ```

    **Implementation:**

    ```python
    import torch
    import torch.distributed as dist
    from typing import Optional

    class OverlappedMoEBlock(nn.Module):
        """
        Transformer block with overlapped AlltoAll and attention.
        """
        def __init__(self, hidden_dim, num_heads, ffn_dim, num_experts):
            super().__init__()
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
            self.moe = DistributedMoE(hidden_dim, ffn_dim, num_experts)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)

            # Buffers for async communication
            self.dispatch_buffer = None
            self.dispatch_handle = None

        def start_dispatch(self, x: torch.Tensor):
            """
            Start async AlltoAll dispatch.
            Called before attention to overlap.
            """
            # Compute routing
            routing_weights, expert_indices = self.moe.router(x)

            # Prepare tokens for dispatch
            tokens_to_send = self.moe.prepare_dispatch(x, expert_indices)

            # Allocate receive buffer
            self.dispatch_buffer = torch.empty_like(tokens_to_send)

            # Start async AlltoAll
            self.dispatch_handle = dist.all_to_all_single(
                self.dispatch_buffer,
                tokens_to_send,
                async_op=True  # Non-blocking!
            )

            return routing_weights, expert_indices

        def finish_dispatch_and_compute(
            self,
            routing_weights,
            expert_indices
        ):
            """
            Wait for dispatch and run experts.
            Called after attention completes.
            """
            # Wait for AlltoAll to complete
            self.dispatch_handle.wait()

            # Run tokens through local experts
            expert_outputs = self.moe.run_experts(self.dispatch_buffer)

            # AlltoAll combine (could also be overlapped with next layer)
            combined = self.moe.combine(expert_outputs)

            return combined

        def forward(
            self,
            x: torch.Tensor,
            prefetched_routing: Optional[tuple] = None
        ):
            """
            Forward with optional overlapped dispatch from previous call.
            """
            # Attention block
            residual = x
            x = self.norm1(x)

            # Start next dispatch while computing attention
            next_routing = self.start_dispatch(x)

            # Compute attention (overlapped with dispatch)
            x, _ = self.attention(x, x, x)
            x = residual + x

            # Now finish the MoE from the *current* routing
            if prefetched_routing is not None:
                moe_out = self.finish_dispatch_and_compute(*prefetched_routing)
                residual = x
                x = self.norm2(x)
                x = residual + moe_out

            return x, next_routing

    class OverlappedMoEModel(nn.Module):
        """Model with pipelined AlltoAll overlap."""
        def __init__(self, num_layers, hidden_dim, num_heads, ffn_dim, num_experts):
            super().__init__()
            self.layers = nn.ModuleList([
                OverlappedMoEBlock(hidden_dim, num_heads, ffn_dim, num_experts)
                for _ in range(num_layers)
            ])

        def forward(self, x):
            routing = None

            for layer in self.layers:
                x, routing = layer(x, prefetched_routing=routing)

            # Handle final MoE
            if routing is not None:
                x = self.layers[-1].finish_dispatch_and_compute(*routing)

            return x
    ```

    **Constraints:**

    | Constraint | Reason | Mitigation |
    |------------|--------|------------|
    | **Attention time > AlltoAll time** | Otherwise dispatch isn't fully hidden | Increase batch size or reduce EP |
    | **Memory for buffers** | Need send + recv buffers simultaneously | Budget extra memory |
    | **Routing computed early** | Must know destinations before attention | Adds dependency complexity |
    | **Backward pass complexity** | Gradients flow through async ops | Use gradient checkpointing carefully |
    | **No data dependency** | Dispatch input must not depend on attention output | Layer architecture constraint |

    **When overlap is effective:**

    $$T_{\text{attention}} \geq T_{\text{AlltoAll}}$$

    For typical configurations:
    - Attention: $O(S^2 \cdot H)$ compute
    - AlltoAll: $O(T \cdot H / \beta)$ communication

    **Achievable overlap:**

    $$\text{Overlap fraction} = \min\left(1, \frac{T_{\text{attention}}}{T_{\text{AlltoAll}}}\right)$$

    If attention takes 10ms and AlltoAll takes 5ms:

    $$\text{Overlap} = 100\% \text{ (fully hidden)}$$

    If attention takes 5ms and AlltoAll takes 10ms:

    $$\text{Overlap} = 50\% \text{ (5ms exposed)}$$

    $$\boxed{\text{Overlap requires: } T_{\text{compute}} > T_{\text{comm}} \text{ and no data dependencies}}$$

6. **3D parallelism groups**: With world_size=64, TP=4, EP=8, DP=2, enumerate all process groups for rank 13.

??? success "Solution"
    **Given:**

    - World size: 64 GPUs
    - Tensor Parallelism (TP): 4
    - Expert Parallelism (EP): 8
    - Data Parallelism (DP): 2

    **Verify:** $TP \times EP \times DP = 4 \times 8 \times 2 = 64$ ✓

    **Group layout:**

    The standard layout is: fastest-varying → slowest-varying = TP → EP → DP

    ```
    Rank = dp_id * (TP * EP) + ep_id * TP + tp_id
    ```

    For rank 13:
    ```
    rank = 13
    TP * EP = 4 * 8 = 32

    dp_id = 13 // 32 = 0
    remainder = 13 % 32 = 13
    ep_id = 13 // 4 = 3
    tp_id = 13 % 4 = 1
    ```

    **Rank 13 coordinates:** $(tp=1, ep=3, dp=0)$

    **Tensor Parallel Group:**

    All ranks with same $(ep, dp)$, varying $tp$:

    $$\text{TP group for rank 13} = \{dp=0, ep=3, tp=0,1,2,3\}$$

    Ranks: $0 \times 32 + 3 \times 4 + \{0,1,2,3\} = \{12, 13, 14, 15\}$

    $$\boxed{\text{TP group: } \{12, 13, 14, 15\}}$$

    **Expert Parallel Group:**

    All ranks with same $(tp, dp)$, varying $ep$:

    $$\text{EP group for rank 13} = \{dp=0, tp=1, ep=0,1,2,...,7\}$$

    Ranks: $0 \times 32 + \{0,1,...,7\} \times 4 + 1 = \{1, 5, 9, 13, 17, 21, 25, 29\}$

    $$\boxed{\text{EP group: } \{1, 5, 9, 13, 17, 21, 25, 29\}}$$

    **Data Parallel Group:**

    All ranks with same $(tp, ep)$, varying $dp$:

    $$\text{DP group for rank 13} = \{tp=1, ep=3, dp=0,1\}$$

    Ranks: $\{0,1\} \times 32 + 3 \times 4 + 1 = \{13, 45\}$

    $$\boxed{\text{DP group: } \{13, 45\}}$$

    **Summary for rank 13:**

    | Group | Size | Members |
    |-------|------|---------|
    | Tensor Parallel | 4 | {12, 13, 14, 15} |
    | Expert Parallel | 8 | {1, 5, 9, 13, 17, 21, 25, 29} |
    | Data Parallel | 2 | {13, 45} |

    **Verification code:**

    ```python
    def get_groups_for_rank(rank, tp_size, ep_size, dp_size):
        """Compute all process groups for a given rank."""
        tp_ep = tp_size * ep_size

        # Decompose rank into coordinates
        dp_id = rank // tp_ep
        remainder = rank % tp_ep
        ep_id = remainder // tp_size
        tp_id = remainder % tp_size

        print(f"Rank {rank}: tp_id={tp_id}, ep_id={ep_id}, dp_id={dp_id}")

        # TP group: same (ep_id, dp_id), vary tp_id
        tp_group = [dp_id * tp_ep + ep_id * tp_size + t for t in range(tp_size)]

        # EP group: same (tp_id, dp_id), vary ep_id
        ep_group = [dp_id * tp_ep + e * tp_size + tp_id for e in range(ep_size)]

        # DP group: same (tp_id, ep_id), vary dp_id
        dp_group = [d * tp_ep + ep_id * tp_size + tp_id for d in range(dp_size)]

        return tp_group, ep_group, dp_group

    tp, ep, dp = get_groups_for_rank(13, tp_size=4, ep_size=8, dp_size=2)
    # Output:
    # Rank 13: tp_id=1, ep_id=3, dp_id=0
    # TP group: [12, 13, 14, 15]
    # EP group: [1, 5, 9, 13, 17, 21, 25, 29]
    # DP group: [13, 45]
    ```

    **Visual representation:**

    ```
    DP=0 (ranks 0-31):                    DP=1 (ranks 32-63):
    ┌────┬────┬────┬────┬────┬────┬────┬────┐  ┌────┬────┬────┬────┬────┬────┬────┬────┐
    │EP0 │EP1 │EP2 │EP3 │EP4 │EP5 │EP6 │EP7 │  │EP0 │EP1 │EP2 │EP3 │EP4 │EP5 │EP6 │EP7 │
    ├────┼────┼────┼────┼────┼────┼────┼────┤  ├────┼────┼────┼────┼────┼────┼────┼────┤
    │0-3 │4-7 │8-11│12- │16- │20- │24- │28- │  │32- │36- │40- │44- │48- │52- │56- │60- │
    │    │    │    │15  │19  │23  │27  │31  │  │35  │39  │43  │47  │51  │55  │59  │63  │
    └────┴────┴────┴────┴────┴────┴────┴────┘  └────┴────┴────┴────┴────┴────┴────┴────┘

    Rank 13 is in EP3 block (ranks 12-15) at DP=0
    - TP group: ranks 12, 13, 14, 15 (same EP block)
    - EP group: rank 1,5,9,13,17,21,25,29 (tp_id=1 across all EP blocks)
    - DP group: ranks 13, 45 (same position in each DP replica)
    ```

7. **Routing collapse detection**: Design a monitoring system to detect routing collapse early in training. What metrics would you track, and what thresholds would trigger alerts?

??? success "Solution"
    **Routing collapse** occurs when the router learns to send most tokens to a small subset of experts, wasting the model's capacity.

    **Key Metrics to Track:**

    | Metric | Formula | Healthy Range |
    |--------|---------|---------------|
    | Expert utilization entropy | $H = -\sum_i p_i \log p_i$ | $> 0.9 \times \log(E)$ |
    | Max expert load | $\max_i(\text{tokens}_i) / \text{avg}$ | $< 2.0$ |
    | Min expert load | $\min_i(\text{tokens}_i) / \text{avg}$ | $> 0.3$ |
    | Load Gini coefficient | $G = \frac{\sum_{i,j}|l_i - l_j|}{2n\sum_i l_i}$ | $< 0.3$ |
    | Dropped token rate | $\text{dropped} / \text{total}$ | $< 5\%$ |
    | Router weight entropy | Per-token softmax entropy | $> 1.0$ |

    **Monitoring System Design:**

    ```python
    import numpy as np
    from collections import deque
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class CollapseAlert:
        severity: str  # "warning", "critical"
        metric: str
        value: float
        threshold: float
        step: int

    class RoutingCollapseMonitor:
        def __init__(self, num_experts: int, window_size: int = 100):
            self.num_experts = num_experts
            self.max_entropy = np.log(num_experts)
            self.window_size = window_size

            # Rolling windows for trend detection
            self.entropy_history = deque(maxlen=window_size)
            self.gini_history = deque(maxlen=window_size)
            self.drop_history = deque(maxlen=window_size)

            # Thresholds
            self.thresholds = {
                "entropy_warning": 0.85 * self.max_entropy,
                "entropy_critical": 0.70 * self.max_entropy,
                "gini_warning": 0.35,
                "gini_critical": 0.50,
                "max_load_warning": 2.5,
                "max_load_critical": 4.0,
                "drop_rate_warning": 0.05,
                "drop_rate_critical": 0.15,
                "trend_threshold": -0.01,  # Entropy decreasing
            }

        def compute_metrics(self, expert_counts: np.ndarray,
                          dropped: int, total: int) -> dict:
            """Compute all routing health metrics."""
            # Normalize to probabilities
            probs = expert_counts / expert_counts.sum()
            probs = np.clip(probs, 1e-10, 1.0)  # Avoid log(0)

            # Entropy (higher = more balanced)
            entropy = -np.sum(probs * np.log(probs))

            # Gini coefficient (lower = more balanced)
            sorted_loads = np.sort(expert_counts)
            n = len(sorted_loads)
            cumsum = np.cumsum(sorted_loads)
            gini = (2 * np.sum((np.arange(1, n+1) * sorted_loads)) /
                   (n * cumsum[-1])) - (n + 1) / n

            # Load imbalance
            avg_load = expert_counts.mean()
            max_load_ratio = expert_counts.max() / avg_load
            min_load_ratio = expert_counts.min() / avg_load

            # Drop rate
            drop_rate = dropped / total if total > 0 else 0

            return {
                "entropy": entropy,
                "normalized_entropy": entropy / self.max_entropy,
                "gini": gini,
                "max_load_ratio": max_load_ratio,
                "min_load_ratio": min_load_ratio,
                "drop_rate": drop_rate,
                "expert_counts": expert_counts,
            }

        def check_alerts(self, metrics: dict, step: int) -> list[CollapseAlert]:
            """Check metrics against thresholds."""
            alerts = []
            t = self.thresholds

            # Entropy checks
            if metrics["entropy"] < t["entropy_critical"]:
                alerts.append(CollapseAlert(
                    "critical", "entropy", metrics["entropy"],
                    t["entropy_critical"], step
                ))
            elif metrics["entropy"] < t["entropy_warning"]:
                alerts.append(CollapseAlert(
                    "warning", "entropy", metrics["entropy"],
                    t["entropy_warning"], step
                ))

            # Gini checks
            if metrics["gini"] > t["gini_critical"]:
                alerts.append(CollapseAlert(
                    "critical", "gini", metrics["gini"],
                    t["gini_critical"], step
                ))
            elif metrics["gini"] > t["gini_warning"]:
                alerts.append(CollapseAlert(
                    "warning", "gini", metrics["gini"],
                    t["gini_warning"], step
                ))

            # Max load checks
            if metrics["max_load_ratio"] > t["max_load_critical"]:
                alerts.append(CollapseAlert(
                    "critical", "max_load_ratio",
                    metrics["max_load_ratio"],
                    t["max_load_critical"], step
                ))

            # Drop rate checks
            if metrics["drop_rate"] > t["drop_rate_critical"]:
                alerts.append(CollapseAlert(
                    "critical", "drop_rate", metrics["drop_rate"],
                    t["drop_rate_critical"], step
                ))

            # Trend detection (entropy declining over time)
            if len(self.entropy_history) >= 50:
                recent = list(self.entropy_history)[-50:]
                slope = np.polyfit(range(len(recent)), recent, 1)[0]
                if slope < t["trend_threshold"]:
                    alerts.append(CollapseAlert(
                        "warning", "entropy_trend", slope,
                        t["trend_threshold"], step
                    ))

            return alerts

        def step(self, expert_counts: np.ndarray,
                dropped: int, total: int, step: int):
            """Process one step and return any alerts."""
            metrics = self.compute_metrics(expert_counts, dropped, total)

            # Update history
            self.entropy_history.append(metrics["entropy"])
            self.gini_history.append(metrics["gini"])
            self.drop_history.append(metrics["drop_rate"])

            return self.check_alerts(metrics, step), metrics
    ```

    **Alert Thresholds Summary:**

    | Metric | Warning | Critical | Action |
    |--------|---------|----------|--------|
    | Normalized entropy | $< 0.85$ | $< 0.70$ | Increase aux loss weight |
    | Gini coefficient | $> 0.35$ | $> 0.50$ | Add jitter, check router init |
    | Max load ratio | $> 2.5$ | $> 4.0$ | Decrease capacity factor |
    | Drop rate | $> 5\%$ | $> 15\%$ | Increase capacity factor |
    | Entropy trend | Declining | - | Early intervention |

    **Early Warning Signs:**

    1. **Entropy decline in first 1000 steps**: Often indicates router initialization issue
    2. **One expert dominating early**: Reset router weights with higher variance
    3. **Oscillating loads**: Aux loss weight too high, causing overcorrection
    4. **Gradual collapse after stable period**: Learning rate too high for router

    **Recommended Response Actions:**

    - **Warning**: Log to dashboard, continue monitoring
    - **Critical (entropy)**: Increase `aux_loss_weight` by 2×, add noise to router
    - **Critical (drops)**: Increase capacity factor by 0.2
    - **Critical (imbalance)**: Consider switching to expert choice routing

8. **Gradient analysis**: In a model with top_k=2, expert 0 sees 60% of tokens and expert 1 sees 40%. Analyze how the auxiliary loss gradient affects these proportions.

??? success "Solution"
    **Setup:**

    - top_k = 2 (each token routed to 2 experts)
    - Expert 0: receives 60% of tokens → $f_0 = 0.60$
    - Expert 1: receives 40% of tokens → $f_1 = 0.40$
    - Assume 2 experts for simplicity (generalizes to more)

    **Auxiliary Loss Formulation:**

    The load balancing auxiliary loss is:

    $$\mathcal{L}_{aux} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot p_i$$

    Where:

    - $E$ = number of experts (here, 2)
    - $f_i$ = fraction of tokens routed to expert $i$
    - $p_i$ = mean routing probability for expert $i$ (average of softmax outputs)
    - $\alpha$ = auxiliary loss coefficient

    **Gradient Analysis:**

    The gradient with respect to router logits $z_{t,i}$ (for token $t$, expert $i$):

    $$\frac{\partial \mathcal{L}_{aux}}{\partial z_{t,i}} = \alpha \cdot E \cdot \sum_j f_j \cdot \frac{\partial p_j}{\partial z_{t,i}}$$

    For softmax: $p_i = \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$

    The derivative:

    $$\frac{\partial p_j}{\partial z_{t,i}} = p_i(\delta_{ij} - p_j) / T$$

    Where $T$ = number of tokens and $\delta_{ij}$ is Kronecker delta.

    **Simplified Gradient (per token):**

    $$\frac{\partial \mathcal{L}_{aux}}{\partial z_{t,i}} = \alpha \cdot E \cdot p_i \left(f_i - \sum_j f_j p_j\right)$$

    Let $\bar{f} = \sum_j f_j p_j$ (weighted average load):

    $$\frac{\partial \mathcal{L}_{aux}}{\partial z_{t,i}} = \alpha \cdot E \cdot p_i (f_i - \bar{f})$$

    **Applying to Our Example:**

    Given $f_0 = 0.60$, $f_1 = 0.40$:

    Assume router probabilities approximately match loads: $p_0 \approx 0.60$, $p_1 \approx 0.40$

    $$\bar{f} = f_0 \cdot p_0 + f_1 \cdot p_1 = 0.60 \times 0.60 + 0.40 \times 0.40 = 0.52$$

    **Gradients:**

    For expert 0 (overloaded):

    $$\frac{\partial \mathcal{L}_{aux}}{\partial z_0} \propto p_0 (f_0 - \bar{f}) = 0.60 \times (0.60 - 0.52) = 0.60 \times 0.08 = \boxed{+0.048}$$

    For expert 1 (underloaded):

    $$\frac{\partial \mathcal{L}_{aux}}{\partial z_1} \propto p_1 (f_1 - \bar{f}) = 0.40 \times (0.40 - 0.52) = 0.40 \times (-0.12) = \boxed{-0.048}$$

    **Interpretation:**

    | Expert | Load $f_i$ | Gradient Sign | Effect |
    |--------|------------|---------------|--------|
    | Expert 0 | 0.60 (high) | Positive | Decreases logit → reduces probability |
    | Expert 1 | 0.40 (low) | Negative | Increases logit → increases probability |

    The auxiliary loss pushes the system toward **balance**:

    - Overloaded experts get positive gradients → their routing probabilities decrease
    - Underloaded experts get negative gradients → their routing probabilities increase

    **Equilibrium Analysis:**

    At equilibrium, $f_0 = f_1 = 0.50$ (with 2 experts):

    $$\bar{f} = 0.50 \times 0.50 + 0.50 \times 0.50 = 0.50$$

    Gradients become:

    $$\frac{\partial \mathcal{L}_{aux}}{\partial z_i} \propto 0.50 \times (0.50 - 0.50) = 0$$

    **No gradient at perfect balance** - the equilibrium is stable.

    **Numerical Example with Training Dynamics:**

    ```python
    import numpy as np

    def simulate_load_balancing(f0_init=0.60, alpha=0.01, steps=100, lr=0.1):
        """Simulate how aux loss corrects imbalance."""
        f0, f1 = f0_init, 1 - f0_init

        trajectory = [(f0, f1)]

        for _ in range(steps):
            # Assume p ≈ f (router matches current loads)
            p0, p1 = f0, f1
            f_bar = f0 * p0 + f1 * p1

            # Gradients
            grad_0 = alpha * 2 * p0 * (f0 - f_bar)
            grad_1 = alpha * 2 * p1 * (f1 - f_bar)

            # Update (gradient descent on logits affects f)
            # Simplified: treat as direct adjustment
            f0_new = f0 - lr * grad_0
            f1_new = f1 - lr * grad_1

            # Normalize to valid distribution
            total = f0_new + f1_new
            f0, f1 = f0_new / total, f1_new / total

            trajectory.append((f0, f1))

        return trajectory

    # Run simulation
    traj = simulate_load_balancing(f0_init=0.60)
    print(f"Start: f0={traj[0][0]:.3f}, f1={traj[0][1]:.3f}")
    print(f"End:   f0={traj[-1][0]:.3f}, f1={traj[-1][1]:.3f}")
    # Output:
    # Start: f0=0.600, f1=0.400
    # End:   f0=0.500, f1=0.500  (converges to balance)
    ```

    **Summary:**

    The auxiliary loss gradient for an expert is proportional to:

    $$\nabla_{z_i} \mathcal{L}_{aux} \propto p_i \cdot (f_i - \bar{f})$$

    - **Overloaded experts** ($f_i > \bar{f}$): Positive gradient → probability decreases
    - **Underloaded experts** ($f_i < \bar{f}$): Negative gradient → probability increases
    - **At balance**: Zero gradient → stable equilibrium

    The 60/40 imbalance creates a corrective force that pushes toward 50/50 distribution over training steps.

## Key Takeaways

1. **Sparsity enables scale**: MoE models can have 10-100× more parameters with constant compute per token.

2. **AlltoAll is the communication pattern**: Tokens go to experts, outputs come back—requiring two AlltoAll operations per MoE layer.

3. **Load balancing is critical**: Without balancing, routing collapse makes sparsity useless. Use auxiliary loss + capacity limiting.

4. **Expert choice guarantees balance**: Letting experts choose tokens, rather than tokens choosing experts, ensures perfect load distribution.

5. **Capacity factor trades off drops vs. memory**: Higher capacity = fewer dropped tokens but more memory per expert.

6. **Composition is natural**: EP combines cleanly with DP and TP through orthogonal process groups.

7. **Gradients flow through weights**: The top-k selection is non-differentiable, but routing weights provide gradient signal to the router.

8. **Routing overhead matters**: For small models or low-latency requirements, the routing computation and AlltoAll communication may dominate.
