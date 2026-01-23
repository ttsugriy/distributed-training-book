---
title: "Case Study: LLaMA 3"
subtitle: "Deconstructing Meta's Training Infrastructure"
---

<div class="chapter-opener" markdown>
LLaMA 3 represents the state of the art in open-weight large language models. By analyzing its training infrastructure through the lens of this book's frameworks, we can understand why specific choices were made and how the mathematics of distributed training shaped the final system.
</div>

<div class="investigation-question" markdown>
**The Question**: How did Meta train a 405B parameter model on 16,000 H100 GPUs across 15 trillion tokens? What constraints drove their parallelism choices, and how do these align with the theoretical frameworks we've developed?
</div>

## The Scale of the Challenge

LLaMA 3 405B required unprecedented scale:

| Metric | Value |
|--------|-------|
| Parameters | 405 billion |
| Training tokens | 15.6 trillion |
| GPUs | 16,384 H100s |
| Training time | ~54 days |
| Estimated FLOPs | ~$3.8 \times 10^{25}$ |

Let's derive these numbers and understand the constraints.

### Compute Requirements

Using the Chinchilla optimal scaling (Chapter 8):

$$C = 6ND$$

where:

- $N = 405 \times 10^9$ parameters
- $D = 15.6 \times 10^{12}$ tokens

$$C = 6 \times (405 \times 10^9) \times (15.6 \times 10^{12}) \approx 3.79 \times 10^{25} \text{ FLOPs}$$

### Hardware Capacity

Each H100 SXM provides:

- Peak FP16/BF16: 1,979 TFLOPS
- Peak FP8: 3,958 TFLOPS
- HBM3 Memory: 80 GB
- Memory Bandwidth: 3.35 TB/s

For 16,384 GPUs at realistic 40% MFU (typical for very large models):

$$\text{Effective FLOPS} = 16384 \times 1979 \times 10^{12} \times 0.40 \approx 1.3 \times 10^{19} \text{ FLOPS}$$

### Training Time Derivation

$$\text{Training time} = \frac{C}{\text{Effective FLOPS}} = \frac{3.79 \times 10^{25}}{1.3 \times 10^{19}} \approx 2.9 \times 10^6 \text{ seconds} \approx 34 \text{ days}$$

The actual training took approximately 54 days, suggesting:

- Effective MFU closer to 25-30%
- Time lost to failures and restarts
- Validation and checkpoint overhead

## Memory Analysis

### Model Memory Requirements

Applying the memory equation from Chapter 19:

**Model parameters (BF16)**:
$$M_{\text{params}} = 405 \times 10^9 \times 2 \text{ bytes} = 810 \text{ GB}$$

**Optimizer state (AdamW)**:
$$M_{\text{opt}} = 405 \times 10^9 \times (4 + 4) \text{ bytes} = 3,240 \text{ GB}$$

**Gradients (BF16)**:
$$M_{\text{grad}} = 405 \times 10^9 \times 2 \text{ bytes} = 810 \text{ GB}$$

**Total without sharding**: 4,860 GB (61 GPUs minimum!)

### Why Full Sharding is Essential

With 80 GB per GPU, we need at least:

$$\text{GPUs for parameters} = \frac{4860}{80} \approx 61$$

But this leaves no room for activations. LLaMA 3 uses **4D parallelism**:

1. **Tensor Parallelism (TP)**: 8-way within node
2. **Pipeline Parallelism (PP)**: 16 stages
3. **Data Parallelism (DP)**: Remaining GPUs
4. **Context Parallelism (CP)**: For long sequences

$$\text{Total GPUs} = TP \times PP \times DP \times CP = 8 \times 16 \times 128 \times 1 = 16,384$$

## Architecture Analysis

LLaMA 3 405B architecture:

| Component | Value |
|-----------|-------|
| Layers | 126 |
| Hidden dimension | 16,384 |
| FFN dimension | 53,248 |
| Attention heads | 128 |
| KV heads | 8 |
| Vocabulary | 128,256 |
| Context length | 8,192 (extended to 128K) |

### Per-Layer Memory

**Attention block**:
$$M_{\text{attn}} = 4 \times d_{\text{model}}^2 = 4 \times 16384^2 \times 2 = 2.15 \text{ GB}$$

Wait—this uses GQA with 8 KV heads, so:
$$M_{QKV} = d_{\text{model}} \times (d_{\text{model}} + 2 \times \frac{d_{\text{model}} \times n_{\text{kv}}}{n_{\text{heads}}}) \times 2$$
$$= 16384 \times (16384 + 2 \times \frac{16384 \times 8}{128}) \times 2 = 16384 \times (16384 + 2048) \times 2 \approx 0.61 \text{ GB}$$

**FFN block** (SwiGLU has 3 matrices):
$$M_{\text{ffn}} = 3 \times d_{\text{model}} \times d_{\text{ffn}} \times 2 = 3 \times 16384 \times 53248 \times 2 \approx 5.24 \text{ GB}$$

**Per layer total**: ~6 GB

**All 126 layers**: ~756 GB (just for layer weights!)

### Activation Memory

For a single forward pass with batch size $B$ and sequence length $S$:

$$M_{\text{act}} = L \times (2 \times B \times S \times d_{\text{model}} + B \times S \times d_{\text{ffn}})$$

With $B=1$, $S=8192$, $L=126$:
$$M_{\text{act}} = 126 \times (2 \times 8192 \times 16384 + 8192 \times 53248) \times 2 \text{ bytes}$$
$$\approx 126 \times (268M + 436M) \times 2 \approx 177 \text{ GB per sequence}$$

This explains why activation checkpointing is essential.

## The Parallelism Configuration

### Tensor Parallelism: 8-way

LLaMA 3 uses TP=8, matching the 8 GPUs per node connected via NVLink.

**Why 8?** Applying the alpha-beta analysis from Chapter 4:

For TP across NVLink (600 GB/s bidirectional):
$$T_{\text{comm}} = \alpha + \frac{4 \times B \times S \times d_{\text{model}}}{8 \times \beta}$$

For $B=1$, $S=8192$, $d=16384$:
$$\text{Data per AllReduce} = 4 \times 1 \times 8192 \times 16384 = 537 \text{ MB}$$

With ring AllReduce:
$$T_{\text{NVLink}} = \frac{2 \times (8-1)/8 \times 537\text{MB}}{600 \text{ GB/s}} \approx 1.6 \text{ ms}$$

If we extended TP to 16 (across nodes via IB):
$$T_{\text{IB}} = \frac{2 \times (16-1)/16 \times 537\text{MB}}{50 \text{ GB/s}} \approx 20 \text{ ms}$$

The 12× slowdown from crossing the node boundary makes TP>8 prohibitive.

### Pipeline Parallelism: 16 Stages

With 126 layers and 16 pipeline stages:
$$\text{Layers per stage} = \lceil 126/16 \rceil = 8 \text{ layers}$$

**Memory per stage**:
$$M_{\text{stage}} = 8 \times 6\text{ GB} = 48 \text{ GB}$$

This leaves ~32 GB for activations, optimizer states (sharded), and working memory.

**Pipeline bubble analysis** (Chapter 16):

For 1F1B schedule with microbatch count $m$:
$$\text{Bubble fraction} = \frac{p-1}{m}$$

With $p=16$ and $m=64$ microbatches:
$$\text{Bubble} = \frac{15}{64} \approx 23\%$$

This represents a significant efficiency loss, but is necessary for memory constraints.

### Data Parallelism: FSDP (ZeRO-3)

The remaining dimension is sharded data parallelism:

$$DP = \frac{16384}{8 \times 16} = 128$$

**FSDP shards**:

- Model parameters: 810 GB / 128 = 6.3 GB per rank
- Optimizer states: 3,240 GB / 128 = 25.3 GB per rank
- Gradients: 810 GB / 128 = 6.3 GB per rank

**Total sharded state**: ~38 GB per rank

**AllGather cost per layer** (FSDP reconstructs before compute):

For one layer (~6 GB / TP):
$$\text{Data per AllGather} = \frac{6\text{ GB}}{8} = 750 \text{ MB}$$

Across 128 DP ranks (using hierarchical NCCL):
$$T_{\text{AllGather}} = \alpha \log_2(128) + \frac{127/128 \times 750\text{MB}}{50 \text{ GB/s}} \approx 15 \text{ ms}$$

With 126 layers and overlap:
$$\text{Theoretical comm time} = 126 \times 15 \text{ ms} = 1.89 \text{ s}$$

This must be overlapped with compute to achieve reasonable efficiency.

## Communication Analysis

### Communication Volumes

Let's compute total communication per step:

**Tensor Parallelism** (per layer, 2 AllReduce):
$$V_{\text{TP}} = 126 \times 2 \times 2 \times \frac{7}{8} \times B \times S \times d / TP$$
$$= 126 \times 2 \times 2 \times 0.875 \times B \times S \times 16384 / 8 \times 2 \text{ bytes}$$

For $BS = 1M$ tokens (distributed):
$$V_{\text{TP}} = 126 \times 4 \times 0.875 \times \frac{1M}{128} \times 2048 \times 2 \approx 14.2 \text{ GB per rank}$$

**Pipeline Parallelism** (point-to-point):
$$V_{\text{PP}} = 2 \times \text{microbatches} \times B_{\mu} \times S \times d \times 2$$
$$= 2 \times 64 \times \text{tokens per } \mu\text{batch} \times 16384 \times 2$$

**Data Parallelism** (ReduceScatter + AllGather per layer):
$$V_{\text{DP}} = 2 \times \frac{127}{128} \times \frac{810\text{ GB}}{8} \approx 200 \text{ GB per rank}$$

### Bandwidth Requirements

For a step time of ~30 seconds (typical for very large batches):

$$\text{Required BW} = \frac{V_{\text{DP}}}{T_{\text{step}}} = \frac{200\text{ GB}}{30\text{ s}} \approx 6.7 \text{ GB/s}$$

This is well within the 50 GB/s IB capability, allowing significant overlap.

## Efficiency Analysis

### Model FLOPs Utilization

Computing MFU for LLaMA 3 405B:

**Forward FLOPs per token**:
$$F_{\text{fwd}} = 2 \times N \times 2 = 4N = 4 \times 405 \times 10^9 = 1.62 \times 10^{12}$$

(Factor of 2 for forward, factor of 2 for backward = 4× total, but we count 6× including backward activations)

**Per step with 1M tokens**:
$$F_{\text{step}} = 6 \times 405 \times 10^9 \times 10^6 = 2.43 \times 10^{18} \text{ FLOPs}$$

**Hardware peak**:
$$F_{\text{peak}} = 16384 \times 1979 \times 10^{12} = 3.24 \times 10^{19} \text{ FLOPS}$$

If step takes 30 seconds:
$$\text{MFU} = \frac{2.43 \times 10^{18}}{3.24 \times 10^{19} \times 30} \approx 0.25 = 25\%$$

### Efficiency Breakdown

Where does the efficiency go?

| Factor | Efficiency Loss |
|--------|----------------|
| Pipeline bubble | 23% |
| Communication overhead | 15% |
| Memory operations | 10% |
| Load imbalance | 5% |
| Failures/restarts | 5% |

Combined: $0.77 \times 0.85 \times 0.90 \times 0.95 \times 0.95 \approx 0.53$

Theoretical MFU with perfect scaling: ~50%
Actual: ~25-30%

## Training Stability Techniques

### Batch Size Dynamics

LLaMA 3 started with smaller batch sizes and ramped up:

| Training Phase | Batch Size (tokens) |
|---------------|---------------------|
| 0-8T tokens | 4M |
| 8T-15T tokens | 8M-16M |

**Why?** From Chapter 10, critical batch size increases during training:
$$B_{\text{crit}} \propto \frac{\text{tr}(\Sigma)}{\text{tr}(H)}$$

Early: high gradient noise (large $\Sigma$) → small $B_{\text{crit}}$
Late: low noise, smooth loss (small $\Sigma$) → large $B_{\text{crit}}$

### Learning Rate Schedule

LLaMA 3 uses:
1. Linear warmup: 2,000 steps
2. Cosine decay to 10% of peak

**Peak learning rate**: $1.5 \times 10^{-4}$

With AdamW:

- $\beta_1 = 0.9$, $\beta_2 = 0.95$
- Weight decay: $0.1$

### Numerical Stability

**Mixed precision strategy**:

- Weights: BF16
- Activations: BF16
- Gradients: BF16
- Optimizer states: FP32
- Loss scaling: Dynamic

**Why BF16 over FP16?** The larger exponent range (8 bits vs 5) prevents overflow during:

- Attention softmax with long sequences
- Large gradient magnitudes early in training
- Cross-entropy loss with large vocabulary

## Fault Tolerance

### Failure Rates

At 16,384 GPUs, failures are constant. Meta reported:

| Failure Type | MTBF (Mean Time Between Failures) |
|--------------|-----------------------------------|
| GPU hardware | ~1 failure/day |
| Network | ~2-3/day |
| Software (OOM, timeout) | ~5-10/day |

**Expected failures per training**:
$$N_{\text{failures}} = 54 \text{ days} \times 10 \text{ failures/day} = 540 \text{ failures}$$

### Checkpointing Strategy

**Checkpoint size**:
$$C_{\text{ckpt}} = 810\text{ GB (params)} + 3240\text{ GB (opt)} = 4.05 \text{ TB}$$

**Checkpoint frequency**: Every 1,000 steps

**Checkpoint time** (to parallel filesystem):
$$T_{\text{ckpt}} = \frac{4.05 \text{ TB}}{16384 \text{ GPUs} \times 5 \text{ GB/s}} \approx 50 \text{ seconds}$$

With asynchronous checkpointing, this overlaps with training.

**Recovery strategy**:
1. Detect failure (NCCL timeout, heartbeat failure)
2. Terminate affected nodes
3. Replace with spare nodes
4. Load most recent checkpoint
5. Resume training

**Time lost per failure**: ~5-10 minutes

### In-Memory Checkpointing

For faster recovery, LLaMA 3 uses in-memory replicated checkpointing:

```python
# Simplified in-memory checkpoint strategy
class ReplicatedCheckpoint:
    def __init__(self, model, optimizer, replicas=2):
        self.replicas = replicas
        # Each checkpoint is replicated across 2 nodes

    def save(self):
        # Save to local CPU memory
        state = {
            'model': gather_sharded_state(model),
            'optimizer': gather_sharded_state(optimizer),
            'step': current_step
        }

        # Replicate to partner node
        partner_rank = get_partner_rank(dist.get_rank())
        dist.send(state, partner_rank)

    def restore(self, failed_ranks):
        # If our checkpoint is valid, provide to failed ranks
        # If our checkpoint failed, request from partner
        pass
```

This enables recovery in seconds rather than minutes.

## Sequence Length Extension

LLaMA 3 was trained on 8K context, then extended to 128K.

### The Challenge

**Memory scaling with sequence length**:
$$M_{\text{attn}} = O(B \times L \times S^2)$$

For $S = 128K$:
$$M_{\text{attn}} = 126 \times 128 \times 8192 \times 128000^2 \times 2 / \text{TP} \rightarrow \text{Explodes!}$$

### Context Parallelism

Solution: Shard the sequence across GPUs.

With CP=4:
$$S_{\text{local}} = \frac{128000}{4} = 32000$$

Each GPU handles 32K tokens, with ring attention for cross-segment dependencies.

**Communication pattern** (Ring Attention):

```
GPU 0: Q[0:32K] × K[0:32K] → send K to GPU 1
GPU 1: Q[32K:64K] × K[0:32K] → send K to GPU 2
...
```

**Additional communication per layer**:
$$V_{\text{CP}} = 2 \times (CP-1) \times B \times S/CP \times d_k \times n_{kv}$$

For GQA with 8 KV heads:
$$V_{\text{CP}} = 2 \times 3 \times B \times 32000 \times 128 \times 8 \times 2 \approx 0.4 \text{ GB}$$

This is manageable with NVLink within the node.

### RoPE Extension

LLaMA 3 uses RoPE (Rotary Position Embeddings) which need adjustment for longer contexts:

$$\text{RoPE}(x, m) = x \odot \cos(m\theta) + \text{rotate}(x) \odot \sin(m\theta)$$

For positions beyond training length, the frequency is adjusted:
$$\theta_i' = \theta_i \times \text{scale}^{-2i/d}$$

where scale = 8 for 8K→128K extension.

## Lessons and Design Patterns

### Pattern 1: Hierarchical Parallelism

```
Within node (NVLink): TP=8
Across nodes (IB): PP=16, DP=128
```

**Principle**: High-bandwidth collectives (AllReduce for TP) stay within node; lower-bandwidth operations (PP point-to-point, DP with overlap) cross nodes.

### Pattern 2: Memory-Compute-Communication Balance

The configuration balances:

- **Memory**: FSDP keeps sharded state within 80GB limit
- **Compute**: Pipeline keeps all stages busy (except bubble)
- **Communication**: Overlap hides most DP communication

### Pattern 3: Progressive Scaling

Training parameters evolved:

- Batch size: Increased during training
- Learning rate: Warmup then decay
- Sequence length: Extended in final phase

### Pattern 4: Defense in Depth for Reliability

Multiple layers of fault tolerance:
1. Hardware monitoring and preemptive replacement
2. In-memory checkpointing for fast recovery
3. Periodic disk checkpoints for disaster recovery
4. Elastic training for node replacement

## Reproducing the Analysis

```python
class LLaMA3Analyzer:
    """Analyze LLaMA 3 training configuration."""

    def __init__(self):
        # Model parameters
        self.params = 405e9
        self.layers = 126
        self.d_model = 16384
        self.d_ffn = 53248
        self.n_heads = 128
        self.n_kv_heads = 8

        # Hardware
        self.gpus = 16384
        self.gpu_memory = 80  # GB
        self.nvlink_bw = 600  # GB/s
        self.ib_bw = 50  # GB/s
        self.peak_flops = 1979e12  # FP16

        # Parallelism
        self.tp = 8
        self.pp = 16
        self.dp = 128

    def model_memory(self):
        """Calculate model memory requirements."""
        params_bytes = self.params * 2  # BF16
        opt_bytes = self.params * 8  # FP32 moments
        grad_bytes = self.params * 2  # BF16

        return {
            'params_gb': params_bytes / 1e9,
            'optimizer_gb': opt_bytes / 1e9,
            'gradients_gb': grad_bytes / 1e9,
            'total_gb': (params_bytes + opt_bytes + grad_bytes) / 1e9
        }

    def sharded_memory_per_gpu(self):
        """Calculate memory per GPU with full sharding."""
        mem = self.model_memory()

        # Sharded across DP dimension
        params_per_gpu = mem['params_gb'] / self.dp
        opt_per_gpu = mem['optimizer_gb'] / self.dp
        grad_per_gpu = mem['gradients_gb'] / self.dp

        # But TP means each rank only has 1/TP of the layer weights
        params_per_gpu /= self.tp

        return {
            'params_gb': params_per_gpu,
            'optimizer_gb': opt_per_gpu,
            'gradients_gb': grad_per_gpu,
            'total_sharded_gb': params_per_gpu + opt_per_gpu + grad_per_gpu
        }

    def activation_memory(self, batch_size: int, seq_len: int):
        """Calculate activation memory with checkpointing."""
        # Per layer: attention + FFN activations
        attn_act = 2 * batch_size * seq_len * self.d_model  # Q, K, V intermediate
        ffn_act = batch_size * seq_len * self.d_ffn  # SwiGLU intermediate

        # With selective checkpointing, store ~1/3 of activations
        per_layer = (attn_act + ffn_act) * 2 * 0.33  # BF16, 1/3 stored

        # Distributed across TP
        per_layer /= self.tp

        # Only layers in this PP stage
        layers_per_stage = self.layers // self.pp

        total = layers_per_stage * per_layer

        return total / 1e9  # GB

    def tp_communication_time(self, batch_size: int, seq_len: int):
        """Time for TP AllReduce."""
        # 2 AllReduce per layer (attention, FFN)
        data_per_allreduce = batch_size * seq_len * self.d_model * 2  # BF16

        # Ring AllReduce within node
        ring_factor = 2 * (self.tp - 1) / self.tp
        time_per = ring_factor * data_per_allreduce / (self.nvlink_bw * 1e9)

        layers_per_stage = self.layers // self.pp
        return layers_per_stage * 2 * time_per

    def dp_communication_time(self):
        """Time for DP gradient sync (FSDP)."""
        mem = self.model_memory()

        # AllGather + ReduceScatter per layer
        params_per_layer = mem['params_gb'] * 1e9 / self.layers / self.tp

        ring_factor = 2 * (self.dp - 1) / self.dp
        time_per_layer = ring_factor * params_per_layer / (self.ib_bw * 1e9)

        layers_per_stage = self.layers // self.pp
        return layers_per_stage * time_per_layer

    def pipeline_bubble_fraction(self, microbatches: int):
        """Calculate pipeline bubble overhead."""
        return (self.pp - 1) / microbatches

    def estimate_mfu(self, tokens_per_step: int, step_time: float):
        """Estimate Model FLOPs Utilization."""
        # FLOPs per token (forward + backward)
        flops_per_token = 6 * self.params

        total_flops = flops_per_token * tokens_per_step
        achieved_flops = total_flops / step_time

        peak_flops = self.gpus * self.peak_flops

        return achieved_flops / peak_flops

    def training_time_estimate(self, total_tokens: int, mfu: float):
        """Estimate total training time."""
        flops_per_token = 6 * self.params
        total_flops = flops_per_token * total_tokens

        effective_flops = self.gpus * self.peak_flops * mfu

        seconds = total_flops / effective_flops
        days = seconds / (24 * 3600)

        return {'seconds': seconds, 'days': days}


def analyze_llama3():
    """Run the analysis."""
    analyzer = LLaMA3Analyzer()

    print("=== LLaMA 3 405B Training Analysis ===\n")

    # Memory
    mem = analyzer.model_memory()
    print(f"Total model state: {mem['total_gb']:.0f} GB")

    sharded = analyzer.sharded_memory_per_gpu()
    print(f"Sharded per GPU: {sharded['total_sharded_gb']:.1f} GB")

    act_mem = analyzer.activation_memory(batch_size=1, seq_len=8192)
    print(f"Activation memory per GPU: {act_mem:.1f} GB")
    print(f"Total per GPU: {sharded['total_sharded_gb'] + act_mem:.1f} GB / 80 GB available\n")

    # Communication
    tp_time = analyzer.tp_communication_time(batch_size=1024, seq_len=8192)
    dp_time = analyzer.dp_communication_time()
    print(f"TP communication time: {tp_time*1000:.1f} ms")
    print(f"DP communication time: {dp_time:.1f} s\n")

    # Efficiency
    bubble = analyzer.pipeline_bubble_fraction(microbatches=64)
    print(f"Pipeline bubble fraction: {bubble*100:.1f}%")

    mfu = analyzer.estimate_mfu(tokens_per_step=1_000_000, step_time=30)
    print(f"Estimated MFU: {mfu*100:.1f}%\n")

    # Training time
    time_est = analyzer.training_time_estimate(
        total_tokens=15.6e12,
        mfu=0.30
    )
    print(f"Estimated training time: {time_est['days']:.0f} days")

    return analyzer


if __name__ == "__main__":
    analyze_llama3()
```

## Exercises

1. **Parallelism trade-offs**: LLaMA 3 uses PP=16. Calculate the memory savings vs. PP=8 and PP=32. What is the pipeline bubble for each?

2. **TP scaling limit**: At what message size does TP=16 (crossing node boundary) become faster than TP=8 despite the lower bandwidth? (Hint: consider the latency term.)

3. **FSDP overlap**: LLaMA 3 overlaps FSDP AllGather with computation. What fraction of compute must be overlapped to hide the 200 GB/s of DP communication at 50 GB/s bandwidth?

4. **Context parallelism**: For 128K context with CP=8, calculate the additional KV cache communication per layer. Compare to standard attention memory scaling.

5. **Failure analysis**: With 10 failures/day and 5-minute recovery time, what fraction of training time is lost? How does this change with 30-second in-memory recovery?

6. **Batch size dynamics**: LLaMA 3 increased batch from 4M to 16M tokens. If critical batch size scaled from 2M to 10M during training, estimate the compute efficiency at each phase.

## Key Takeaways

1. **Scale demands 4D+ parallelism**: 405B parameters across 16K GPUs requires combining TP, PP, DP, and CP.

2. **Node boundaries matter**: NVLink (600 GB/s) vs InfiniBand (50 GB/s) dictates where each parallelism dimension operates.

3. **Memory constrains everything**: The 80GB GPU limit forces FSDP sharding and activation checkpointing.

4. **Efficiency is hard at scale**: Even Meta's optimized infrastructure achieves only ~30% MFU due to pipeline bubbles, communication, and overhead.

5. **Fault tolerance is mandatory**: With hundreds of failures per training run, recovery must be fast and automated.

6. **Dynamic training parameters**: Batch size, learning rate, and sequence length all evolve during training to match the changing loss landscape.
