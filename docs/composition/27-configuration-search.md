---
title: "Configuration Search"
subtitle: "Finding Optimal Parallelism in Exponential Spaces"
---

<div class="chapter-opener" markdown>
With 5 parallelism dimensions and dozens of hyperparameters, the configuration space for large-scale training grows exponentially. A 10,000 GPU cluster with DP, PP, TP, CP, and EP each ranging from 1 to 16 has over 100,000 possible configurations. Finding the optimal one is not trial-and-error—it's applied mathematics.
</div>

<div class="investigation-question" markdown>
**The Question**: Given a model architecture, target batch size, and cluster specification, how do you systematically find the configuration that minimizes training time while satisfying memory constraints? Can we do better than exhaustive search?
</div>

!!! note "Code style"
    Code examples in this chapter illustrate search *algorithms and data structures*. They are simplified for clarity — production configuration search tools (e.g., Alpa, Megatron's auto-parallelism) handle many additional constraints. Focus on the algorithmic ideas, not the implementation details.

!!! tip "Recommended reading path"
    If you want the core ideas first, read each section's problem statement, formulas, and decision rules before diving into implementation blocks.
    Treat long code listings as optional deep dives you can return to when building an optimizer.

## The Configuration Space

### Dimensions of Configuration

A complete distributed training configuration specifies:

```python
@dataclass
class TrainingConfig:
    """Complete specification of distributed training setup."""

    # Parallelism dimensions
    dp_size: int          # Data parallelism degree
    pp_size: int          # Pipeline parallelism stages
    tp_size: int          # Tensor parallelism degree
    cp_size: int          # Context parallelism (for long sequences)
    ep_size: int          # Expert parallelism (for MoE)

    # Memory optimizations
    zero_stage: int       # ZeRO optimization level (0, 1, 2, 3)
    activation_checkpointing: bool
    offload_optimizer: bool
    offload_params: bool

    # Micro-batching
    micro_batch_size: int
    gradient_accumulation_steps: int

    # Model shape
    seq_length: int
    hidden_dim: int

    # Pipeline schedule
    pipeline_schedule: str  # "1F1B", "interleaved", "zero-bubble"
    num_interleaved_stages: int

    # Communication
    overlap_comm_compute: bool
    bucket_size_mb: int

    # Precision
    precision: str        # "fp32", "fp16", "bf16", "fp8"

    @property
    def global_batch_size(self) -> int:
        return (self.dp_size * self.micro_batch_size *
                self.gradient_accumulation_steps)

    @property
    def total_gpus(self) -> int:
        return self.dp_size * self.pp_size * self.tp_size * self.cp_size * self.ep_size
```

### The Combinatorial Explosion

For a 1024 GPU cluster with:

- DP ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
- PP ∈ {1, 2, 4, 8, 16, 32}
- TP ∈ {1, 2, 4, 8}
- ZeRO ∈ {0, 1, 2, 3}
- micro_batch_size ∈ {1, 2, 4, 8, 16, 32}
- gradient_accumulation ∈ {1, 2, 4, 8, 16, 32, 64}

Valid combinations satisfying DP × PP × TP = 1024:

| Configuration | Count |
|---------------|-------|
| (1024, 1, 1) | 1 |
| (512, 2, 1), (512, 1, 2) | 2 |
| (256, 4, 1), (256, 2, 2), (256, 1, 4) | 3 |
| (128, 8, 1), (128, 4, 2), (128, 2, 4), (128, 1, 8) | 4 |
| ... | ... |

Total parallelism combinations: **~50**

With other hyperparameters: **50 × 4 × 6 × 7 = 8,400** configurations.

Add precision, checkpointing, offloading: **50,000+** configurations.

**Exhaustive search is infeasible.**

## Constraint Satisfaction

Before searching for optimal, we must filter for feasible.

### Memory Constraints

Each GPU has fixed HBM capacity $M_{\text{GPU}}$:

$$M_{\text{model}} + M_{\text{optimizer}} + M_{\text{activations}} + M_{\text{gradients}} \leq M_{\text{GPU}}$$

```python
class MemoryConstraint:
    """Check if configuration fits in GPU memory."""

    def __init__(self,
                 model_params: int,
                 hidden_dim: int,
                 num_layers: int,
                 seq_length: int,
                 vocab_size: int,
                 gpu_memory_gb: float = 80.0):

        self.model_params = model_params
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.gpu_memory = gpu_memory_gb * (1 << 30)  # Convert to bytes

    def is_feasible(self, config: TrainingConfig) -> Tuple[bool, Dict[str, float]]:
        """Check memory feasibility and return breakdown."""

        bytes_per_param = self._bytes_per_param(config.precision)

        # Model memory (sharded by TP and PP, then by ZeRO-3)
        params_per_gpu = self.model_params / (config.tp_size * config.pp_size)
        if config.zero_stage >= 3:
            params_per_gpu /= config.dp_size

        model_mem = params_per_gpu * bytes_per_param

        # Optimizer memory (sharded by ZeRO)
        optimizer_states = 2 if config.precision == "fp32" else 3  # m, v, [master]
        optimizer_bytes = 4 * optimizer_states  # Always FP32

        optimizer_mem = params_per_gpu * optimizer_bytes
        if config.zero_stage >= 1:
            optimizer_mem /= config.dp_size

        # Gradient memory (sharded by ZeRO-2+)
        gradient_mem = params_per_gpu * bytes_per_param
        if config.zero_stage >= 2:
            gradient_mem /= config.dp_size

        # Activation memory
        layers_per_stage = self.num_layers // config.pp_size
        activation_mem = self._activation_memory(
            config, layers_per_stage
        )

        # Apply offloading
        if config.offload_optimizer:
            optimizer_mem = 0
        if config.offload_params:
            model_mem *= 0.1  # Keep only active layer

        total = model_mem + optimizer_mem + gradient_mem + activation_mem

        breakdown = {
            'model_gb': model_mem / (1 << 30),
            'optimizer_gb': optimizer_mem / (1 << 30),
            'gradient_gb': gradient_mem / (1 << 30),
            'activation_gb': activation_mem / (1 << 30),
            'total_gb': total / (1 << 30),
            'available_gb': self.gpu_memory / (1 << 30),
        }

        return total <= self.gpu_memory, breakdown

    def _activation_memory(self,
                           config: TrainingConfig,
                           layers: int) -> float:
        """Compute activation memory per GPU."""

        batch = config.micro_batch_size
        seq = self.seq_length // config.cp_size
        hidden = self.hidden_dim // config.tp_size

        bytes_per_element = self._bytes_per_param(config.precision)

        # Activations per layer
        # Input: batch × seq × hidden
        # Attention: batch × heads × seq × seq
        # FFN: batch × seq × 4*hidden

        # Tensor counts (bytes applied below)
        per_layer = batch * seq * hidden * 2  # Input + output
        per_layer += batch * seq * seq * 2    # Attention scores + probs (approx)
        per_layer += batch * seq * hidden * 4  # FFN intermediate

        total = per_layer * layers * bytes_per_element

        # Activation checkpointing reduces by ~layers factor
        if config.activation_checkpointing:
            total /= layers

        return total

    def _bytes_per_param(self, precision: str) -> int:
        return {'fp32': 4, 'fp16': 2, 'bf16': 2, 'fp8': 1}[precision]
```

### Bandwidth Constraints

Communication time must not dominate compute:

$$T_{\text{comm}} \leq \alpha \cdot T_{\text{compute}}$$

Typically $\alpha \leq 0.2$ (communication should be ≤20% of compute).

```python
class BandwidthConstraint:
    """Check if configuration has acceptable communication overhead."""

    def __init__(self,
                 model_params: int,
                 flops_per_token: int,
                 hidden_dim: int,
                 seq_length: int,
                 num_layers: int,
                 intra_node_bw: float = 300e9,   # NVLink: 300 GB/s
                 inter_node_bw: float = 100e9,   # IB: 100 GB/s
                 gpu_flops: float = 989e12,     # H100: 989 TFLOPs FP16/BF16 dense
                 max_comm_ratio: float = 0.2):

        self.model_params = model_params
        self.flops_per_token = flops_per_token
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.intra_bw = intra_node_bw
        self.inter_bw = inter_node_bw
        self.gpu_flops = gpu_flops
        self.max_comm_ratio = max_comm_ratio

    def is_feasible(self, config: TrainingConfig) -> Tuple[bool, Dict[str, float]]:
        """Check bandwidth feasibility."""

        bytes_per_param = 2 if config.precision in ['fp16', 'bf16'] else 4

        # TP communication (AllReduce per layer, high frequency)
        # Happens within node (NVLink)
        if config.tp_size > 1:
            layers_per_stage = self.num_layers // config.pp_size
            activation_size = (config.micro_batch_size *
                               self.seq_length *
                               self.hidden_dim *
                               bytes_per_param)
            per_allreduce = 2 * (config.tp_size - 1) / config.tp_size * activation_size
            tp_volume = 2 * layers_per_stage * per_allreduce  # ~2 AllReduces per layer
            tp_time = tp_volume / self.intra_bw
        else:
            tp_time = 0

        # DP communication (AllReduce gradients, once per step)
        # May cross nodes (IB)
        if config.dp_size > 1:
            params_per_rank = self.model_params / (config.tp_size * config.pp_size)
            grad_volume = params_per_rank * bytes_per_param

            # ZeRO reduces communication
            if config.zero_stage >= 1:
                # ReduceScatter + AllGather instead of AllReduce
                dp_volume = grad_volume * 2 * (config.dp_size - 1) / config.dp_size
            else:
                dp_volume = grad_volume * 2 * (config.dp_size - 1) / config.dp_size

            # Determine if inter-node
            gpus_per_node = 8  # Typical
            if config.dp_size <= gpus_per_node // config.tp_size:
                dp_bw = self.intra_bw
            else:
                dp_bw = self.inter_bw

            dp_time = dp_volume / dp_bw
        else:
            dp_time = 0

        # PP communication (point-to-point, per micro-batch)
        if config.pp_size > 1:
            activation_size = (config.micro_batch_size *
                             self.seq_length *
                             self.hidden_dim *
                             bytes_per_param)
            pp_volume = activation_size * 2  # Forward + backward
            pp_time = pp_volume / self.inter_bw
        else:
            pp_time = 0

        total_comm_time = tp_time + dp_time + pp_time

        # Compute time
        tokens = config.micro_batch_size * config.seq_length
        flops = self.flops_per_token * tokens
        compute_time = flops / self.gpu_flops

        comm_ratio = total_comm_time / (compute_time + total_comm_time)

        breakdown = {
            'tp_time_ms': tp_time * 1000,
            'dp_time_ms': dp_time * 1000,
            'pp_time_ms': pp_time * 1000,
            'compute_time_ms': compute_time * 1000,
            'comm_ratio': comm_ratio,
        }

        return comm_ratio <= self.max_comm_ratio, breakdown
```

### Divisibility Constraints

Configuration must be mathematically consistent:

```python
def divisibility_constraints(config: TrainingConfig,
                             num_layers: int,
                             num_attention_heads: int,
                             vocab_size: int,
                             target_gpus: int) -> List[str]:
    """Check all divisibility requirements."""

    violations = []

    # GPU count constraint
    actual_gpus = config.dp_size * config.pp_size * config.tp_size
    if actual_gpus != target_gpus:
        violations.append(
            f"GPU count: {actual_gpus} != target {target_gpus}"
        )

    # Layers divisible by PP stages
    if num_layers % config.pp_size != 0:
        violations.append(
            f"Layers {num_layers} not divisible by PP {config.pp_size}"
        )

    # Attention heads divisible by TP
    if num_attention_heads % config.tp_size != 0:
        violations.append(
            f"Heads {num_attention_heads} not divisible by TP {config.tp_size}"
        )

    # Vocab size divisible by TP (for embedding sharding)
    if vocab_size % config.tp_size != 0:
        violations.append(
            f"Vocab {vocab_size} not divisible by TP {config.tp_size}"
        )

    # Global batch size constraint
    if config.global_batch_size < config.dp_size:
        violations.append(
            f"Global batch {config.global_batch_size} < DP {config.dp_size}"
        )

    return violations
```

## Cost Models

Rather than running each configuration, we predict performance analytically.

### The Performance Model

```python
class PerformanceModel:
    """Predict training throughput without execution."""

    def __init__(self,
                 model_params: int,
                 hidden_dim: int,
                 num_layers: int,
                 seq_length: int,
                 hardware: HardwareSpec):

        self.model_params = model_params
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.hw = hardware

    def predict_step_time(self, config: TrainingConfig) -> StepTimePrediction:
        """Predict time for one training step."""

        # Compute time
        t_compute = self._compute_time(config)

        # Communication times
        t_tp = self._tp_comm_time(config)
        t_dp = self._dp_comm_time(config)
        t_pp = self._pp_comm_time(config)

        # Pipeline bubble
        t_bubble = self._pipeline_bubble(config)

        # Memory operations
        t_memory = self._memory_time(config)

        # Overlap adjustments
        if config.overlap_comm_compute:
            # DP comm overlaps with compute
            t_dp_exposed = max(0, t_dp - t_compute * 0.8)
        else:
            t_dp_exposed = t_dp

        # Total time
        # TP comm is in critical path (per-layer)
        # PP comm happens at stage boundaries
        t_total = t_compute + t_tp + t_dp_exposed + t_bubble + t_memory

        return StepTimePrediction(
            compute=t_compute,
            tp_comm=t_tp,
            dp_comm=t_dp,
            dp_exposed=t_dp_exposed,
            pp_comm=t_pp,
            bubble=t_bubble,
            memory=t_memory,
            total=t_total,
        )

    def _compute_time(self, config: TrainingConfig) -> float:
        """Time for forward + backward compute."""

        tokens_per_step = (config.micro_batch_size *
                          config.seq_length *
                          config.gradient_accumulation_steps)

        # FLOPs for transformer (forward only)
        # Approximately 2 * params * tokens for attention + FFN
        flops_forward = 2 * self.model_params * tokens_per_step

        # Backward is ~2x forward
        flops_total = flops_forward * 3  # Forward + backward

        # Divide by parallelism
        flops_per_gpu = flops_total / (config.tp_size * config.pp_size)

        # Account for efficiency loss from small batches
        efficiency = self._compute_efficiency(config)

        return flops_per_gpu / (self.hw.flops * efficiency)

    def _compute_efficiency(self, config: TrainingConfig) -> float:
        """GPU compute efficiency based on batch size."""

        # Small batches underutilize GPU
        tokens = config.micro_batch_size * config.seq_length
        min_efficient_tokens = 2048  # Empirical

        if tokens >= min_efficient_tokens:
            return 0.5  # Typical for well-optimized kernels
        else:
            return 0.5 * (tokens / min_efficient_tokens) ** 0.5

    def _tp_comm_time(self, config: TrainingConfig) -> float:
        """Tensor parallel AllReduce time."""

        if config.tp_size == 1:
            return 0

        # Two AllReduces per layer: after attention, after FFN
        # Volume per AllReduce: batch × seq × hidden / tp × bytes × 2
        bytes_per = 2 if config.precision in ['fp16', 'bf16'] else 4

        volume_per_layer = (config.micro_batch_size *
                          config.seq_length *
                          self.hidden_dim / config.tp_size *
                          bytes_per * 2)  # AllReduce factor

        layers_per_stage = self.num_layers // config.pp_size
        total_volume = volume_per_layer * 2 * layers_per_stage

        # TP is within node (NVLink)
        return total_volume / self.hw.intra_node_bw

    def _dp_comm_time(self, config: TrainingConfig) -> float:
        """Data parallel gradient synchronization time."""

        if config.dp_size == 1:
            return 0

        bytes_per = 2 if config.precision in ['fp16', 'bf16'] else 4

        # Gradient volume
        params_per_gpu = self.model_params / config.tp_size / config.pp_size
        grad_volume = params_per_gpu * bytes_per

        # AllReduce volume
        allreduce_volume = grad_volume * 2 * (config.dp_size - 1) / config.dp_size

        # Determine bandwidth (may cross nodes)
        if config.dp_size * config.tp_size <= 8:  # Within node
            bw = self.hw.intra_node_bw
        else:
            bw = self.hw.inter_node_bw

        return allreduce_volume / bw

    def _pp_comm_time(self, config: TrainingConfig) -> float:
        """Pipeline parallel P2P communication time."""

        if config.pp_size == 1:
            return 0

        bytes_per = 2 if config.precision in ['fp16', 'bf16'] else 4

        # Activation size at stage boundary
        activation_size = (config.micro_batch_size *
                          config.seq_length *
                          self.hidden_dim *
                          bytes_per)

        # Forward + backward per micro-batch
        volume_per_mb = activation_size * 2

        # Total micro-batches
        num_mb = config.gradient_accumulation_steps

        # P2P is usually across nodes
        return volume_per_mb * num_mb / self.hw.inter_node_bw

    def _pipeline_bubble(self, config: TrainingConfig) -> float:
        """Pipeline bubble overhead."""

        if config.pp_size == 1:
            return 0

        # Bubble fraction
        num_mb = config.gradient_accumulation_steps

        if config.pipeline_schedule == "1F1B":
            # Bubble = (p-1) / (m + p - 1)
            bubble_frac = (config.pp_size - 1) / (num_mb + config.pp_size - 1)
        elif config.pipeline_schedule == "interleaved":
            # Bubble = (p-1) / (m * v + p - 1) where v = virtual stages
            v = config.num_interleaved_stages
            bubble_frac = (config.pp_size - 1) / (num_mb * v + config.pp_size - 1)
        else:  # zero-bubble
            bubble_frac = 0.05  # Small overhead for synchronization

        # Bubble adds this fraction to total time
        compute_time = self._compute_time(config)
        return compute_time * bubble_frac / (1 - bubble_frac)

    def _memory_time(self, config: TrainingConfig) -> float:
        """Overhead from memory operations (offloading, etc.)."""

        overhead = 0

        if config.offload_optimizer:
            # PCIe transfer for optimizer states
            optimizer_size = self.model_params * 12  # FP32 m, v, master
            overhead += optimizer_size / self.hw.pcie_bw

        if config.offload_params:
            # PCIe transfer for parameters
            params_size = self.model_params * 2  # FP16
            overhead += params_size / self.hw.pcie_bw * 2  # To and from

        return overhead

@dataclass
class StepTimePrediction:
    """Breakdown of predicted step time."""

    compute: float
    tp_comm: float
    dp_comm: float
    dp_exposed: float
    pp_comm: float
    bubble: float
    memory: float
    total: float

    def throughput_tokens_per_sec(self,
                                   batch_size: int,
                                   seq_length: int) -> float:
        return batch_size * seq_length / self.total

@dataclass
class HardwareSpec:
    """Hardware specifications for a single GPU."""

    flops: float = 989e12         # Peak TFLOPs (H100 FP16/BF16 dense)
    memory: float = 80e9           # HBM capacity
    intra_node_bw: float = 300e9   # NVLink bandwidth
    inter_node_bw: float = 100e9   # InfiniBand bandwidth
    pcie_bw: float = 32e9          # PCIe bandwidth
```

### Model Validation

Cost models must be validated against actual measurements:

```python
class ModelValidator:
    """Validate cost model against real measurements."""

    def __init__(self, performance_model: PerformanceModel):
        self.model = performance_model
        self.measurements: List[Tuple[TrainingConfig, float]] = []

    def add_measurement(self,
                        config: TrainingConfig,
                        actual_step_time: float) -> None:
        """Add an actual measurement for calibration."""
        self.measurements.append((config, actual_step_time))

    def compute_error(self) -> Dict[str, float]:
        """Compute prediction error statistics."""

        errors = []

        for config, actual in self.measurements:
            predicted = self.model.predict_step_time(config).total
            relative_error = (predicted - actual) / actual
            errors.append(relative_error)

        return {
            'mean_error': np.mean(errors),
            'abs_mean_error': np.mean(np.abs(errors)),
            'max_error': np.max(np.abs(errors)),
            'std_error': np.std(errors),
        }

    def calibrate(self) -> Dict[str, float]:
        """Fit correction factors to minimize prediction error."""

        from scipy.optimize import minimize

        def objective(params):
            compute_scale, comm_scale = params

            total_error = 0
            for config, actual in self.measurements:
                pred = self.model.predict_step_time(config)
                adjusted = (pred.compute * compute_scale +
                           (pred.tp_comm + pred.dp_exposed + pred.pp_comm) * comm_scale +
                           pred.bubble + pred.memory)
                total_error += (adjusted - actual) ** 2

            return total_error

        result = minimize(objective, [1.0, 1.0], method='Nelder-Mead')

        return {
            'compute_scale': result.x[0],
            'comm_scale': result.x[1],
        }
```

## Search Algorithms

### Grid Search with Pruning

```python
class PrunedGridSearch:
    """Grid search with early constraint-based pruning."""

    def __init__(self,
                 memory_constraint: MemoryConstraint,
                 bandwidth_constraint: BandwidthConstraint,
                 performance_model: PerformanceModel,
                 target_gpus: int,
                 num_layers: int,
                 num_heads: int):

        self.memory = memory_constraint
        self.bandwidth = bandwidth_constraint
        self.perf_model = performance_model
        self.target_gpus = target_gpus
        self.num_layers = num_layers
        self.num_heads = num_heads

    def search(self,
               dp_choices: List[int],
               pp_choices: List[int],
               tp_choices: List[int],
               zero_choices: List[int],
               mbs_choices: List[int],
               ga_choices: List[int]) -> List[Tuple[TrainingConfig, float]]:
        """Search configuration space with pruning."""

        valid_configs = []
        pruned_counts = {'gpu': 0, 'divisibility': 0, 'memory': 0, 'bandwidth': 0}

        for dp in dp_choices:
            for pp in pp_choices:
                for tp in tp_choices:
                    # Early pruning: GPU count
                    if dp * pp * tp != self.target_gpus:
                        pruned_counts['gpu'] += 1
                        continue

                    # Early pruning: divisibility
                    if self.num_layers % pp != 0:
                        pruned_counts['divisibility'] += 1
                        continue
                    if self.num_heads % tp != 0:
                        pruned_counts['divisibility'] += 1
                        continue

                    for zero in zero_choices:
                        for mbs in mbs_choices:
                            for ga in ga_choices:
                                config = TrainingConfig(
                                    dp_size=dp,
                                    pp_size=pp,
                                    tp_size=tp,
                                    cp_size=1,
                                    ep_size=1,
                                    zero_stage=zero,
                                    activation_checkpointing=True,
                                    offload_optimizer=False,
                                    offload_params=False,
                                    micro_batch_size=mbs,
                                    gradient_accumulation_steps=ga,
                                    seq_length=self.seq_length,
                                    hidden_dim=self.hidden_dim,
                                    pipeline_schedule="1F1B",
                                    num_interleaved_stages=1,
                                    overlap_comm_compute=True,
                                    bucket_size_mb=25,
                                    precision="bf16",
                                )

                                # Memory constraint
                                mem_ok, _ = self.memory.is_feasible(config)
                                if not mem_ok:
                                    pruned_counts['memory'] += 1
                                    continue

                                # Bandwidth constraint
                                bw_ok, _ = self.bandwidth.is_feasible(config)
                                if not bw_ok:
                                    pruned_counts['bandwidth'] += 1
                                    continue

                                # Predict performance
                                pred = self.perf_model.predict_step_time(config)
                                valid_configs.append((config, pred.total))

        print(f"Pruning statistics: {pruned_counts}")
        print(f"Valid configurations: {len(valid_configs)}")

        # Sort by predicted step time
        valid_configs.sort(key=lambda x: x[1])

        return valid_configs
```

### Bayesian Optimization

For expensive evaluations (actual runs), Bayesian optimization is more efficient:

```python
class BayesianConfigSearch:
    """Bayesian optimization for configuration search."""

    def __init__(self,
                 objective_fn: Callable[[TrainingConfig], float],
                 constraints: List[Callable[[TrainingConfig], bool]],
                 param_space: Dict[str, List[Any]]):

        self.objective = objective_fn
        self.constraints = constraints
        self.param_space = param_space

        # Gaussian Process surrogate
        self.observations: List[Tuple[TrainingConfig, float]] = []

    def suggest_next(self) -> TrainingConfig:
        """Suggest next configuration to evaluate."""

        if len(self.observations) < 10:
            # Initial random exploration
            return self._random_sample()

        # Fit GP to observations
        X, y = self._prepare_data()
        gp = self._fit_gaussian_process(X, y)

        # Maximize acquisition function
        best_config = None
        best_acquisition = float('-inf')

        for _ in range(1000):  # Random search for acquisition max
            config = self._random_sample()

            if not all(c(config) for c in self.constraints):
                continue

            x = self._config_to_vector(config)
            acq = self._expected_improvement(x, gp, np.min(y))

            if acq > best_acquisition:
                best_acquisition = acq
                best_config = config

        return best_config

    def observe(self, config: TrainingConfig, value: float) -> None:
        """Record observation."""
        self.observations.append((config, value))

    def _expected_improvement(self,
                               x: np.ndarray,
                               gp,
                               y_best: float) -> float:
        """Expected improvement acquisition function."""

        mu, sigma = gp.predict(x.reshape(1, -1), return_std=True)

        if sigma < 1e-6:
            return 0.0

        z = (y_best - mu) / sigma
        ei = (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

        return ei[0]

    def _random_sample(self) -> TrainingConfig:
        """Generate random valid configuration."""

        while True:
            dp = random.choice(self.param_space['dp_size'])
            pp = random.choice(self.param_space['pp_size'])
            tp = random.choice(self.param_space['tp_size'])

            if dp * pp * tp != self.param_space['target_gpus']:
                continue

            config = TrainingConfig(
                dp_size=dp,
                pp_size=pp,
                tp_size=tp,
                cp_size=1,
                ep_size=1,
                zero_stage=random.choice(self.param_space['zero_stage']),
                activation_checkpointing=True,
                offload_optimizer=False,
                offload_params=False,
                micro_batch_size=random.choice(self.param_space['micro_batch_size']),
                gradient_accumulation_steps=random.choice(
                    self.param_space['gradient_accumulation']
                ),
                seq_length=self.param_space['seq_length'],
                hidden_dim=self.param_space['hidden_dim'],
                pipeline_schedule="1F1B",
                num_interleaved_stages=1,
                overlap_comm_compute=True,
                bucket_size_mb=25,
                precision="bf16",
            )

            return config

    def _config_to_vector(self, config: TrainingConfig) -> np.ndarray:
        """Convert configuration to numerical vector."""

        return np.array([
            np.log2(config.dp_size),
            np.log2(config.pp_size),
            np.log2(config.tp_size),
            config.zero_stage,
            np.log2(config.micro_batch_size),
            np.log2(config.gradient_accumulation_steps),
        ])

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for GP."""

        X = np.array([self._config_to_vector(c) for c, _ in self.observations])
        y = np.array([v for _, v in self.observations])

        return X, y

    def _fit_gaussian_process(self, X: np.ndarray, y: np.ndarray):
        """Fit Gaussian Process to data."""

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gp.fit(X, y)

        return gp
```

### Evolutionary Search

For complex configuration spaces with many local optima:

```python
class EvolutionarySearch:
    """Evolutionary algorithm for configuration optimization."""

    def __init__(self,
                 fitness_fn: Callable[[TrainingConfig], float],
                 constraints: List[Callable[[TrainingConfig], bool]],
                 param_space: Dict[str, List[Any]],
                 population_size: int = 50,
                 generations: int = 100):

        self.fitness = fitness_fn
        self.constraints = constraints
        self.param_space = param_space
        self.pop_size = population_size
        self.generations = generations

    def search(self) -> Tuple[TrainingConfig, float]:
        """Run evolutionary search."""

        # Initialize population
        population = self._initialize_population()

        best_config = None
        best_fitness = float('inf')

        for gen in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for config in population:
                if all(c(config) for c in self.constraints):
                    score = self.fitness(config)
                else:
                    score = float('inf')  # Infeasible

                fitness_scores.append(score)

                if score < best_fitness:
                    best_fitness = score
                    best_config = config

            print(f"Generation {gen}: best fitness = {best_fitness:.4f}")

            # Selection (tournament)
            parents = self._tournament_selection(population, fitness_scores)

            # Crossover
            offspring = self._crossover(parents)

            # Mutation
            offspring = [self._mutate(c) for c in offspring]

            # Replace population
            population = offspring

        return best_config, best_fitness

    def _initialize_population(self) -> List[TrainingConfig]:
        """Create initial random population."""

        population = []

        while len(population) < self.pop_size:
            config = self._random_config()
            if all(c(config) for c in self.constraints):
                population.append(config)

        return population

    def _tournament_selection(self,
                               population: List[TrainingConfig],
                               fitness: List[float],
                               tournament_size: int = 3) -> List[TrainingConfig]:
        """Select parents via tournament selection."""

        parents = []

        for _ in range(self.pop_size):
            tournament_idx = random.sample(range(len(population)), tournament_size)
            winner_idx = min(tournament_idx, key=lambda i: fitness[i])
            parents.append(population[winner_idx])

        return parents

    def _crossover(self, parents: List[TrainingConfig]) -> List[TrainingConfig]:
        """Uniform crossover between pairs of parents."""

        offspring = []
        random.shuffle(parents)

        for i in range(0, len(parents), 2):
            if i + 1 >= len(parents):
                offspring.append(parents[i])
                continue

            p1, p2 = parents[i], parents[i + 1]

            # Create children by mixing parent attributes
            child1_attrs = {}
            child2_attrs = {}

            for field in ['dp_size', 'pp_size', 'tp_size', 'zero_stage',
                         'micro_batch_size', 'gradient_accumulation_steps']:
                if random.random() < 0.5:
                    child1_attrs[field] = getattr(p1, field)
                    child2_attrs[field] = getattr(p2, field)
                else:
                    child1_attrs[field] = getattr(p2, field)
                    child2_attrs[field] = getattr(p1, field)

            # Repair GPU constraint
            child1_attrs = self._repair_gpu_constraint(child1_attrs)
            child2_attrs = self._repair_gpu_constraint(child2_attrs)

            offspring.append(self._attrs_to_config(child1_attrs))
            offspring.append(self._attrs_to_config(child2_attrs))

        return offspring[:self.pop_size]

    def _mutate(self, config: TrainingConfig, mutation_rate: float = 0.1) -> TrainingConfig:
        """Mutate configuration with given probability."""

        attrs = {
            'dp_size': config.dp_size,
            'pp_size': config.pp_size,
            'tp_size': config.tp_size,
            'zero_stage': config.zero_stage,
            'micro_batch_size': config.micro_batch_size,
            'gradient_accumulation_steps': config.gradient_accumulation_steps,
        }

        for field in attrs:
            if random.random() < mutation_rate:
                attrs[field] = random.choice(self.param_space[field])

        # Repair constraints
        attrs = self._repair_gpu_constraint(attrs)

        return self._attrs_to_config(attrs)

    def _repair_gpu_constraint(self, attrs: Dict) -> Dict:
        """Repair configuration to satisfy GPU count constraint."""

        target = self.param_space['target_gpus']

        # Adjust DP to satisfy constraint
        current = attrs['pp_size'] * attrs['tp_size']
        if target % current == 0:
            attrs['dp_size'] = target // current
        else:
            # Find valid combination
            for pp in self.param_space['pp_size']:
                for tp in self.param_space['tp_size']:
                    if target % (pp * tp) == 0:
                        attrs['pp_size'] = pp
                        attrs['tp_size'] = tp
                        attrs['dp_size'] = target // (pp * tp)
                        return attrs

        return attrs

    def _attrs_to_config(self, attrs: Dict) -> TrainingConfig:
        """Convert attributes dict to TrainingConfig."""

        return TrainingConfig(
            dp_size=attrs['dp_size'],
            pp_size=attrs['pp_size'],
            tp_size=attrs['tp_size'],
            cp_size=1,
            ep_size=1,
            zero_stage=attrs['zero_stage'],
            activation_checkpointing=True,
            offload_optimizer=False,
            offload_params=False,
            micro_batch_size=attrs['micro_batch_size'],
            gradient_accumulation_steps=attrs['gradient_accumulation_steps'],
            seq_length=self.param_space['seq_length'],
            hidden_dim=self.param_space['hidden_dim'],
            pipeline_schedule="1F1B",
            num_interleaved_stages=1,
            overlap_comm_compute=True,
            bucket_size_mb=25,
            precision="bf16",
        )

    def _random_config(self) -> TrainingConfig:
        """Generate random configuration."""

        while True:
            dp = random.choice(self.param_space['dp_size'])
            pp = random.choice(self.param_space['pp_size'])
            tp = random.choice(self.param_space['tp_size'])

            if dp * pp * tp == self.param_space['target_gpus']:
                break

        return TrainingConfig(
            dp_size=dp,
            pp_size=pp,
            tp_size=tp,
            cp_size=1,
            ep_size=1,
            zero_stage=random.choice(self.param_space['zero_stage']),
            activation_checkpointing=True,
            offload_optimizer=False,
            offload_params=False,
            micro_batch_size=random.choice(self.param_space['micro_batch_size']),
            gradient_accumulation_steps=random.choice(
                self.param_space['gradient_accumulation']
            ),
            seq_length=self.param_space['seq_length'],
            hidden_dim=self.param_space['hidden_dim'],
            pipeline_schedule="1F1B",
            num_interleaved_stages=1,
            overlap_comm_compute=True,
            bucket_size_mb=25,
            precision="bf16",
        )
```

## Hierarchical Search

Large configuration spaces benefit from hierarchical decomposition:

```python
class HierarchicalSearch:
    """Search parallelism first, then memory optimizations, then micro-batching."""

    def __init__(self,
                 model_spec: ModelSpec,
                 hardware_spec: HardwareSpec,
                 target_gpus: int,
                 target_batch_size: int):

        self.model = model_spec
        self.hardware = hardware_spec
        self.target_gpus = target_gpus
        self.target_batch = target_batch_size

    def search(self) -> TrainingConfig:
        """Hierarchical configuration search."""

        # Level 1: Find valid parallelism dimensions
        parallelism_configs = self._search_parallelism()
        print(f"Level 1: {len(parallelism_configs)} parallelism configs")

        # Level 2: For each parallelism, find memory optimization
        memory_configs = []
        for p_config in parallelism_configs:
            m_configs = self._search_memory_optimizations(p_config)
            memory_configs.extend(m_configs)
        print(f"Level 2: {len(memory_configs)} memory configs")

        # Level 3: For each memory config, optimize micro-batching
        final_configs = []
        for m_config in memory_configs:
            f_config = self._search_micro_batching(m_config)
            if f_config:
                final_configs.append(f_config)
        print(f"Level 3: {len(final_configs)} final configs")

        # Rank by predicted throughput
        final_configs.sort(
            key=lambda c: self._predict_throughput(c),
            reverse=True
        )

        return final_configs[0] if final_configs else None

    def _search_parallelism(self) -> List[TrainingConfig]:
        """Find valid parallelism dimension combinations."""

        valid = []

        for pp in [1, 2, 4, 8, 16, 32]:
            for tp in [1, 2, 4, 8]:
                if self.target_gpus % (pp * tp) != 0:
                    continue

                dp = self.target_gpus // (pp * tp)

                # Check divisibility
                if self.model.num_layers % pp != 0:
                    continue
                if self.model.num_heads % tp != 0:
                    continue

                config = TrainingConfig(
                    dp_size=dp,
                    pp_size=pp,
                    tp_size=tp,
                    cp_size=1,
                    ep_size=1,
                    zero_stage=0,  # Placeholder
                    activation_checkpointing=False,
                    offload_optimizer=False,
                    offload_params=False,
                    micro_batch_size=1,  # Placeholder
                    gradient_accumulation_steps=1,  # Placeholder
                    seq_length=self.model.seq_length,
                    hidden_dim=self.model.hidden_dim,
                    pipeline_schedule="1F1B",
                    num_interleaved_stages=1,
                    overlap_comm_compute=True,
                    bucket_size_mb=25,
                    precision="bf16",
                )

                valid.append(config)

        return valid

    def _search_memory_optimizations(self,
                                      base_config: TrainingConfig) -> List[TrainingConfig]:
        """Find memory optimizations that fit in GPU."""

        valid = []

        for zero in [0, 1, 2, 3]:
            for ckpt in [False, True]:
                for offload_opt in [False, True]:
                    # Skip invalid combinations
                    if offload_opt and zero < 2:
                        continue

                    config = TrainingConfig(
                        dp_size=base_config.dp_size,
                        pp_size=base_config.pp_size,
                        tp_size=base_config.tp_size,
                        cp_size=1,
                        ep_size=1,
                        zero_stage=zero,
                        activation_checkpointing=ckpt,
                        offload_optimizer=offload_opt,
                        offload_params=False,
                        micro_batch_size=1,
                        gradient_accumulation_steps=1,
                        seq_length=base_config.seq_length,
                        hidden_dim=base_config.hidden_dim,
                        pipeline_schedule="1F1B",
                        num_interleaved_stages=1,
                        overlap_comm_compute=True,
                        bucket_size_mb=25,
                        precision="bf16",
                    )

                    # Check if base memory fits
                    if self._check_memory_feasible(config, micro_batch=1):
                        valid.append(config)

        return valid

    def _search_micro_batching(self,
                                base_config: TrainingConfig) -> Optional[TrainingConfig]:
        """Find optimal micro-batch size and gradient accumulation."""

        best_config = None
        best_throughput = 0

        for mbs in [1, 2, 4, 8, 16, 32, 64]:
            if not self._check_memory_feasible(base_config, micro_batch=mbs):
                break  # Larger won't fit either

            # Calculate gradient accumulation to achieve target batch
            per_dp_batch = self.target_batch // base_config.dp_size
            ga = per_dp_batch // mbs

            if ga < 1:
                continue

            config = TrainingConfig(
                dp_size=base_config.dp_size,
                pp_size=base_config.pp_size,
                tp_size=base_config.tp_size,
                cp_size=1,
                ep_size=1,
                zero_stage=base_config.zero_stage,
                activation_checkpointing=base_config.activation_checkpointing,
                offload_optimizer=base_config.offload_optimizer,
                offload_params=base_config.offload_params,
                micro_batch_size=mbs,
                gradient_accumulation_steps=ga,
                seq_length=base_config.seq_length,
                hidden_dim=base_config.hidden_dim,
                pipeline_schedule="1F1B",
                num_interleaved_stages=1,
                overlap_comm_compute=True,
                bucket_size_mb=25,
                precision="bf16",
            )

            throughput = self._predict_throughput(config)

            if throughput > best_throughput:
                best_throughput = throughput
                best_config = config

        return best_config

    def _check_memory_feasible(self,
                                config: TrainingConfig,
                                micro_batch: int) -> bool:
        """Check if configuration fits in GPU memory."""

        # Simplified memory check
        bytes_per_param = 2  # BF16

        # Model params per GPU
        params = self.model.params / config.tp_size / config.pp_size

        # With ZeRO-3, further divide by DP
        if config.zero_stage >= 3:
            params /= config.dp_size

        model_mem = params * bytes_per_param

        # Optimizer (12 bytes per param for Adam)
        opt_mem = params * 12
        if config.zero_stage >= 1:
            opt_mem /= config.dp_size
        if config.offload_optimizer:
            opt_mem = 0

        # Activations
        layers_per_stage = self.model.num_layers // config.pp_size
        act_per_layer = micro_batch * self.model.seq_length * self.model.hidden_dim * 2
        act_mem = act_per_layer * layers_per_stage

        if config.activation_checkpointing:
            act_mem /= layers_per_stage

        total = model_mem + opt_mem + act_mem

        return total < self.hardware.memory * 0.9  # 90% threshold

    def _predict_throughput(self, config: TrainingConfig) -> float:
        """Predict tokens per second."""

        perf_model = PerformanceModel(
            self.model.params,
            self.model.hidden_dim,
            self.model.num_layers,
            self.model.seq_length,
            self.hardware
        )

        step_time = perf_model.predict_step_time(config).total
        tokens_per_step = config.global_batch_size * self.model.seq_length

        return tokens_per_step / step_time
```

## Auto-Tuning Integration

### DeepSpeed Auto-Tuner Style

```python
class AutoTuner:
    """Automatic configuration tuning with profiling."""

    def __init__(self,
                 model_fn: Callable[[], nn.Module],
                 dataset: Dataset,
                 target_gpus: int,
                 profile_steps: int = 10):

        self.model_fn = model_fn
        self.dataset = dataset
        self.target_gpus = target_gpus
        self.profile_steps = profile_steps

        self.results: Dict[str, Tuple[TrainingConfig, float]] = {}

    def run_profiling(self, config: TrainingConfig) -> float:
        """Run actual profiling with given configuration."""

        try:
            # Initialize model with config
            model = self._setup_distributed_model(config)
            optimizer = self._setup_optimizer(model, config)
            dataloader = self._setup_dataloader(config)

            # Warmup
            for _ in range(3):
                batch = next(iter(dataloader))
                self._train_step(model, optimizer, batch)

            # Profile
            start_time = time.time()
            for i, batch in enumerate(dataloader):
                if i >= self.profile_steps:
                    break
                self._train_step(model, optimizer, batch)

            elapsed = time.time() - start_time
            step_time = elapsed / self.profile_steps

            # Cleanup
            del model, optimizer
            torch.cuda.empty_cache()

            return step_time

        except RuntimeError as e:
            if "out of memory" in str(e):
                return float('inf')  # OOM = infeasible
            raise

    def auto_tune(self,
                  max_trials: int = 50,
                  strategy: str = "bayesian") -> TrainingConfig:
        """Run automatic tuning."""

        param_space = {
            'dp_size': [2**i for i in range(int(np.log2(self.target_gpus)) + 1)],
            'pp_size': [1, 2, 4, 8],
            'tp_size': [1, 2, 4, 8],
            'zero_stage': [0, 1, 2, 3],
            'micro_batch_size': [1, 2, 4, 8, 16],
            'gradient_accumulation': [1, 2, 4, 8, 16, 32],
            'target_gpus': self.target_gpus,
        }

        if strategy == "bayesian":
            searcher = BayesianConfigSearch(
                objective_fn=self.run_profiling,
                constraints=[self._basic_constraints],
                param_space=param_space
            )

            for trial in range(max_trials):
                config = searcher.suggest_next()
                step_time = self.run_profiling(config)
                searcher.observe(config, step_time)

                print(f"Trial {trial}: step_time = {step_time:.4f}s")

                self.results[self._config_key(config)] = (config, step_time)

        # Return best
        best_key = min(self.results, key=lambda k: self.results[k][1])
        return self.results[best_key][0]

    def _basic_constraints(self, config: TrainingConfig) -> bool:
        """Basic constraint check."""

        if config.dp_size * config.pp_size * config.tp_size != self.target_gpus:
            return False

        return True

    def _config_key(self, config: TrainingConfig) -> str:
        """Create unique key for configuration."""

        return (f"dp{config.dp_size}_pp{config.pp_size}_tp{config.tp_size}_"
                f"z{config.zero_stage}_mbs{config.micro_batch_size}_"
                f"ga{config.gradient_accumulation_steps}")

    def _setup_distributed_model(self, config: TrainingConfig) -> nn.Module:
        """Set up model with given parallelism configuration."""
        # Implementation depends on framework (DeepSpeed, Megatron, etc.)
        raise NotImplementedError

    def _setup_optimizer(self, model: nn.Module, config: TrainingConfig):
        """Set up optimizer with proper ZeRO configuration."""
        raise NotImplementedError

    def _setup_dataloader(self, config: TrainingConfig):
        """Set up dataloader with proper batch sizing."""
        raise NotImplementedError

    def _train_step(self, model, optimizer, batch):
        """Execute one training step."""
        raise NotImplementedError
```

## Configuration Recommendations

### Rule-Based Initial Guess

Before expensive search, use heuristics for a good starting point:

```python
class ConfigurationAdvisor:
    """Rule-based configuration recommendations."""

    def recommend(self,
                  model_params: int,
                  hidden_dim: int,
                  num_layers: int,
                  num_heads: int,
                  seq_length: int,
                  target_gpus: int,
                  target_batch_size: int,
                  gpu_memory_gb: float = 80.0) -> TrainingConfig:
        """Recommend initial configuration based on heuristics."""

        # Rule 1: TP size based on model width
        # Use TP if model is too wide for single GPU
        bytes_per_param = 16  # fp16 params+grads + fp32 master/m/v (AdamW)
        params_per_gpu = model_params * bytes_per_param
        if params_per_gpu > gpu_memory_gb * 1e9:
            tp_size = min(8, self._ceil_power_of_2(
                params_per_gpu / (gpu_memory_gb * 1e9 * 0.5)
            ))
        else:
            tp_size = 1

        # Ensure divisibility
        while num_heads % tp_size != 0 and tp_size > 1:
            tp_size //= 2

        # Rule 2: PP size based on layers and remaining GPUs
        remaining_gpus = target_gpus // tp_size

        if num_layers >= 64 and remaining_gpus >= 8:
            pp_size = min(8, num_layers // 8)
        elif num_layers >= 32 and remaining_gpus >= 4:
            pp_size = 4
        else:
            pp_size = 1

        # Ensure divisibility
        while num_layers % pp_size != 0 and pp_size > 1:
            pp_size //= 2

        # Rule 3: DP fills the rest
        dp_size = target_gpus // (tp_size * pp_size)

        # Rule 4: ZeRO based on model size and DP
        bytes_per_param = 2
        params_per_dp = model_params / tp_size / pp_size
        mem_per_dp = params_per_dp * (bytes_per_param + 12)  # Model + optimizer

        if mem_per_dp > gpu_memory_gb * 1e9 * 0.8:
            if dp_size >= 8:
                zero_stage = 3
            elif dp_size >= 4:
                zero_stage = 2
            else:
                zero_stage = 1
        else:
            zero_stage = 1  # ZeRO-1 is usually beneficial

        # Rule 5: Micro-batch size
        # Start small, increase until memory limit
        micro_batch_size = 1
        activation_mem = micro_batch_size * seq_length * hidden_dim * 2
        layers_per_stage = num_layers // pp_size

        while (activation_mem * layers_per_stage <
               gpu_memory_gb * 1e9 * 0.2 and micro_batch_size < 32):
            micro_batch_size *= 2
            activation_mem = micro_batch_size * seq_length * hidden_dim * 2

        micro_batch_size //= 2  # Back off one step

        # Rule 6: Gradient accumulation
        per_dp_batch = target_batch_size // dp_size
        gradient_accumulation = max(1, per_dp_batch // micro_batch_size)

        # Rule 7: Activation checkpointing if large model
        activation_checkpointing = model_params > 1e9

        return TrainingConfig(
            dp_size=dp_size,
            pp_size=pp_size,
            tp_size=tp_size,
            cp_size=1,
            ep_size=1,
            zero_stage=zero_stage,
            activation_checkpointing=activation_checkpointing,
            offload_optimizer=False,
            offload_params=False,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            seq_length=seq_length,
            hidden_dim=hidden_dim,
            pipeline_schedule="interleaved" if pp_size >= 4 else "1F1B",
            num_interleaved_stages=2 if pp_size >= 4 else 1,
            overlap_comm_compute=True,
            bucket_size_mb=25,
            precision="bf16",
        )

    def _ceil_power_of_2(self, x: float) -> int:
        """Round up to nearest power of 2."""
        return 2 ** int(np.ceil(np.log2(max(1, x))))
```

### Case Studies

#### GPT-3 175B on 1024 A100s

```python
# Model specification
model_params = 175e9
hidden_dim = 12288
num_layers = 96
num_heads = 96
seq_length = 2048
vocab_size = 50257

# Hardware
target_gpus = 1024
gpu_memory = 80  # GB

# Recommended configuration
config = TrainingConfig(
    dp_size=64,
    pp_size=8,
    tp_size=2,
    cp_size=1,
    ep_size=1,
    zero_stage=1,
    activation_checkpointing=True,
    offload_optimizer=False,
    offload_params=False,
    micro_batch_size=2,
    gradient_accumulation_steps=8,
    seq_length=seq_length,
    hidden_dim=hidden_dim,
    pipeline_schedule="interleaved",
    num_interleaved_stages=2,
    overlap_comm_compute=True,
    bucket_size_mb=25,
    precision="bf16",
)

# Global batch size: 64 * 2 * 8 = 1024 samples
# Per-GPU memory: ~60GB (fits in 80GB)
# Predicted efficiency: ~50% MFU
```

#### LLaMA 70B on 128 H100s

```python
# Model specification
model_params = 70e9
hidden_dim = 8192
num_layers = 80
num_heads = 64
seq_length = 8192  # Long context
vocab_size = 32000

# Hardware
target_gpus = 128
gpu_memory = 80

# Recommended configuration
config = TrainingConfig(
    dp_size=32,
    pp_size=4,
    tp_size=1,  # GQA reduces TP need
    cp_size=1,
    ep_size=1,
    zero_stage=1,
    activation_checkpointing=True,
    offload_optimizer=False,
    offload_params=False,
    micro_batch_size=1,
    gradient_accumulation_steps=4,
    seq_length=seq_length,
    hidden_dim=hidden_dim,
    pipeline_schedule="1F1B",
    num_interleaved_stages=1,
    overlap_comm_compute=True,
    bucket_size_mb=25,
    precision="bf16",
)

# Global batch size: 32 * 1 * 4 = 128 samples
# Tokens per batch: 128 * 8192 = 1M tokens
```

## Exercises

1. **Configuration counting**: For 512 GPUs, with DP ∈ {1, 2, 4, ..., 512}, PP ∈ {1, 2, 4, 8}, TP ∈ {1, 2, 4, 8}, how many valid (DP, PP, TP) combinations exist?

??? success "Solution"
    **Constraint:**

    $$DP \times PP \times TP = 512$$

    **Available values:**

    - DP ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512} (10 powers of 2)
    - PP ∈ {1, 2, 4, 8}
    - TP ∈ {1, 2, 4, 8}

    **Enumerate valid combinations:**

    For each (PP, TP) pair, find DP = 512 / (PP × TP):

    | PP | TP | PP × TP | DP = 512/(PP×TP) | Valid? |
    |----|----|---------|-----------------:|--------|
    | 1 | 1 | 1 | 512 | ✓ |
    | 1 | 2 | 2 | 256 | ✓ |
    | 1 | 4 | 4 | 128 | ✓ |
    | 1 | 8 | 8 | 64 | ✓ |
    | 2 | 1 | 2 | 256 | ✓ |
    | 2 | 2 | 4 | 128 | ✓ |
    | 2 | 4 | 8 | 64 | ✓ |
    | 2 | 8 | 16 | 32 | ✓ |
    | 4 | 1 | 4 | 128 | ✓ |
    | 4 | 2 | 8 | 64 | ✓ |
    | 4 | 4 | 16 | 32 | ✓ |
    | 4 | 8 | 32 | 16 | ✓ |
    | 8 | 1 | 8 | 64 | ✓ |
    | 8 | 2 | 16 | 32 | ✓ |
    | 8 | 4 | 32 | 16 | ✓ |
    | 8 | 8 | 64 | 8 | ✓ |

    All 16 combinations yield valid DP values (all are powers of 2 ≤ 512).

    **Answer:**

    $$\boxed{16 \text{ valid configurations}}$$

    **Verification:** $|PP| \times |TP| = 4 \times 4 = 16$

    All combinations are valid because:
    - PP × TP ∈ {1, 2, 4, 8, 16, 32, 64} ⊆ divisors of 512
    - All resulting DP values are powers of 2 ≤ 512

2. **Memory constraint**: A 30B parameter model with 40 layers, hidden dimension 7168. Using TP=4, PP=4, ZeRO-1 on a DP=16 cluster. Calculate memory per GPU (assume BF16 + FP32 optimizer).

??? success "Solution"
    **Given:**

    - Ψ = 30B parameters
    - L = 40 layers
    - H = 7168 hidden dimension
    - TP = 4, PP = 4, DP = 16
    - Total GPUs = 4 × 4 × 16 = 256

    **Parameter distribution:**

    $$\Psi_{GPU} = \frac{\Psi}{TP \times PP} = \frac{30B}{4 \times 4} = 1.875B \text{ parameters/GPU}$$

    **Static memory components:**

    | Component | Formula | Per-GPU Memory |
    |-----------|---------|----------------|
    | Parameters (BF16) | $2 \times \Psi_{GPU}$ | 3.75 GB |
    | Gradients (BF16) | $2 \times \Psi_{GPU}$ | 3.75 GB |
    | Optimizer (FP32) | $12 \times \Psi_{GPU} / DP$ (ZeRO-1) | 1.41 GB |

    **Optimizer memory with ZeRO-1:**

    ZeRO-1 shards optimizer states across DP dimension:

    $$M_{opt}^{ZeRO1} = \frac{12 \times 1.875B}{16} = 1.41 \text{ GB}$$

    **Activation memory:**

    Per-layer activation (with TP=4):

    $$M_{act}^{layer} = \frac{BSH \times 34}{TP}$$

    Assuming B=4, S=4096:

    $$M_{act}^{layer} = \frac{4 \times 4096 \times 7168 \times 34}{4} = 997 \text{ MB/layer}$$

    Layers per PP stage: $40/4 = 10$ layers

    With activation checkpointing (every layer):

    $$M_{act}^{total} \approx 2 \times 997 = 1.99 \text{ GB}$$

    Without checkpointing:

    $$M_{act}^{total} = 10 \times 997 = 9.97 \text{ GB}$$

    **Total memory per GPU:**

    | Component | Memory |
    |-----------|--------|
    | Parameters (BF16) | 3.75 GB |
    | Gradients (BF16) | 3.75 GB |
    | Optimizer (ZeRO-1) | 1.41 GB |
    | Activations (ckpt) | ~2 GB |
    | Working buffers | ~2 GB |
    | **Total** | $\boxed{\sim 13 \text{ GB}}$ |

    Without checkpointing:

    | Component | Memory |
    |-----------|--------|
    | Parameters (BF16) | 3.75 GB |
    | Gradients (BF16) | 3.75 GB |
    | Optimizer (ZeRO-1) | 1.41 GB |
    | Activations | ~10 GB |
    | Working buffers | ~2 GB |
    | **Total** | $\boxed{\sim 21 \text{ GB}}$ |

    Both fit comfortably in 80GB H100.

3. **Search space reduction**: You have 10,000 configurations. Applying GPU count constraint removes 80%, divisibility removes 50% of remainder, memory removes 30% of remainder. How many configs remain?

??? success "Solution"
    **Initial configuration count:** 10,000

    **Stage 1: GPU count constraint**

    Removes 80%:

    $$N_1 = 10000 \times (1 - 0.80) = 10000 \times 0.20 = 2000$$

    **Stage 2: Divisibility constraint**

    Removes 50% of remainder:

    $$N_2 = 2000 \times (1 - 0.50) = 2000 \times 0.50 = 1000$$

    **Stage 3: Memory constraint**

    Removes 30% of remainder:

    $$N_3 = 1000 \times (1 - 0.30) = 1000 \times 0.70 = 700$$

    **Answer:**

    $$\boxed{700 \text{ configurations remain}}$$

    **Cumulative reduction:**

    $$\text{Survival rate} = 0.20 \times 0.50 \times 0.70 = 0.07 = 7\%$$

    **Visualization:**

    | Stage | Configs In | Removed | Configs Out |
    |-------|------------|---------|-------------|
    | Initial | - | - | 10,000 |
    | GPU count | 10,000 | 8,000 (80%) | 2,000 |
    | Divisibility | 2,000 | 1,000 (50%) | 1,000 |
    | Memory | 1,000 | 300 (30%) | 700 |

    This demonstrates why constraint pruning is so effective—even moderate rejection rates compound to eliminate 93% of the search space.

4. **Bayesian vs Grid**: Grid search evaluates 1000 configurations in 100 hours. Bayesian optimization achieves same best result in 50 evaluations. If each eval takes 6 minutes, calculate time savings.

??? success "Solution"
    **Grid search:**

    - Evaluations: 1,000
    - Total time: 100 hours

    Time per evaluation: $\frac{100 \times 60}{1000} = 6$ minutes ✓ (matches given)

    **Bayesian optimization:**

    - Evaluations needed: 50
    - Time per evaluation: 6 minutes

    Total time:

    $$T_{bayes} = 50 \times 6 = 300 \text{ minutes} = \boxed{5 \text{ hours}}$$

    **Time savings:**

    $$\text{Savings} = 100 - 5 = \boxed{95 \text{ hours}}$$

    $$\text{Speedup} = \frac{100}{5} = \boxed{20\times}$$

    **Efficiency comparison:**

    | Method | Evaluations | Time | Result |
    |--------|-------------|------|--------|
    | Grid search | 1,000 | 100 hours | Best config |
    | Bayesian opt | 50 | 5 hours | Same best config |
    | **Savings** | **95%** | **95%** | - |

    **Why Bayesian is so effective:**

    1. **Exploitation**: Focuses on promising regions
    2. **Exploration**: Samples uncertain areas
    3. **Model-guided**: Learns from each evaluation

    **Cost analysis at $4/GPU-hour:**

    Assuming 64 GPUs per evaluation:

    | Method | GPU-hours | Cost |
    |--------|-----------|------|
    | Grid | 64 × 100 = 6,400 | $25,600 |
    | Bayesian | 64 × 5 = 320 | $1,280 |
    | **Savings** | 6,080 | **$24,320** |

5. **Heuristic validation**: Use the ConfigurationAdvisor to generate a config for a 13B parameter model (40 layers, h=5120, 40 heads) on 64 GPUs. Then validate with the MemoryConstraint.

??? success "Solution"
    **Model specification:**

    - Ψ = 13B parameters
    - L = 40 layers
    - H = 5120 hidden dimension
    - A = 40 attention heads
    - GPUs = 64

    **ConfigurationAdvisor heuristics:**

    **Step 1: TP selection**

    Model size → TP heuristic:
    - 13B is medium-sized
    - Hidden dim 5120 is divisible by 8
    - Prefer TP ≤ 8 (within a node)

    $$\boxed{TP = 4}$$ (keeps layers reasonably sized)

    **Step 2: PP selection**

    After TP, remaining GPUs: 64/4 = 16

    Layers per stage with various PP:
    - PP=1: 40 layers/stage (no pipeline)
    - PP=2: 20 layers/stage
    - PP=4: 10 layers/stage
    - PP=8: 5 layers/stage

    Balance: 40 must divide evenly → PP ∈ {1, 2, 4, 5, 8, 10, 20, 40}

    Constrained by 64/TP divisibility: PP ∈ {1, 2, 4, 8, 16}

    $$\boxed{PP = 4}$$ (10 layers per stage, good balance)

    **Step 3: DP selection**

    $$DP = \frac{64}{TP \times PP} = \frac{64}{4 \times 4} = \boxed{4}$$

    **Generated configuration:**

    | Dimension | Value |
    |-----------|-------|
    | TP | 4 |
    | PP | 4 |
    | DP | 4 |
    | Total | 64 ✓ |

    **Memory validation:**

    Parameters per GPU:

    $$\Psi_{GPU} = \frac{13B}{TP \times PP} = \frac{13B}{16} = 812.5M$$

    **Memory breakdown:**

    | Component | Formula | Memory |
    |-----------|---------|--------|
    | Parameters (BF16) | $2 \times 812.5M$ | 1.63 GB |
    | Gradients (BF16) | $2 \times 812.5M$ | 1.63 GB |
    | Optimizer (FP32) | $12 \times 812.5M$ | 9.75 GB |
    | **Static total** | | **13.0 GB** |

    With ZeRO-1 (DP=4):

    $$M_{opt}^{ZeRO1} = 9.75 / 4 = 2.44 \text{ GB}$$
    Static with ZeRO-1: 5.7 GB

    **Activation memory (B=8, S=4096):**

    $$M_{act}^{layer} = \frac{8 \times 4096 \times 5120 \times 34}{4} = 1.43 \text{ GB/layer}$$

    Layers per stage: 10

    With checkpointing:

    $$M_{act} \approx 2 \times 1.43 = 2.86 \text{ GB}$$

    **Total per GPU:**

    | Component | Memory |
    |-----------|--------|
    | Static (ZeRO-1) | 5.7 GB |
    | Activations (ckpt) | ~3 GB |
    | Working buffers | ~2 GB |
    | **Total** | **~11 GB** |

    **Validation result:**

    $$11 \text{ GB} < 80 \text{ GB (H100)} \quad \boxed{\checkmark \text{ PASS}}$$

    Plenty of headroom for larger batch sizes.

6. **Pipeline bubble trade-off**: Compare configs (PP=4, GA=16) vs (PP=8, GA=32) for bubble fraction. At what GA does the higher PP become advantageous?

??? success "Solution"
    **Bubble fraction formula:**

    $$\text{Bubble} = \frac{PP - 1}{GA + PP - 1}$$

    **Config 1: PP=4, GA=16**

    $$\text{Bubble}_1 = \frac{4 - 1}{16 + 4 - 1} = \frac{3}{19} = 15.8\%$$

    **Config 2: PP=8, GA=32**

    $$\text{Bubble}_2 = \frac{8 - 1}{32 + 8 - 1} = \frac{7}{39} = 17.9\%$$

    **Comparison:**

    | Config | PP | GA | Bubble Fraction |
    |--------|----|----|-----------------|
    | 1 | 4 | 16 | 15.8% |
    | 2 | 8 | 32 | 17.9% |

    Config 1 (PP=4, GA=16) has lower bubble fraction.

    **When does PP=8 become advantageous?**

    For PP=8 to have lower bubble than PP=4 with GA=16:

    $$\frac{7}{GA + 7} < \frac{3}{19}$$

    Solve:

    $$7 \times 19 < 3 \times (GA + 7)$$

    $$133 < 3 \cdot GA + 21$$

    $$112 < 3 \cdot GA$$

    $$GA > 37.3$$

    $$\boxed{GA \geq 38}$$ for PP=8 to have lower bubble than PP=4/GA=16

    **Verification at GA=38:**

    $$\text{Bubble}_{PP=8,GA=38} = \frac{7}{38 + 7} = \frac{7}{45} = 15.6\%$$

    vs PP=4, GA=16: 15.8% ✓

    **General formula for breakeven:**

    For PP₂ to have same bubble as PP₁ with GA₁:

    $$\frac{PP_2 - 1}{GA_2 + PP_2 - 1} = \frac{PP_1 - 1}{GA_1 + PP_1 - 1}$$

    Solving for GA₂:

    $$GA_2 = \frac{(PP_2 - 1)(GA_1 + PP_1 - 1)}{PP_1 - 1} - (PP_2 - 1)$$

    **Trade-off analysis:**

    | PP | GA for 10% bubble | GA for 5% bubble |
    |----|-------------------|------------------|
    | 4 | 27 | 57 |
    | 8 | 63 | 133 |
    | 16 | 135 | 285 |

    **When to prefer higher PP:**

    - Higher PP enables smaller per-GPU memory (more sharding)
    - But requires more gradient accumulation to maintain efficiency
    - Higher PP also has more communication overhead between stages

    **Summary:**

    $$\boxed{GA \geq 38 \text{ for PP=8 to beat PP=4/GA=16}}$$

## Key Takeaways

1. **Configuration space is exponential**: Systematic search is required, not trial-and-error.

2. **Constraints prune aggressively**: Memory, bandwidth, and divisibility constraints eliminate most configurations early.

3. **Cost models enable fast search**: Predicting performance analytically is much cheaper than running each configuration.

4. **Hierarchical search is effective**: Optimize parallelism dimensions first, then memory optimizations, then micro-batching.

5. **Bayesian optimization beats grid search**: For expensive evaluations, adaptive methods find optima with fewer trials.

6. **Heuristics provide good starting points**: Rule-based recommendations get within 20% of optimal quickly.

7. **Validation is essential**: Cost models must be calibrated against actual measurements.

8. **The search is worth it**: A 2× throughput improvement saves months of training time at scale.

