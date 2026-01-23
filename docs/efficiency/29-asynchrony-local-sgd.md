---
title: "Asynchrony and Local SGD"
subtitle: "Breaking the Synchronization Barrier"
---

<div class="chapter-opener" markdown>
Synchronous training is mathematically clean but operationally costly. Every worker waits for the slowest. Asynchronous methods eliminate this barrier, but introduce staleness. Local SGD finds a middle ground: synchronize periodically rather than constantly. The mathematics of these trade-offs reveals when each approach wins.
</div>

<div class="investigation-question" markdown>
**The Question**: If workers can compute gradients in parallel without waiting, why does asynchronous training often converge slower than synchronous? What's lost in translation, and can we recover it?
</div>

## The Synchronization Tax

In synchronous data parallelism, each step requires:

```
All workers:   [Compute gradient]
               [Wait for slowest]
               [AllReduce]
               [Update weights]
               [Barrier]
```

The slowest worker determines throughput:

$$T_{\text{step}} = \max_{i \in [P]} T_{\text{compute}}^{(i)} + T_{\text{comm}}$$

### The Straggler Problem

Worker compute times vary due to:

- Hardware variance (thermal throttling, memory speeds)
- Software variance (garbage collection, OS scheduling)
- Data variance (variable-length sequences, dynamic computation)

Let $T_i \sim \text{Distribution}$ be worker $i$'s compute time.

For $P$ workers with i.i.d. times:
$$\mathbb{E}[\max_i T_i] > \mathbb{E}[T_i]$$

The gap grows with $P$. For exponential distribution:
$$\mathbb{E}[\max_i T_i] = H_P \cdot \mathbb{E}[T_i]$$

Where $H_P = 1 + 1/2 + ... + 1/P \approx \ln P$ is the harmonic number.

**With 1000 workers**: Expected straggler delay is ~7× average compute time!

## Asynchronous SGD

Remove the synchronization barrier entirely.

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import threading
import queue
import time


@dataclass
class ParameterServer:
    """
    Central parameter server for asynchronous SGD.

    Workers push gradients asynchronously; server applies immediately.
    """

    def __init__(self, initial_weights: Dict[str, np.ndarray],
                 learning_rate: float):
        self.weights = {k: v.copy() for k, v in initial_weights.items()}
        self.lr = learning_rate
        self.lock = threading.Lock()
        self.version = 0
        self.gradient_queue = queue.Queue()

    def get_weights(self) -> tuple:
        """Get current weights and version."""
        with self.lock:
            return {k: v.copy() for k, v in self.weights.items()}, self.version

    def push_gradient(self, gradients: Dict[str, np.ndarray],
                      worker_version: int):
        """
        Apply gradient update asynchronously.

        Args:
            gradients: Computed gradients
            worker_version: Version of weights used to compute gradients
        """
        with self.lock:
            staleness = self.version - worker_version

            # Apply update
            for name, grad in gradients.items():
                self.weights[name] -= self.lr * grad

            self.version += 1

            return staleness


class AsyncWorker:
    """Asynchronous SGD worker."""

    def __init__(self, worker_id: int, server: ParameterServer,
                 data_iterator, model_fn, loss_fn):
        self.worker_id = worker_id
        self.server = server
        self.data_iter = data_iterator
        self.model_fn = model_fn
        self.loss_fn = loss_fn
        self.running = False

    def train_loop(self, num_steps: int):
        """
        Asynchronous training loop.

        No synchronization with other workers.
        """
        self.running = True

        for step in range(num_steps):
            if not self.running:
                break

            # 1. Pull current weights
            weights, version = self.server.get_weights()

            # 2. Compute gradient on local batch
            batch = next(self.data_iter)
            gradients = self._compute_gradient(weights, batch)

            # 3. Push gradient (staleness may have accumulated)
            staleness = self.server.push_gradient(gradients, version)

            if step % 100 == 0:
                print(f"Worker {self.worker_id}, step {step}, "
                      f"staleness {staleness}")

    def _compute_gradient(self, weights: Dict[str, np.ndarray],
                          batch) -> Dict[str, np.ndarray]:
        """Compute gradient using local weights."""
        # Forward pass
        predictions = self.model_fn(weights, batch['x'])
        loss = self.loss_fn(predictions, batch['y'])

        # Backward pass (simplified)
        gradients = {}
        for name in weights:
            # Numerical gradient for illustration
            eps = 1e-5
            weights_plus = weights.copy()
            weights_plus[name] = weights[name] + eps
            loss_plus = self.loss_fn(
                self.model_fn(weights_plus, batch['x']), batch['y']
            )
            gradients[name] = (loss_plus - loss) / eps

        return gradients


def run_async_training(num_workers: int, num_steps: int,
                       initial_weights: Dict[str, np.ndarray],
                       lr: float, data_iterators: list,
                       model_fn, loss_fn):
    """Launch asynchronous training."""
    server = ParameterServer(initial_weights, lr)

    workers = []
    threads = []

    for i in range(num_workers):
        worker = AsyncWorker(i, server, data_iterators[i], model_fn, loss_fn)
        workers.append(worker)

        thread = threading.Thread(target=worker.train_loop, args=(num_steps,))
        threads.append(thread)

    # Start all workers
    for thread in threads:
        thread.start()

    # Wait for completion
    for thread in threads:
        thread.join()

    return server.get_weights()[0]
```

### Staleness and Convergence

The key difference from synchronous SGD: gradients are computed on **stale weights**.

Worker $i$ computes $g_i = \nabla L(w^{(t-\tau_i)}, x_i)$ but applies to $w^{(t)}$.

**Staleness** $\tau_i$ is the number of updates since worker pulled weights.

With $P$ workers and equal compute times:
$$\mathbb{E}[\tau] = P - 1$$

**Convergence impact**: Stale gradients introduce bias.

For smooth non-convex functions:
$$\mathbb{E}[||w^{(T)} - w^*||^2] = O\left(\frac{1}{\sqrt{T}}\right) + O(\tau^2 \eta^2 G^2)$$

Where:

- $\tau$ = maximum staleness
- $\eta$ = learning rate
- $G$ = gradient bound

**Key insight**: Must reduce learning rate to compensate for staleness:
$$\eta_{\text{async}} = \frac{\eta_{\text{sync}}}{\tau}$$

This partially negates the throughput advantage!

### Staleness-Adaptive Learning Rate

```python
class StalenessAdaptiveServer(ParameterServer):
    """
    Parameter server with staleness-aware learning rate.

    Reduces learning rate proportionally to gradient staleness.
    """

    def __init__(self, initial_weights: Dict[str, np.ndarray],
                 base_lr: float, staleness_discount: float = 0.9):
        super().__init__(initial_weights, base_lr)
        self.base_lr = base_lr
        self.discount = staleness_discount

    def push_gradient(self, gradients: Dict[str, np.ndarray],
                      worker_version: int):
        """Apply gradient with staleness-adjusted learning rate."""
        with self.lock:
            staleness = self.version - worker_version

            # Discount learning rate based on staleness
            # Option 1: Linear decay
            # effective_lr = self.base_lr / (1 + staleness)

            # Option 2: Exponential decay
            effective_lr = self.base_lr * (self.discount ** staleness)

            # Apply update
            for name, grad in gradients.items():
                self.weights[name] -= effective_lr * grad

            self.version += 1

            return staleness


class BoundedStalenessServer(ParameterServer):
    """
    Parameter server that bounds staleness.

    Workers must wait if their weights are too stale.
    """

    def __init__(self, initial_weights: Dict[str, np.ndarray],
                 learning_rate: float, max_staleness: int):
        super().__init__(initial_weights, learning_rate)
        self.max_staleness = max_staleness
        self.waiting_workers = []
        self.condition = threading.Condition(self.lock)

    def push_gradient(self, gradients: Dict[str, np.ndarray],
                      worker_version: int):
        """Apply gradient, potentially waiting if too stale."""
        with self.condition:
            # Wait until staleness is acceptable
            while self.version - worker_version > self.max_staleness:
                self.condition.wait()

            staleness = self.version - worker_version

            # Apply update
            for name, grad in gradients.items():
                self.weights[name] -= self.lr * grad

            self.version += 1

            # Wake up waiting workers
            self.condition.notify_all()

            return staleness
```

## Hogwild!

For sparse gradients, lock-free asynchronous updates are possible.

**Key insight**: If gradient updates touch different parameters with high probability, lock contention is rare.

```python
import numpy as np
from numpy.lib.stride_tricks import as_strided

class HogwildSGD:
    """
    Lock-free asynchronous SGD for sparse problems.

    Recht et al. (2011) showed that for sparse problems,
    allowing race conditions actually works!

    Note: Requires problems where gradients are sparse and
    have low overlap probability.
    """

    def __init__(self, num_features: int, learning_rate: float):
        # Shared memory (no locks!)
        self.weights = np.zeros(num_features, dtype=np.float64)
        self.lr = learning_rate

    def update(self, indices: np.ndarray, gradient_values: np.ndarray):
        """
        Apply sparse gradient update without locking.

        Race conditions are okay! The math still works out
        (in expectation) for sparse enough gradients.
        """
        # Atomic-ish updates (numpy operations are often atomic)
        # In practice, use specialized atomic ops or accept races
        self.weights[indices] -= self.lr * gradient_values

    def get_weights(self) -> np.ndarray:
        """Read current weights (may be inconsistent)."""
        return self.weights.copy()


class SparseGradientWorker:
    """Worker for Hogwild! training."""

    def __init__(self, worker_id: int, hogwild: HogwildSGD,
                 data_iterator, gradient_fn):
        self.worker_id = worker_id
        self.hogwild = hogwild
        self.data_iter = data_iterator
        self.gradient_fn = gradient_fn

    def train_loop(self, num_steps: int):
        """Training loop with lock-free updates."""
        for step in range(num_steps):
            # Get current weights (may be slightly stale)
            weights = self.hogwild.get_weights()

            # Compute sparse gradient
            batch = next(self.data_iter)
            indices, values = self.gradient_fn(weights, batch)

            # Update without locking
            self.hogwild.update(indices, values)
```

**When Hogwild! works**:

- Sparsity: $\mathbb{E}[|\text{supp}(g_i) \cap \text{supp}(g_j)|] \ll d$
- Examples: matrix factorization, sparse logistic regression

**When it fails**:

- Dense gradients (neural networks)
- High update frequency on same parameters

## Local SGD

A middle ground: synchronize periodically rather than every step.

$$\text{Local SGD}(H) = \begin{cases}
\text{Local updates} & \text{for } H-1 \text{ steps} \\
\text{AllReduce (average)} & \text{every } H \text{ steps}
\end{cases}$$

```python
from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class LocalSGDConfig:
    """Configuration for Local SGD."""
    num_workers: int
    local_steps: int  # H: steps between synchronization
    learning_rate: float


class LocalSGDWorker:
    """
    Local SGD worker: accumulate updates, sync periodically.

    Also known as:

    - Federated Averaging (FedAvg) in federated learning
    - Periodic averaging SGD
    """

    def __init__(self, worker_id: int, config: LocalSGDConfig,
                 initial_weights: Dict[str, np.ndarray]):
        self.worker_id = worker_id
        self.config = config
        self.local_weights = {k: v.copy() for k, v in initial_weights.items()}
        self.step_in_epoch = 0

    def local_step(self, gradient: Dict[str, np.ndarray]):
        """Take a local SGD step."""
        for name, grad in gradient.items():
            self.local_weights[name] -= self.config.learning_rate * grad
        self.step_in_epoch += 1

    def should_sync(self) -> bool:
        """Check if it's time to synchronize."""
        return self.step_in_epoch >= self.config.local_steps

    def get_weights_for_sync(self) -> Dict[str, np.ndarray]:
        """Get weights to contribute to averaging."""
        return self.local_weights

    def receive_averaged_weights(self, averaged: Dict[str, np.ndarray]):
        """Update local weights with global average."""
        self.local_weights = {k: v.copy() for k, v in averaged.items()}
        self.step_in_epoch = 0


def local_sgd_training(workers: List[LocalSGDWorker],
                       data_iterators: list,
                       gradient_fn,
                       total_steps: int,
                       allreduce_fn) -> Dict[str, np.ndarray]:
    """
    Run Local SGD training.

    Args:
        workers: List of LocalSGDWorker instances
        data_iterators: Per-worker data iterators
        gradient_fn: Function(weights, batch) -> gradients
        total_steps: Total training steps
        allreduce_fn: Function to average weights across workers

    Returns:
        Final averaged weights
    """
    num_workers = len(workers)
    step = 0

    while step < total_steps:
        # Each worker takes local steps
        for worker, data_iter in zip(workers, data_iterators):
            batch = next(data_iter)
            gradient = gradient_fn(worker.local_weights, batch)
            worker.local_step(gradient)

        step += 1

        # Check if time to sync
        if workers[0].should_sync():
            # Gather all weights
            all_weights = [w.get_weights_for_sync() for w in workers]

            # Average (AllReduce simulation)
            averaged = {}
            for name in all_weights[0]:
                stacked = np.stack([w[name] for w in all_weights])
                averaged[name] = np.mean(stacked, axis=0)

            # Distribute averaged weights
            for worker in workers:
                worker.receive_averaged_weights(averaged)

            print(f"Synced at step {step}")

    return workers[0].local_weights
```

### Convergence Analysis

**Theorem** (Stich, 2018): For smooth non-convex functions, Local SGD with $H$ local steps converges at rate:

$$\mathbb{E}\left[\frac{1}{T}\sum_{t=1}^T ||\nabla f(w_t)||^2\right] \leq O\left(\frac{1}{\sqrt{HT}}\right) + O\left(\frac{H\sigma^2}{T}\right)$$

Where:

- First term: standard SGD convergence
- Second term: penalty from local divergence

**Key insight**: $H$ can be much larger than 1 while maintaining convergence!

**Optimal $H$**: Balance communication savings against drift penalty:
$$H^* = O\left(\sqrt{\frac{T}{\sigma^2}}\right)$$

As training progresses ($T$ increases), can use larger $H$.

### Client Drift and Correction

In heterogeneous settings (different data distributions), local models drift apart.

```python
class LocalSGDWithMomentumCorrection:
    """
    Local SGD with SCAFFOLD-style variance reduction.

    Tracks control variates to correct for client drift.
    """

    def __init__(self, worker_id: int, config: LocalSGDConfig,
                 initial_weights: Dict[str, np.ndarray]):
        self.worker_id = worker_id
        self.config = config
        self.local_weights = {k: v.copy() for k, v in initial_weights.items()}

        # Control variates (SCAFFOLD)
        self.local_control = {k: np.zeros_like(v) for k, v in initial_weights.items()}
        self.global_control = {k: np.zeros_like(v) for k, v in initial_weights.items()}

        self.step_in_epoch = 0

    def local_step(self, gradient: Dict[str, np.ndarray]):
        """
        Take corrected local step.

        Uses control variate: g - c_i + c (where c = global, c_i = local)
        """
        for name, grad in gradient.items():
            # Corrected gradient
            correction = self.global_control[name] - self.local_control[name]
            corrected_grad = grad + correction

            self.local_weights[name] -= self.config.learning_rate * corrected_grad

        self.step_in_epoch += 1

    def update_control_variate(self, global_control: Dict[str, np.ndarray],
                               steps_taken: int):
        """Update control variates after sync."""
        for name in self.local_weights:
            # New local control = old + (1/H)(gradient_sum)
            # Approximated by the difference
            delta = self.global_control[name] - global_control[name]
            self.local_control[name] = self.local_control[name] - delta

        self.global_control = {k: v.copy() for k, v in global_control.items()}


class FedProx:
    """
    FedProx: Local SGD with proximal regularization.

    Adds regularization term to keep local model close to global.
    """

    def __init__(self, worker_id: int, config: LocalSGDConfig,
                 initial_weights: Dict[str, np.ndarray], mu: float = 0.01):
        self.worker_id = worker_id
        self.config = config
        self.mu = mu  # Proximal coefficient

        self.local_weights = {k: v.copy() for k, v in initial_weights.items()}
        self.global_weights = {k: v.copy() for k, v in initial_weights.items()}
        self.step_in_epoch = 0

    def local_step(self, gradient: Dict[str, np.ndarray]):
        """
        Proximal local step.

        Loss = original_loss + (μ/2) * ||w - w_global||²
        Gradient += μ * (w - w_global)
        """
        for name, grad in gradient.items():
            # Proximal term gradient
            prox_grad = self.mu * (self.local_weights[name] - self.global_weights[name])
            total_grad = grad + prox_grad

            self.local_weights[name] -= self.config.learning_rate * total_grad

        self.step_in_epoch += 1

    def receive_averaged_weights(self, averaged: Dict[str, np.ndarray]):
        """Update both local and global reference."""
        self.local_weights = {k: v.copy() for k, v in averaged.items()}
        self.global_weights = {k: v.copy() for k, v in averaged.items()}
        self.step_in_epoch = 0
```

## DiLoCo: Distributed Low-Communication

Douillard et al. (2023) applied Local SGD to LLM pre-training.

```python
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class DiLoCoConfig:
    """Configuration for DiLoCo distributed training."""
    num_workers: int
    inner_steps: int  # H: steps between outer updates
    inner_optimizer: str  # "sgd" or "adam"
    outer_optimizer: str  # For averaging step ("sgd" or "nesterov")
    inner_lr: float
    outer_lr: float
    outer_momentum: float


class DiLoCoWorker:
    """
    DiLoCo worker for low-communication LLM training.

    Key insight: Use different optimizers for inner (local) and
    outer (sync) updates.
    """

    def __init__(self, worker_id: int, config: DiLoCoConfig,
                 initial_weights: Dict[str, np.ndarray]):
        self.worker_id = worker_id
        self.config = config

        # Current weights
        self.weights = {k: v.copy() for k, v in initial_weights.items()}

        # Weights at last sync (for computing pseudo-gradient)
        self.sync_weights = {k: v.copy() for k, v in initial_weights.items()}

        # Inner optimizer state (Adam)
        self.m = {k: np.zeros_like(v) for k, v in initial_weights.items()}
        self.v = {k: np.zeros_like(v) for k, v in initial_weights.items()}
        self.inner_step = 0

        # Outer optimizer state (Nesterov momentum)
        self.outer_momentum = {k: np.zeros_like(v) for k, v in initial_weights.items()}

    def inner_update(self, gradient: Dict[str, np.ndarray]):
        """
        Inner optimizer step (Adam).

        Standard Adam update on local worker.
        """
        self.inner_step += 1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8

        for name, grad in gradient.items():
            # Adam moments
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * grad
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[name] / (1 - beta1 ** self.inner_step)
            v_hat = self.v[name] / (1 - beta2 ** self.inner_step)

            # Update
            self.weights[name] -= self.config.inner_lr * m_hat / (np.sqrt(v_hat) + eps)

    def compute_pseudo_gradient(self) -> Dict[str, np.ndarray]:
        """
        Compute pseudo-gradient: difference from sync point.

        Δ = w_sync - w_local (note: negative of weight change)
        """
        delta = {}
        for name in self.weights:
            delta[name] = self.sync_weights[name] - self.weights[name]
        return delta

    def outer_update(self, averaged_delta: Dict[str, np.ndarray]):
        """
        Outer optimizer step (Nesterov momentum).

        Apply averaged pseudo-gradient with momentum.
        """
        beta = self.config.outer_momentum

        for name in self.weights:
            # Nesterov momentum
            self.outer_momentum[name] = (beta * self.outer_momentum[name] +
                                         averaged_delta[name])

            # Update sync point
            self.sync_weights[name] = (self.sync_weights[name] -
                                       self.config.outer_lr * self.outer_momentum[name])

            # Reset local weights to sync point
            self.weights[name] = self.sync_weights[name].copy()

        # Reset inner optimizer state
        self.inner_step = 0
        self.m = {k: np.zeros_like(v) for k, v in self.weights.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.weights.items()}


def diloco_training(config: DiLoCoConfig,
                    initial_weights: Dict[str, np.ndarray],
                    data_iterators: list,
                    gradient_fn,
                    outer_steps: int,
                    allreduce_fn) -> Dict[str, np.ndarray]:
    """
    Run DiLoCo training.

    DiLoCo enables training across multiple clusters with
    minimal inter-cluster communication.
    """
    workers = [DiLoCoWorker(i, config, initial_weights)
               for i in range(config.num_workers)]

    for outer_step in range(outer_steps):
        # Inner loop: H local steps
        for inner_step in range(config.inner_steps):
            for worker, data_iter in zip(workers, data_iterators):
                batch = next(data_iter)
                gradient = gradient_fn(worker.weights, batch)
                worker.inner_update(gradient)

        # Outer step: sync pseudo-gradients
        all_deltas = [w.compute_pseudo_gradient() for w in workers]

        # Average pseudo-gradients
        averaged_delta = {}
        for name in all_deltas[0]:
            stacked = np.stack([d[name] for d in all_deltas])
            averaged_delta[name] = np.mean(stacked, axis=0)

        # Apply outer update
        for worker in workers:
            worker.outer_update(averaged_delta)

        print(f"Outer step {outer_step + 1}/{outer_steps}")

    return workers[0].weights
```

**DiLoCo results**:

- 500× less communication than fully synchronous training
- Matches quality of synchronous training on 70B parameter models
- Enables training across geographic regions

## Choosing Synchronization Strategy

### Communication-Compute Trade-off

```python
from dataclasses import dataclass
from enum import Enum


class SyncStrategy(Enum):
    FULLY_SYNC = "fully_synchronous"  # AllReduce every step
    BOUNDED_ASYNC = "bounded_async"    # Async with max staleness
    LOCAL_SGD = "local_sgd"            # Periodic sync
    DILOCO = "diloco"                  # Different inner/outer optimizers
    HOGWILD = "hogwild"                # Lock-free for sparse


@dataclass
class WorkloadProfile:
    """Workload characteristics."""
    compute_time_ms: float          # Single step compute
    communication_time_ms: float    # AllReduce time
    compute_variance: float         # Variance in compute time
    gradient_sparsity: float        # Fraction of non-zero gradients
    data_heterogeneity: float       # KL divergence between worker distributions


class SyncStrategyAdvisor:
    """Recommend synchronization strategy based on workload."""

    def recommend(self, profile: WorkloadProfile) -> SyncStrategy:
        """
        Select synchronization strategy.

        Decision tree based on workload characteristics.
        """
        # Compute communication overhead ratio
        overhead = profile.communication_time_ms / profile.compute_time_ms

        # Very sparse gradients → Hogwild!
        if profile.gradient_sparsity < 0.01:
            return SyncStrategy.HOGWILD

        # Low overhead → synchronous is fine
        if overhead < 0.1:
            return SyncStrategy.FULLY_SYNC

        # High variance but low overhead → bounded async
        if profile.compute_variance > 0.5 and overhead < 0.5:
            return SyncStrategy.BOUNDED_ASYNC

        # High overhead but homogeneous data → Local SGD
        if overhead > 0.5 and profile.data_heterogeneity < 0.1:
            return SyncStrategy.LOCAL_SGD

        # High overhead with heterogeneous data → DiLoCo
        if overhead > 0.5:
            return SyncStrategy.DILOCO

        # Default
        return SyncStrategy.LOCAL_SGD

    def estimate_speedup(self, strategy: SyncStrategy,
                        profile: WorkloadProfile,
                        num_workers: int,
                        sync_interval: int = 100) -> float:
        """Estimate speedup over fully synchronous."""
        base_step = profile.compute_time_ms + profile.communication_time_ms

        if strategy == SyncStrategy.FULLY_SYNC:
            return 1.0

        elif strategy == SyncStrategy.BOUNDED_ASYNC:
            # No waiting for stragglers, slight convergence penalty
            speedup = 1 + profile.compute_variance
            convergence_penalty = 0.9  # ~10% slower convergence
            return speedup * convergence_penalty

        elif strategy == SyncStrategy.LOCAL_SGD:
            # Amortize communication over H steps
            local_step = profile.compute_time_ms
            sync_step = profile.compute_time_ms + profile.communication_time_ms
            avg_step = (local_step * (sync_interval - 1) + sync_step) / sync_interval
            speedup = base_step / avg_step

            # Slight convergence penalty
            convergence_penalty = 1 - 0.001 * sync_interval
            return speedup * max(convergence_penalty, 0.9)

        elif strategy == SyncStrategy.DILOCO:
            # Similar to Local SGD but with better convergence
            local_step = profile.compute_time_ms
            sync_step = profile.compute_time_ms + profile.communication_time_ms
            avg_step = (local_step * (sync_interval - 1) + sync_step) / sync_interval
            speedup = base_step / avg_step

            # Better convergence than plain Local SGD
            convergence_penalty = 1 - 0.0005 * sync_interval
            return speedup * max(convergence_penalty, 0.95)

        return 1.0


def optimal_sync_interval(compute_time: float, comm_time: float,
                          variance_growth: float) -> int:
    """
    Find optimal synchronization interval H.

    Balance:

    - Communication savings: higher H → less overhead
    - Variance penalty: higher H → more local drift

    Optimal: H* ≈ sqrt(comm_time / variance_growth)
    """
    # Simplified model
    H_optimal = int(np.sqrt(comm_time / (compute_time * variance_growth)))
    return max(1, min(H_optimal, 1000))  # Clamp to reasonable range
```

### Decision Matrix

| Characteristic | Strategy | When to Use |
|---------------|----------|-------------|
| Low comm overhead | Fully Sync | Communication < 10% of compute |
| High straggler variance | Bounded Async | Fast workers shouldn't wait |
| High comm overhead, homogeneous data | Local SGD | Periodic sync sufficient |
| High comm overhead, heterogeneous data | DiLoCo | Need variance reduction |
| Extremely sparse gradients | Hogwild! | Matrix factorization, sparse ML |
| Cross-datacenter | DiLoCo | Very high latency |

## Exercises

1. **Staleness analysis**: In async SGD with 16 workers and equal compute times, what's the expected staleness? How should learning rate be adjusted?

2. **Local SGD interval**: Given compute time = 100ms, communication time = 50ms, and variance growth rate = 0.001 per step, find the optimal sync interval $H$.

3. **Convergence comparison**: Implement Local SGD and fully synchronous SGD. Train a simple model on MNIST. Compare:

   - Wall-clock time to reach 95% accuracy
   - Number of gradient steps
   - Total communication volume

4. **DiLoCo implementation**: Implement DiLoCo with Adam as inner optimizer and Nesterov as outer optimizer. Compare to standard Local SGD on a language modeling task.

5. **Hogwild! sparsity threshold**: Theoretically, at what sparsity level does Hogwild! converge at the same rate as locked SGD? Verify empirically.

6. **Heterogeneous data**: Create a setting where workers have different data distributions. Compare Local SGD, FedProx, and SCAFFOLD. Which handles heterogeneity best?

## Key Takeaways

1. **Synchronization is expensive**: The straggler problem grows with worker count.

2. **Staleness hurts convergence**: Must reduce learning rate for async SGD.

3. **Local SGD works surprisingly well**: Can sync every 100+ steps with minimal quality loss.

4. **DiLoCo scales to LLMs**: 500× less communication while matching quality.

5. **Different optimizers for inner/outer**: DiLoCo's key insight—Adam locally, Nesterov globally.

6. **Variance reduction helps heterogeneity**: SCAFFOLD and FedProx handle non-IID data.

7. **Choose strategy based on profile**: No single approach wins everywhere.
