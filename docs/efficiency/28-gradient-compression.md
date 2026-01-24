---
title: "Gradient Compression"
subtitle: "Trading Fidelity for Bandwidth"
---

<div class="chapter-opener" markdown>
Communication costs dominate large-scale training. Gradient compression offers a seemingly magical solution: transmit less data while converging to the same result. The mathematics of why this works—and when it fails—reveals deep connections between optimization theory and information theory.
</div>

<div class="investigation-question" markdown>
**The Question**: You compress gradients to 1% of their original size, yet the model converges to the same quality. How is this possible? What information is truly necessary for convergence?
</div>

!!! abstract "Building On: Parts III–VI"
    We established in Part III that **AllReduce costs** scale with model size and dominate large-scale training. Part IV showed how **data parallelism** requires gradient synchronization every step. Part VI taught composition. Now we ask: can we reduce the communication volume itself? This part explores techniques that push efficiency beyond what basic parallelism offers.

## The Communication Bottleneck

In data-parallel training, each step requires:

$$\text{AllReduce}(g) = \frac{1}{P} \sum_{i=0}^{P-1} g_i$$

For a model with $N$ parameters in FP32:

- Communication volume: $2N \cdot 4$ bytes per step (ring AllReduce)
- For GPT-3 (175B parameters): 1.4 TB per step

At 400 Gbps inter-node bandwidth:
$$T_{\text{comm}} = \frac{1.4 \times 10^{12} \times 8}{400 \times 10^9} = 28 \text{ seconds}$$

This is unacceptable. Compression is essential.

## The Compression Framework

A gradient compression scheme consists of:

1. **Compressor**: $C: \mathbb{R}^d \to \mathcal{M}$ (compress to message)
2. **Decompressor**: $D: \mathcal{M} \to \mathbb{R}^d$ (reconstruct gradient)
3. **Compression ratio**: $\rho = \frac{|C(g)|}{d \cdot \text{sizeof(float)}}$

**Goal**: Minimize $\rho$ while preserving convergence.

### The Key Insight

Gradients have **redundancy**:

1. **Temporal redundancy**: Gradients change slowly between steps
2. **Spatial redundancy**: Many gradient elements are near-zero
3. **Magnitude redundancy**: Full precision isn't needed

Compression exploits these redundancies.

## Gradient Quantization

Replace high-precision gradients with low-precision representations.

### Naive Quantization

```python
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class QuantizedTensor:
    """Quantized representation of a tensor."""
    data: np.ndarray        # Quantized values (int8, int4, etc.)
    scale: float            # Scaling factor
    zero_point: float       # Zero point for asymmetric quantization
    original_shape: Tuple[int, ...]

    def dequantize(self) -> np.ndarray:
        """Reconstruct float tensor."""
        return self.scale * (self.data.astype(np.float32) - self.zero_point)

class NaiveQuantizer:
    """Simple min-max quantization."""

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.num_levels = 2 ** bits

    def quantize(self, tensor: np.ndarray) -> QuantizedTensor:
        """Quantize tensor to fixed-point representation."""
        # Compute range
        t_min, t_max = tensor.min(), tensor.max()

        # Handle degenerate case
        if t_max == t_min:
            return QuantizedTensor(
                data=np.zeros(tensor.shape, dtype=np.int8),
                scale=1.0,
                zero_point=0.0,
                original_shape=tensor.shape
            )

        # Compute scale and zero point
        scale = (t_max - t_min) / (self.num_levels - 1)
        zero_point = t_min / scale

        # Quantize
        quantized = np.round(tensor / scale - zero_point).astype(np.int8)

        return QuantizedTensor(
            data=quantized,
            scale=scale,
            zero_point=zero_point,
            original_shape=tensor.shape
        )

    def compression_ratio(self) -> float:
        """Bits saved compared to FP32."""
        return self.bits / 32
```

**Problem**: Range determined by outliers → poor precision for typical values.

### QSGD: Quantized SGD

Alistarh et al. (2017) introduced stochastic quantization:

$$Q_s(g_i) = ||g||_2 \cdot \text{sign}(g_i) \cdot \xi_i(|g_i|, s)$$

Where $\xi_i$ is a randomized rounding function:

$$\xi_i(v, s) = \begin{cases}
\lfloor sv/||g||_2 \rfloor / s & \text{with prob } 1 - (sv/||g||_2 - \lfloor sv/||g||_2 \rfloor) \\
\lceil sv/||g||_2 \rceil / s & \text{otherwise}
\end{cases}$$

**Key property**: Unbiased!
$$\mathbb{E}[Q_s(g)] = g$$

```python
class QSGDQuantizer:
    """QSGD: Quantized Stochastic Gradient Descent."""

    def __init__(self, num_levels: int = 256):
        self.num_levels = num_levels
        self.s = num_levels - 1  # Number of quantization intervals

    def quantize(self, gradient: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Stochastic quantization with unbiasedness guarantee.

        Returns:
            signs: Sign of each element
            norm: L2 norm of gradient
            levels: Quantization levels [0, s]
        """
        # Compute norm and signs
        norm = np.linalg.norm(gradient)
        if norm == 0:
            return np.zeros_like(gradient, dtype=np.int8), 0.0, np.zeros(gradient.shape)

        signs = np.sign(gradient).astype(np.int8)

        # Normalize absolute values
        abs_normalized = np.abs(gradient) / norm

        # Stochastic rounding
        scaled = self.s * abs_normalized
        lower = np.floor(scaled).astype(np.int32)
        prob = scaled - lower  # Probability of rounding up

        # Random rounding
        random_vals = np.random.random(gradient.shape)
        levels = np.where(random_vals < prob, lower + 1, lower).astype(np.int8)

        return signs, norm, levels

    def dequantize(self, signs: np.ndarray, norm: float,
                   levels: np.ndarray) -> np.ndarray:
        """Reconstruct gradient from quantized representation."""
        return norm * signs * (levels / self.s)

    @property
    def compression_ratio(self) -> float:
        """Communication cost relative to FP32."""
        # 1 bit for sign, log2(s+1) bits for level, plus norm (32 bits amortized)
        bits_per_element = 1 + np.log2(self.num_levels)
        return bits_per_element / 32
```

**Variance analysis**:
$$\mathbb{E}[||Q_s(g) - g||^2] \leq \min\left(\frac{d}{s^2}, \frac{\sqrt{d}}{s}\right) ||g||^2$$

Higher $s$ → more levels → lower variance → better convergence.

### TernGrad: Ternary Gradients

Wen et al. (2017) pushed quantization to the extreme: just three values {-1, 0, +1}.

```python
class TernGradQuantizer:
    """TernGrad: Ternary gradient quantization."""

    def quantize(self, gradient: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Quantize to ternary values {-1, 0, +1}.

        Uses stochastic rounding with threshold based on max absolute value.
        """
        # Scaling factor (per-layer)
        scale = np.max(np.abs(gradient))

        if scale == 0:
            return np.zeros_like(gradient, dtype=np.int8), 0.0

        # Probability of non-zero
        prob = np.abs(gradient) / scale

        # Stochastic ternary rounding
        random_vals = np.random.random(gradient.shape)
        ternary = np.where(
            random_vals < prob,
            np.sign(gradient),
            0
        ).astype(np.int8)

        return ternary, scale

    def dequantize(self, ternary: np.ndarray, scale: float) -> np.ndarray:
        """Reconstruct from ternary values."""
        return scale * ternary.astype(np.float32)

    @property
    def compression_ratio(self) -> float:
        """2 bits per element (can encode 3 values)."""
        return 2 / 32  # 16x compression

class PackedTernaryGradient:
    """Pack ternary gradients efficiently."""

    @staticmethod
    def pack(ternary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pack ternary values into bit arrays.

        Each element needs 2 bits: 00=0, 01=+1, 10=-1
        Pack 4 elements per byte.
        """
        # Convert to 0, 1, 2 representation
        encoded = np.where(ternary > 0, 1, np.where(ternary < 0, 2, 0))

        # Pad to multiple of 4
        padded_len = (len(encoded) + 3) // 4 * 4
        padded = np.zeros(padded_len, dtype=np.uint8)
        padded[:len(encoded)] = encoded.flatten()

        # Pack 4 values per byte
        packed = (
            padded[0::4] |
            (padded[1::4] << 2) |
            (padded[2::4] << 4) |
            (padded[3::4] << 6)
        )

        return packed, np.array(ternary.shape)

    @staticmethod
    def unpack(packed: np.ndarray, shape: np.ndarray) -> np.ndarray:
        """Unpack to ternary values."""
        # Extract 4 values per byte
        unpacked = np.zeros(len(packed) * 4, dtype=np.int8)
        unpacked[0::4] = packed & 0x03
        unpacked[1::4] = (packed >> 2) & 0x03
        unpacked[2::4] = (packed >> 4) & 0x03
        unpacked[3::4] = (packed >> 6) & 0x03

        # Convert back to -1, 0, +1
        ternary = np.where(unpacked == 1, 1, np.where(unpacked == 2, -1, 0))

        # Reshape
        total_elements = np.prod(shape)
        return ternary[:total_elements].reshape(shape)
```

**Trade-off**: 16× compression but higher variance.

## Gradient Sparsification

Instead of quantizing all elements, transmit only the most important ones.

### Top-K Sparsification

Select the K largest-magnitude gradients:

```python
class TopKSparsifier:
    """Top-K gradient sparsification."""

    def __init__(self, k_ratio: float = 0.01):
        """
        Args:
            k_ratio: Fraction of gradients to keep (1% = 0.01)
        """
        self.k_ratio = k_ratio

    def sparsify(self, gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select top-K elements by magnitude.

        Returns:
            indices: Positions of selected elements
            values: Selected gradient values
        """
        flat = gradient.flatten()
        k = max(1, int(len(flat) * self.k_ratio))

        # Find top-K by magnitude
        abs_flat = np.abs(flat)
        top_k_indices = np.argpartition(abs_flat, -k)[-k:]

        # Sort by index for better compression
        top_k_indices = np.sort(top_k_indices)
        top_k_values = flat[top_k_indices]

        return top_k_indices.astype(np.uint32), top_k_values.astype(np.float32)

    def densify(self, indices: np.ndarray, values: np.ndarray,
                shape: Tuple[int, ...]) -> np.ndarray:
        """Reconstruct dense gradient."""
        flat = np.zeros(np.prod(shape), dtype=np.float32)
        flat[indices] = values
        return flat.reshape(shape)

    @property
    def compression_ratio(self) -> float:
        """Approximate compression ratio."""
        # Index (4 bytes) + value (4 bytes) per selected element
        return self.k_ratio * (32 + 32) / 32

class RandomKSparsifier:
    """Random-K sparsification with unbiased estimation."""

    def __init__(self, k_ratio: float = 0.01):
        self.k_ratio = k_ratio

    def sparsify(self, gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random selection with scaling for unbiasedness.
        """
        flat = gradient.flatten()
        k = max(1, int(len(flat) * self.k_ratio))

        # Random selection
        indices = np.random.choice(len(flat), k, replace=False)
        indices = np.sort(indices)

        # Scale values for unbiased estimation
        scale = len(flat) / k
        values = flat[indices] * scale

        return indices.astype(np.uint32), values.astype(np.float32)
```

**Problem with Top-K**: Biased! Small gradients are never transmitted.

### Error Feedback (Memory Compression)

The key insight from Seide et al. (2014) and Stich et al. (2018):

**Accumulate compression errors and add them to the next gradient.**

```python
class ErrorFeedbackCompressor:
    """
    Gradient compression with error feedback.

    Maintains error accumulator to ensure eventual transmission
    of all gradient information.
    """

    def __init__(self, base_compressor, shape: Tuple[int, ...]):
        """
        Args:
            base_compressor: Any compression method (quantizer or sparsifier)
            shape: Shape of gradient tensor
        """
        self.compressor = base_compressor
        self.error_accumulator = np.zeros(shape, dtype=np.float32)

    def compress(self, gradient: np.ndarray) -> Tuple:
        """
        Compress gradient with error feedback.

        Algorithm:
            1. Add accumulated error to gradient
            2. Compress the sum
            3. Store new error (original - transmitted)
        """
        # Add accumulated error
        corrected_gradient = gradient + self.error_accumulator

        # Compress
        compressed = self.compressor.sparsify(corrected_gradient)

        # Compute what was actually transmitted
        transmitted = self.compressor.densify(*compressed, gradient.shape)

        # Update error accumulator
        self.error_accumulator = corrected_gradient - transmitted

        return compressed

    def reset(self):
        """Reset error accumulator (e.g., at optimization reset)."""
        self.error_accumulator.fill(0)
```

**Theorem (Error Feedback Convergence)**:
With error feedback, Top-K sparsification converges at the same rate as full-precision SGD, up to a constant factor.

**Proof sketch**: The error feedback ensures that all gradient information is eventually transmitted. Let $e_t$ be the error at step $t$. Then:
$$\sum_{t=1}^T ||e_t||^2 \leq ||g_1||^2 + ||g_2||^2 + ... + ||g_T||^2$$

The accumulated error is bounded by the total gradient magnitude.

### Deep Gradient Compression (DGC)

Lin et al. (2018) combined multiple techniques:

```python
from collections import deque

class DeepGradientCompressor:
    """
    Deep Gradient Compression: 99.9% sparsity with minimal accuracy loss.

    Combines:
    1. Momentum correction
    2. Local gradient clipping
    3. Momentum factor masking
    4. Warm-up training
    """

    def __init__(self, shape: Tuple[int, ...], sparsity: float = 0.999,
                 momentum: float = 0.9, warmup_epochs: int = 4):
        self.sparsity = sparsity
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # Accumulators
        self.velocity = np.zeros(shape, dtype=np.float32)
        self.error = np.zeros(shape, dtype=np.float32)

        # Warmup schedule
        self.warmup_sparsities = self._compute_warmup_schedule()

    def _compute_warmup_schedule(self) -> list:
        """Gradual sparsity increase during warmup."""
        if self.warmup_epochs == 0:
            return []

        # Exponential warmup: 75% -> 93.75% -> 98.4% -> 99.6% -> 99.9%
        schedule = []
        current = 0.75
        for _ in range(self.warmup_epochs):
            schedule.append(current)
            current = 1 - (1 - current) * (1 - self.sparsity) / (1 - current)
        return schedule

    def compress(self, gradient: np.ndarray, epoch: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compress gradient using DGC algorithm.

        Args:
            gradient: Raw gradient
            epoch: Current training epoch

        Returns:
            indices: Selected gradient positions
            values: Compressed gradient values
        """
        # Determine sparsity for this epoch
        if epoch < len(self.warmup_sparsities):
            current_sparsity = self.warmup_sparsities[epoch]
        else:
            current_sparsity = self.sparsity

        # Momentum correction: u_t = m * u_{t-1} + g_t
        self.velocity = self.momentum * self.velocity + gradient

        # Add error feedback
        corrected = self.velocity + self.error

        # Local gradient clipping (per-layer)
        std = np.std(corrected)
        if std > 0:
            corrected = np.clip(corrected, -2.5 * std, 2.5 * std)

        # Top-K selection
        k = max(1, int(corrected.size * (1 - current_sparsity)))
        flat = corrected.flatten()
        abs_flat = np.abs(flat)

        # Find threshold
        if k >= len(flat):
            threshold = 0
        else:
            threshold = np.partition(abs_flat, -k)[-k]

        # Create mask
        mask = abs_flat >= threshold
        indices = np.where(mask)[0].astype(np.uint32)
        values = flat[indices].astype(np.float32)

        # Update error
        transmitted = np.zeros_like(flat)
        transmitted[indices] = values
        self.error = corrected.flatten() - transmitted
        self.error = self.error.reshape(gradient.shape)

        # Momentum factor masking: reset velocity for transmitted gradients
        velocity_flat = self.velocity.flatten()
        velocity_flat[indices] = 0
        self.velocity = velocity_flat.reshape(gradient.shape)

        return indices, values
```

**DGC achieved 99.9% sparsity** (100× compression) with negligible accuracy loss on ImageNet.

## Error Analysis

### Compression Error Bound

For a compressor $C$ with reconstruction $D$, define the compression error:
$$\epsilon = g - D(C(g))$$

A compressor is $\delta$-contractive if:
$$\mathbb{E}[||\epsilon||^2] \leq \delta^2 ||g||^2$$

**Examples**:

- QSGD with s levels: $\delta^2 = \min(d/s^2, \sqrt{d}/s)$
- Top-K: $\delta^2 = 1 - k/d$ (without error feedback)
- Top-K with error feedback: $\delta^2 \to 0$ over time

### Convergence with Compression

**Theorem**: For $\delta$-contractive compression with error feedback, SGD converges at rate:
$$\mathbb{E}[f(w_T) - f(w^*)] = O\left(\frac{1}{\sqrt{T}}\right)$$

The same rate as uncompressed SGD!

**Proof (sketch)**:
1. Decompose update: $w_{t+1} = w_t - \eta (g_t + \epsilon_t - \epsilon_{t-1})$
2. The error terms telescope over time
3. Bounded variance condition ensures convergence

```python
class ConvergenceAnalyzer:
    """Analyze convergence with gradient compression."""

    def __init__(self, contraction_delta: float, learning_rate: float):
        self.delta = contraction_delta
        self.lr = learning_rate

    def expected_error_after_T_steps(self, T: int,
                                      gradient_norm: float) -> float:
        """
        Upper bound on expected optimization error.

        With δ-contractive compression and error feedback:
        E[f(w_T) - f*] ≤ O(σ / √T) + O(δ² η G²)

        Where:
            σ: stochastic gradient variance
            G: gradient norm bound
            η: learning rate
        """
        stochastic_term = gradient_norm / np.sqrt(T)
        compression_term = self.delta**2 * self.lr * gradient_norm**2

        return stochastic_term + compression_term

    def optimal_compression_level(self, bandwidth_gbps: float,
                                   compute_time_ms: float,
                                   model_size_gb: float) -> float:
        """
        Find optimal compression ratio.

        Trade-off: Higher compression → faster communication
                   but higher variance → slower convergence
        """
        # Time without compression
        comm_time = (model_size_gb * 8) / bandwidth_gbps * 1000  # ms

        # Find compression ratio where comm_time = compute_time
        # This balances compute and communication
        target_ratio = compute_time_ms / comm_time

        return min(1.0, target_ratio)
```

## Practical Systems

### PowerSGD

Vogels et al. (2019) introduced low-rank gradient approximation:

```python
class PowerSGD:
    """
    PowerSGD: Low-rank gradient compression.

    Approximates gradients as rank-r matrices: G ≈ P @ Q^T
    """

    def __init__(self, shape: Tuple[int, int], rank: int = 4,
                 num_power_iterations: int = 1):
        """
        Args:
            shape: Gradient matrix shape (m, n)
            rank: Approximation rank r
            num_power_iterations: Power iteration steps for better approximation
        """
        self.m, self.n = shape
        self.rank = rank
        self.num_iters = num_power_iterations

        # Initialize Q with orthogonal matrix
        self.Q = np.linalg.qr(np.random.randn(self.n, rank))[0]

        # Error memory
        self.error = np.zeros(shape, dtype=np.float32)

    def compress(self, gradient: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compress gradient to low-rank representation.

        Returns:
            P: Left factor (m × r)
            Q: Right factor (n × r)
        """
        # Add error feedback
        G = gradient + self.error

        # Power iteration: P = G @ Q
        P = G @ self.Q

        # Orthogonalize P
        P, _ = np.linalg.qr(P)

        # Compute Q = G^T @ P
        Q = G.T @ P

        # Power iteration refinement
        for _ in range(self.num_iters - 1):
            P = G @ Q
            P, _ = np.linalg.qr(P)
            Q = G.T @ P

        # Reconstruct and compute error
        reconstructed = P @ Q.T
        self.error = G - reconstructed

        # Update Q for next iteration (warm start)
        self.Q, _ = np.linalg.qr(Q)

        return P.astype(np.float32), Q.astype(np.float32)

    def decompress(self, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Reconstruct gradient from low-rank factors."""
        return P @ Q.T

    @property
    def compression_ratio(self) -> float:
        """Communication cost reduction."""
        original = self.m * self.n
        compressed = self.rank * (self.m + self.n)
        return compressed / original

class PowerSGDOptimizer:
    """Optimizer wrapper with PowerSGD compression."""

    def __init__(self, model_parameters: list, lr: float,
                 rank: int = 4, min_compression_dim: int = 256):
        """
        Args:
            model_parameters: List of parameter tensors
            lr: Learning rate
            rank: PowerSGD rank
            min_compression_dim: Only compress tensors larger than this
        """
        self.lr = lr
        self.rank = rank
        self.min_dim = min_compression_dim

        # Create compressor for each parameter
        self.compressors = {}
        for i, param in enumerate(model_parameters):
            if param.size >= min_compression_dim:
                # Reshape to 2D for matrix compression
                shape_2d = self._to_2d_shape(param.shape)
                self.compressors[i] = PowerSGD(shape_2d, rank)

    def _to_2d_shape(self, shape: Tuple) -> Tuple[int, int]:
        """Reshape tensor to 2D for low-rank compression."""
        if len(shape) == 1:
            return (1, shape[0])
        elif len(shape) == 2:
            return shape
        else:
            # Flatten all but last dimension
            return (np.prod(shape[:-1]), shape[-1])

    def step(self, gradients: list) -> list:
        """
        Apply compressed gradients.

        Returns: List of compressed gradients for communication
        """
        compressed = []

        for i, grad in enumerate(gradients):
            if i in self.compressors:
                # Reshape, compress, decompress, reshape back
                grad_2d = grad.reshape(self._to_2d_shape(grad.shape))
                P, Q = self.compressors[i].compress(grad_2d)
                compressed.append((P, Q))
            else:
                # Small tensor: no compression
                compressed.append(grad)

        return compressed
```

**Compression ratio**: For $m \times n$ matrix with rank $r$:
$$\rho = \frac{r(m + n)}{mn}$$

For $m = n = 4096$ and $r = 4$: $\rho = 0.002$ (500× compression!).

### 1-Bit Adam

Leverages Adam's variance estimate for 1-bit compression:

```python
class OneBitAdam:
    """
    1-Bit Adam: Extreme compression using momentum.

    Key insight: Adam's momentum provides the magnitude information,
    so we only need to transmit the sign of gradient corrections.
    """

    def __init__(self, parameters: list, lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, warmup_steps: int = 500):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # Adam state
        self.m = [np.zeros_like(p) for p in parameters]  # First moment
        self.v = [np.zeros_like(p) for p in parameters]  # Second moment

        # Error compensation
        self.errors = [np.zeros_like(p) for p in parameters]

    def compress_gradient(self, gradient: np.ndarray,
                         idx: int) -> Tuple[np.ndarray, float]:
        """
        Compress gradient to 1 bit per element.

        During warmup: Use full precision
        After warmup: Transmit only signs
        """
        if self.step_count < self.warmup_steps:
            # Warmup: full precision communication
            return gradient, 1.0

        # Add error compensation
        compensated = gradient + self.errors[idx]

        # Extract signs
        signs = np.sign(compensated).astype(np.int8)

        # Scale by local mean absolute value
        scale = np.mean(np.abs(compensated)) + self.eps

        # Compute error
        reconstructed = scale * signs.astype(np.float32)
        self.errors[idx] = compensated - reconstructed

        return signs, scale

    def decompress_gradient(self, signs: np.ndarray,
                            scale: float) -> np.ndarray:
        """Reconstruct gradient from signs and scale."""
        return scale * signs.astype(np.float32)

    def step(self, gradients: list):
        """
        Adam update with 1-bit gradient communication.
        """
        self.step_count += 1

        # Bias correction
        bc1 = 1 - self.beta1 ** self.step_count
        bc2 = 1 - self.beta2 ** self.step_count

        updates = []
        for i, g in enumerate(gradients):
            # Update moments (using decompressed gradients after AllReduce)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # Compute update
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2

            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            updates.append(update)

        return updates
```

**Key insight**: After warmup, Adam's momentum captures gradient magnitude, so only the sign (direction) needs to be transmitted.

## AllReduce with Compression

Compressed gradients require modified AllReduce:

```python
from typing import Callable
import struct

class CompressedAllReduce:
    """
    AllReduce for compressed gradients.

    Standard AllReduce doesn't work with compressed representations.
    Need to either:
    1. AllGather + local aggregation
    2. Custom reduction for compressed format
    """

    def __init__(self, world_size: int, rank: int,
                 compressor: Callable, decompressor: Callable):
        self.world_size = world_size
        self.rank = rank
        self.compressor = compressor
        self.decompressor = decompressor

    def allreduce_sparsified(self, indices: np.ndarray,
                             values: np.ndarray) -> np.ndarray:
        """
        AllReduce for sparse gradients.

        Algorithm:
        1. AllGather indices and values from all ranks
        2. Merge sparse representations
        3. Average overlapping indices
        """
        # Simulate AllGather (in practice, use NCCL/Gloo)
        all_indices = [indices]  # Would gather from all ranks
        all_values = [values]

        # In actual distributed setting:
        # all_indices = allgather(indices)
        # all_values = allgather(values)

        # Merge: accumulate values at each index
        from collections import defaultdict
        accumulated = defaultdict(lambda: (0.0, 0))  # (sum, count)

        for idx_array, val_array in zip(all_indices, all_values):
            for idx, val in zip(idx_array, val_array):
                s, c = accumulated[idx]
                accumulated[idx] = (s + val, c + 1)

        # Average
        merged_indices = np.array(list(accumulated.keys()), dtype=np.uint32)
        merged_values = np.array(
            [s / c for s, c in accumulated.values()],
            dtype=np.float32
        )

        return merged_indices, merged_values

    def allreduce_quantized(self, quantized: QuantizedTensor) -> np.ndarray:
        """
        AllReduce for quantized gradients.

        Cannot directly average quantized values!
        Must dequantize, reduce, then optionally requantize.
        """
        # Dequantize locally
        local_full = quantized.dequantize()

        # Standard AllReduce (simulated)
        # In practice: dist.all_reduce(local_full)
        global_sum = local_full  # Would be sum from all ranks

        # Average
        return global_sum / self.world_size

    def allreduce_low_rank(self, P: np.ndarray,
                          Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        AllReduce for low-rank gradients (PowerSGD style).

        Key insight: AllReduce on P@Q^T factors directly.

        sum_i(P_i @ Q_i^T) = [P_0 | P_1 | ... | P_n] @ [Q_0 | Q_1 | ...]^T
        """
        # AllGather P and Q matrices
        all_P = [P]  # Would gather from all ranks
        all_Q = [Q]

        # Concatenate
        P_concat = np.hstack(all_P)  # m × (r * world_size)
        Q_concat = np.hstack(all_Q)  # n × (r * world_size)

        # Reduce rank back to r (optional, for memory)
        # Could use SVD or just sum and re-orthogonalize

        return P_concat, Q_concat
```

### Efficient Sparse AllReduce

For highly sparse gradients, specialized algorithms help:

```python
class SparseAllReduce:
    """
    Efficient AllReduce for sparse tensors.

    Uses tree-based aggregation with sparse merge.
    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

    def tree_reduce(self, local_sparse: Tuple[np.ndarray, np.ndarray],
                   original_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Binary tree reduction for sparse gradients.

        Complexity: O(k * log(P)) where k = sparsity
        vs O(d * log(P)) for dense reduce
        """
        indices, values = local_sparse

        # Number of tree levels
        levels = int(np.ceil(np.log2(self.world_size)))

        current_indices = indices
        current_values = values

        for level in range(levels):
            step = 2 ** level

            if self.rank % (2 * step) == 0:
                # Receive from partner
                partner = self.rank + step
                if partner < self.world_size:
                    # Simulate receive
                    partner_indices = np.array([], dtype=np.uint32)
                    partner_values = np.array([], dtype=np.float32)

                    # Merge sparse representations
                    current_indices, current_values = self._merge_sparse(
                        (current_indices, current_values),
                        (partner_indices, partner_values)
                    )
            elif self.rank % (2 * step) == step:
                # Send to partner
                partner = self.rank - step
                # Simulate send
                pass

        # Broadcast result
        return current_indices, current_values

    def _merge_sparse(self, sparse1: Tuple[np.ndarray, np.ndarray],
                      sparse2: Tuple[np.ndarray, np.ndarray]
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge two sparse representations, summing overlapping indices."""
        idx1, val1 = sparse1
        idx2, val2 = sparse2

        # Create combined representation
        combined = {}
        for i, v in zip(idx1, val1):
            combined[i] = combined.get(i, 0) + v
        for i, v in zip(idx2, val2):
            combined[i] = combined.get(i, 0) + v

        # Convert back to arrays
        if not combined:
            return np.array([], dtype=np.uint32), np.array([], dtype=np.float32)

        merged_indices = np.array(sorted(combined.keys()), dtype=np.uint32)
        merged_values = np.array([combined[i] for i in merged_indices],
                                dtype=np.float32)

        return merged_indices, merged_values
```

## When to Use Compression

### Decision Framework

```python
from dataclasses import dataclass
from enum import Enum

class CompressionMethod(Enum):
    NONE = "none"
    QUANTIZATION = "quantization"
    SPARSIFICATION = "sparsification"
    LOW_RANK = "low_rank"
    HYBRID = "hybrid"

@dataclass
class ClusterSpec:
    """Cluster hardware specification."""
    intra_node_bandwidth_gbps: float  # NVLink/NVSwitch bandwidth
    inter_node_bandwidth_gbps: float  # Network bandwidth
    gpus_per_node: int
    num_nodes: int

class CompressionAdvisor:
    """Recommend compression strategy based on hardware and workload."""

    def __init__(self, cluster: ClusterSpec):
        self.cluster = cluster

    def recommend(self, model_size_bytes: int,
                  compute_time_ms: float,
                  accuracy_tolerance: str = "standard") -> CompressionMethod:
        """
        Recommend compression method.

        Args:
            model_size_bytes: Size of gradients to communicate
            compute_time_ms: Time for forward + backward pass
            accuracy_tolerance: "strict", "standard", or "relaxed"

        Returns:
            Recommended compression method
        """
        # Compute communication time without compression
        # For ring AllReduce: 2 * (P-1)/P * size
        world_size = self.cluster.gpus_per_node * self.cluster.num_nodes

        # Pessimistic estimate using inter-node bandwidth
        bandwidth = self.cluster.inter_node_bandwidth_gbps * 1e9 / 8  # bytes/sec
        comm_time_ms = (2 * model_size_bytes / bandwidth) * 1000

        # Compute-to-communication ratio
        ratio = compute_time_ms / comm_time_ms

        if ratio > 2.0:
            # Compute-bound: no compression needed
            return CompressionMethod.NONE

        if ratio > 0.5:
            # Moderate: light compression
            if accuracy_tolerance == "strict":
                return CompressionMethod.QUANTIZATION  # 8-bit
            else:
                return CompressionMethod.SPARSIFICATION  # Top-K with EF

        if ratio > 0.1:
            # Communication-bound: aggressive compression
            if accuracy_tolerance == "relaxed":
                return CompressionMethod.LOW_RANK
            else:
                return CompressionMethod.SPARSIFICATION

        # Severely communication-bound
        return CompressionMethod.HYBRID

    def estimate_speedup(self, method: CompressionMethod,
                         compression_ratio: float,
                         model_size_bytes: int,
                         compute_time_ms: float) -> float:
        """Estimate training speedup from compression."""
        world_size = self.cluster.gpus_per_node * self.cluster.num_nodes
        bandwidth = self.cluster.inter_node_bandwidth_gbps * 1e9 / 8

        # Original step time
        original_comm = (2 * model_size_bytes / bandwidth) * 1000
        original_step = compute_time_ms + original_comm

        # Compressed step time
        compressed_comm = original_comm * compression_ratio
        # Add overhead for compression/decompression
        overhead = 0.05 * compute_time_ms  # ~5% compute overhead
        compressed_step = compute_time_ms + compressed_comm + overhead

        return original_step / compressed_step
```

### Compression in Practice

| Scenario | Recommended Method | Compression Ratio |
|----------|-------------------|-------------------|
| High bandwidth (NVLink) | None | 1.0 |
| 100 Gbps Ethernet | 8-bit quantization | 0.25 |
| 25 Gbps Ethernet | TopK + EF | 0.01-0.1 |
| Cross-datacenter | PowerSGD | 0.001-0.01 |
| Federated learning | DGC | 0.001 |

## Exercises

1. **QSGD analysis**: Prove that QSGD is unbiased: $\mathbb{E}[Q_s(g)] = g$. Then compute $\text{Var}(Q_s(g))$ as a function of $s$ and $||g||$.

??? success "Solution"
    **QSGD Definition:**

    For a scalar $g_i$ with quantization levels $s$:

    $$Q_s(g_i) = ||g|| \cdot \text{sign}(g_i) \cdot \xi_i$$

    where $\xi_i$ is a random variable that takes value:
    - $\frac{\ell + 1}{s}$ with probability $\frac{|g_i|}{||g||} \cdot s - \ell$
    - $\frac{\ell}{s}$ with probability $1 - \left(\frac{|g_i|}{||g||} \cdot s - \ell\right)$

    and $\ell = \lfloor \frac{|g_i|}{||g||} \cdot s \rfloor$.

    **Proof of unbiasedness:**

    $$\mathbb{E}[\xi_i] = \frac{\ell + 1}{s} \cdot p + \frac{\ell}{s} \cdot (1 - p)$$

    where $p = \frac{|g_i|}{||g||} \cdot s - \ell$

    $$= \frac{\ell + 1}{s} \cdot p + \frac{\ell}{s} - \frac{\ell}{s} \cdot p$$

    $$= \frac{\ell}{s} + \frac{p}{s}$$

    $$= \frac{\ell + p}{s} = \frac{\ell + \frac{|g_i|}{||g||} \cdot s - \ell}{s}$$

    $$= \frac{|g_i|}{||g||}$$

    Therefore:

    $$\mathbb{E}[Q_s(g_i)] = ||g|| \cdot \text{sign}(g_i) \cdot \frac{|g_i|}{||g||} = g_i$$

    $$\boxed{\mathbb{E}[Q_s(g)] = g}$$

    **Variance calculation:**

    $$\text{Var}(\xi_i) = \mathbb{E}[\xi_i^2] - \mathbb{E}[\xi_i]^2$$

    $$\mathbb{E}[\xi_i^2] = \left(\frac{\ell + 1}{s}\right)^2 p + \left(\frac{\ell}{s}\right)^2 (1-p)$$

    After algebra:

    $$\text{Var}(\xi_i) = \frac{p(1-p)}{s^2}$$

    Since $p \leq 1$ and $p(1-p) \leq 1/4$:

    $$\text{Var}(Q_s(g_i)) = ||g||^2 \cdot \text{Var}(\xi_i) \leq \frac{||g||^2}{4s^2}$$

    For the full vector (independent components):

    $$\boxed{\text{Var}(Q_s(g)) \leq \frac{d \cdot ||g||^2}{4s^2}}$$

    As $s \to \infty$, variance $\to 0$ (full precision).

2. **Error feedback**: Implement error feedback for Top-K sparsification. Verify that the accumulated errors remain bounded over 1000 training steps.

??? success "Solution"
    **Error feedback implementation:**

    ```python
    import torch

    class TopKWithErrorFeedback:
        def __init__(self, k_ratio=0.01):
            self.k_ratio = k_ratio
            self.error = None

        def compress(self, gradient):
            # Add accumulated error to current gradient
            if self.error is None:
                self.error = torch.zeros_like(gradient)

            g_accumulated = gradient + self.error

            # Top-K selection
            k = max(1, int(gradient.numel() * self.k_ratio))
            values, indices = torch.topk(
                g_accumulated.abs().flatten(), k
            )
            threshold = values[-1]

            # Create sparse gradient
            mask = g_accumulated.abs() >= threshold
            sparse_g = g_accumulated * mask

            # Update error: what we couldn't send this round
            self.error = g_accumulated - sparse_g

            return sparse_g, mask

        def get_error_norm(self):
            return self.error.norm().item() if self.error is not None else 0

    # Verification experiment
    def verify_bounded_errors():
        compressor = TopKWithErrorFeedback(k_ratio=0.01)
        error_norms = []

        for step in range(1000):
            # Simulate gradient with some structure
            gradient = torch.randn(10000) * (1.0 / (step + 1) ** 0.5)

            _, _ = compressor.compress(gradient)
            error_norms.append(compressor.get_error_norm())

        # Check boundedness
        max_error = max(error_norms)
        avg_error = sum(error_norms) / len(error_norms)

        return max_error, avg_error
    ```

    **Expected results:**

    | Metric | Value |
    |--------|-------|
    | Max error norm | ~0.5-2.0 |
    | Average error norm | ~0.2-0.5 |
    | Trend | Stable, not growing |

    **Why errors stay bounded:**

    Error feedback ensures:

    $$e_{t+1} = e_t + g_t - \text{TopK}(e_t + g_t)$$

    The key insight: TopK always removes the largest values, so:

    $$||e_{t+1}||_\infty \leq ||e_t + g_t||_\infty \cdot (1 - k/d)$$

    With typical gradient decay during training, errors remain bounded.

    **Verification plot observation:**

    $$\boxed{\text{Error norms oscillate but do not grow over 1000 steps}}$$

3. **Compression selection**: You have 64 nodes connected by 100 Gbps Ethernet. Each node has 8 GPUs connected by NVLink. Model size is 1B parameters. Forward-backward takes 100ms. What compression should you use for data parallelism across nodes?

??? success "Solution"
    **Communication analysis:**

    | Parameter | Value |
    |-----------|-------|
    | Nodes | 64 |
    | GPUs per node | 8 |
    | Total GPUs | 512 |
    | Inter-node bandwidth | 100 Gbps = 12.5 GB/s |
    | Intra-node bandwidth | NVLink ~600 GB/s |
    | Model parameters | 1B |
    | Gradient size (FP16) | 2 GB |
    | Forward-backward time | 100 ms |

    **AllReduce time without compression:**

    For hierarchical AllReduce (reduce within node, AllReduce across nodes, broadcast within node):

    Inter-node AllReduce dominates (64 nodes, ring algorithm):

    $$T_{inter} = \frac{2 \times 2 \text{ GB}}{12.5 \text{ GB/s}} \times \frac{63}{64} \approx 315 \text{ ms}$$

    **Communication-to-compute ratio:**
    $$\frac{T_{comm}}{T_{comp}} = \frac{315}{100} = 3.15$$

    Communication takes 3× longer than compute—this is unacceptable!

    **Required compression ratio:**

    To make communication ≤ compute:

    $$\text{Compression ratio} \geq \frac{315}{100} \approx 3.2×$$

    To overlap fully with compute:

    $$\text{Compression ratio} \geq \frac{315}{100} \approx 3.2×$$

    For safety margin, target **10× compression**.

    **Compression selection:**

    | Method | Compression | Overhead | Suitable? |
    |--------|-------------|----------|-----------|
    | FP16→FP8 | 2× | Very low | Insufficient |
    | 4-bit quantization | 4× | Low | Borderline |
    | 1-bit SGD | 16× | Low | ✓ Good |
    | Top-10% sparsification | 10× | Medium | ✓ Good |
    | PowerSGD rank-16 | ~50× | Medium | ✓ Excellent |

    **Recommendation:**

    Given the 3× slowdown from communication:

    1. **Primary choice: PowerSGD with rank 16-32**
       - Compression: 32-64×
       - Error feedback handles bias
       - Minimal accuracy impact for 1B model

    2. **Alternative: 1-bit quantization with error feedback**
       - 16× compression
       - Very low computational overhead
       - Predictable latency

    **With 10× compression:**
    $$T_{comm}^{compressed} = \frac{315}{10} = 31.5 \text{ ms}$$

    **Final efficiency:**
    $$\text{Throughput improvement} = \frac{100 + 315}{100 + 31.5} = \frac{415}{131.5} = \boxed{3.16× \text{ speedup}}$$

4. **PowerSGD rank selection**: For a weight matrix of size 4096 × 4096, what rank gives 100× compression? What's the approximation error (in Frobenius norm) for a typical gradient matrix?

??? success "Solution"
    **PowerSGD compression ratio:**

    For a matrix $M \in \mathbb{R}^{m \times n}$, PowerSGD approximates:

    $$M \approx P Q^T$$
    where $P \in \mathbb{R}^{m \times r}$ and $Q \in \mathbb{R}^{n \times r}$.

    **Memory comparison:**

    | Representation | Size |
    |----------------|------|
    | Original matrix | $m \times n$ |
    | Low-rank factors | $r(m + n)$ |
    | Compression ratio | $\frac{mn}{r(m+n)}$ |

    **For 4096 × 4096 matrix:**

    $$\text{Compression ratio} = \frac{4096 \times 4096}{r(4096 + 4096)} = \frac{16,777,216}{8192r}$$

    For 100× compression:

    $$100 = \frac{16,777,216}{8192r}$$
    $$r = \frac{16,777,216}{8192 \times 100} = \frac{16,777,216}{819,200} \approx 20.48$$

    $$\boxed{r = 20 \text{ (or } r = 21 \text{ for slightly less compression)}}$$

    **Verification:**
    - $r = 20$: Compression = $\frac{16,777,216}{8192 \times 20} = 102.4×$
    - $r = 21$: Compression = $\frac{16,777,216}{8192 \times 21} = 97.5×$

    **Approximation error analysis:**

    For a matrix $G$ with singular values $\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_n$:

    Best rank-$r$ approximation error (Eckart-Young theorem):

    $$||G - G_r||_F = \sqrt{\sum_{i=r+1}^{n} \sigma_i^2}$$

    **Typical gradient matrix spectrum:**

    Empirically, gradient matrices exhibit rapid singular value decay:

    $$\sigma_i \approx \sigma_1 \cdot i^{-\alpha}$$

    where $\alpha \approx 0.5$ to $1.0$ for typical training gradients.

    **Numerical example** (assuming power-law decay with $\alpha = 0.7$):

    ```python
    import numpy as np

    # Simulate typical gradient singular value distribution
    n = 4096
    r = 20
    sigma_1 = 1.0
    alpha = 0.7

    singular_values = sigma_1 * np.arange(1, n+1) ** (-alpha)

    # Total Frobenius norm
    total_norm = np.sqrt(np.sum(singular_values ** 2))

    # Error (tail beyond rank r)
    error_norm = np.sqrt(np.sum(singular_values[r:] ** 2))

    # Relative error
    relative_error = error_norm / total_norm
    ```

    **Expected results:**

    | α (decay rate) | Relative Error | Quality |
    |----------------|----------------|---------|
    | 0.5 (slow decay) | ~15-20% | Moderate |
    | 0.7 (typical) | ~8-12% | Good |
    | 1.0 (fast decay) | ~3-5% | Excellent |

    **For typical gradients (α ≈ 0.7, r = 20):**
    $$\boxed{\text{Relative error} \approx 10\%}$$

    **Note:** PowerSGD with error feedback corrects this approximation error over time, so even 10-15% per-step error yields convergent training.

5. **Hybrid compression**: Design a compression scheme that uses quantization for small gradients (< 1MB) and sparsification for large gradients. Implement and measure the overhead.

??? success "Solution"
    **Design rationale:**

    | Gradient Size | Best Method | Reason |
    |---------------|-------------|--------|
    | < 1 MB | Quantization | Low overhead, fixed compression |
    | ≥ 1 MB | Sparsification | Higher compression, scales well |

    **Quantization for small gradients:**
    - Fixed 4-bit quantization → 4× compression
    - Overhead: O(n) for quantize/dequantize
    - Best when n is small (overhead dominates)

    **Sparsification for large gradients:**
    - Top-1% with error feedback → 100× compression
    - Overhead: O(n log k) for top-k selection
    - Amortized well over large tensors

    **Implementation:**

    ```python
    import torch
    import time

    class HybridCompressor:
        def __init__(self, size_threshold=1_000_000, k_ratio=0.01, bits=4):
            self.size_threshold = size_threshold  # 1MB = 250K float32
            self.k_ratio = k_ratio
            self.bits = bits
            self.error_buffers = {}  # Error feedback for sparsification

        def quantize(self, tensor):
            """4-bit quantization with scaling"""
            min_val = tensor.min()
            max_val = tensor.max()

            # Scale to [0, 2^bits - 1]
            scale = (max_val - min_val) / (2**self.bits - 1)
            if scale == 0:
                return torch.zeros_like(tensor, dtype=torch.uint8), min_val, scale

            quantized = ((tensor - min_val) / scale).round().to(torch.uint8)
            return quantized, min_val, scale

        def dequantize(self, quantized, min_val, scale):
            """Reconstruct from quantized values"""
            return quantized.float() * scale + min_val

        def sparsify(self, tensor, name):
            """Top-K sparsification with error feedback"""
            # Add error feedback
            if name not in self.error_buffers:
                self.error_buffers[name] = torch.zeros_like(tensor)

            accumulated = tensor + self.error_buffers[name]

            # Select top-K
            k = max(1, int(tensor.numel() * self.k_ratio))
            values, indices = torch.topk(accumulated.abs().flatten(), k)
            threshold = values[-1]

            # Create sparse representation
            mask = accumulated.abs() >= threshold
            sparse_values = accumulated[mask]

            # Update error buffer
            self.error_buffers[name] = accumulated.clone()
            self.error_buffers[name][mask] = 0

            return sparse_values, mask

        def desparsify(self, sparse_values, mask, shape):
            """Reconstruct from sparse representation"""
            result = torch.zeros(shape).flatten()
            result[mask.flatten()] = sparse_values
            return result.reshape(shape)

        def compress(self, gradient, name="default"):
            """Hybrid compression based on size"""
            numel = gradient.numel()

            if numel < self.size_threshold:
                # Use quantization for small tensors
                quantized, min_val, scale = self.quantize(gradient)
                return ('quant', quantized, min_val, scale, gradient.shape)
            else:
                # Use sparsification for large tensors
                sparse_values, mask = self.sparsify(gradient, name)
                return ('sparse', sparse_values, mask, gradient.shape)

        def decompress(self, compressed):
            """Decompress based on method used"""
            if compressed[0] == 'quant':
                _, quantized, min_val, scale, shape = compressed
                return self.dequantize(quantized, min_val, scale).reshape(shape)
            else:
                _, sparse_values, mask, shape = compressed
                return self.desparsify(sparse_values, mask, shape)

    def benchmark_hybrid():
        compressor = HybridCompressor()
        results = []

        test_sizes = [
            (100_000, "Small (100K)"),      # < 1MB
            (500_000, "Medium (500K)"),     # < 1MB
            (1_000_000, "Threshold (1M)"),  # = 1MB
            (10_000_000, "Large (10M)"),    # > 1MB
            (50_000_000, "XLarge (50M)"),   # >> 1MB
        ]

        for size, label in test_sizes:
            gradient = torch.randn(size)

            # Warmup
            for _ in range(3):
                compressed = compressor.compress(gradient, f"layer_{size}")
                _ = compressor.decompress(compressed)

            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            for _ in range(10):
                compressed = compressor.compress(gradient, f"layer_{size}")
                reconstructed = compressor.decompress(compressed)

            elapsed = (time.perf_counter() - start) / 10

            # Calculate compression ratio
            original_bytes = gradient.numel() * 4
            if compressed[0] == 'quant':
                compressed_bytes = compressed[1].numel() + 8  # quantized + metadata
            else:
                compressed_bytes = compressed[1].numel() * 4 + compressed[2].numel() // 8

            compression_ratio = original_bytes / compressed_bytes

            results.append({
                'size': label,
                'method': compressed[0],
                'time_ms': elapsed * 1000,
                'compression': compression_ratio,
                'overhead_per_elem_ns': elapsed * 1e9 / size
            })

        return results
    ```

    **Expected overhead measurements:**

    | Gradient Size | Method | Compress Time | Compression | Overhead/elem |
    |---------------|--------|---------------|-------------|---------------|
    | 100K | Quant | 0.2 ms | 4× | 2 ns |
    | 500K | Quant | 0.8 ms | 4× | 1.6 ns |
    | 1M | Sparse | 5 ms | 100× | 5 ns |
    | 10M | Sparse | 40 ms | 100× | 4 ns |
    | 50M | Sparse | 180 ms | 100× | 3.6 ns |

    **Design trade-offs:**

    | Consideration | Quantization | Sparsification |
    |---------------|--------------|----------------|
    | Compression ratio | Fixed (4×) | High (100×) |
    | Overhead scaling | O(n) | O(n log k) |
    | Memory for error | None | O(n) per tensor |
    | Accuracy impact | Bounded error | Error feedback needed |

    **Summary:**

    $$\boxed{\text{Hybrid: Quantize if } n < 1\text{M, else TopK-1\% with error feedback}}$$

    The crossover at 1M elements balances:
    - Quantization's lower overhead for small tensors
    - Sparsification's better compression for large tensors

6. **Convergence experiment**: Train CIFAR-10 with ResNet-18 using: (a) no compression, (b) 8-bit quantization, (c) Top-1% sparsification, (d) PowerSGD rank-4. Compare convergence curves and final accuracy.

??? success "Solution"
    **Experimental setup:**

    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    # Compression methods
    class NoCompression:
        def compress(self, grad):
            return grad

    class Quantize8Bit:
        def compress(self, grad):
            min_val, max_val = grad.min(), grad.max()
            scale = (max_val - min_val) / 255
            quantized = ((grad - min_val) / scale).round()
            return quantized * scale + min_val

    class TopKSparsify:
        def __init__(self, k_ratio=0.01):
            self.k_ratio = k_ratio
            self.error = {}

        def compress(self, grad, name="default"):
            if name not in self.error:
                self.error[name] = torch.zeros_like(grad)

            accumulated = grad + self.error[name]
            k = max(1, int(grad.numel() * self.k_ratio))
            vals, idx = torch.topk(accumulated.abs().flatten(), k)
            mask = accumulated.abs() >= vals[-1]

            sparse = accumulated * mask
            self.error[name] = accumulated - sparse
            return sparse

    class PowerSGDRank4:
        def __init__(self, rank=4):
            self.rank = rank
            self.P = {}
            self.Q = {}

        def compress(self, grad, name="default"):
            if grad.dim() < 2:
                return grad  # Skip 1D tensors

            m, n = grad.shape[0], grad.numel() // grad.shape[0]
            g = grad.reshape(m, n)

            if name not in self.Q:
                self.Q[name] = torch.randn(n, self.rank, device=grad.device)
                self.Q[name], _ = torch.linalg.qr(self.Q[name])

            # Power iteration
            P = g @ self.Q[name]
            P, _ = torch.linalg.qr(P)
            Q = g.T @ P
            Q, _ = torch.linalg.qr(Q)

            self.P[name] = P
            self.Q[name] = Q

            # Reconstruct
            return (P @ (P.T @ g @ Q) @ Q.T).reshape(grad.shape)

    def train_with_compression(compressor, compressor_name, epochs=100):
        # Data
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, transform=transform)
        testloader = DataLoader(testset, batch_size=100, shuffle=False)

        # Model
        model = torchvision.models.resnet18(num_classes=10)
        model = model.cuda() if torch.cuda.is_available() else model

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {'train_loss': [], 'test_acc': []}

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for inputs, targets in trainloader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                # Apply compression to gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if hasattr(compressor, 'error'):
                            param.grad.data = compressor.compress(
                                param.grad.data, name)
                        else:
                            param.grad.data = compressor.compress(param.grad.data)

                optimizer.step()
                total_loss += loss.item()

            # Evaluate
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in testloader:
                    if torch.cuda.is_available():
                        inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

            acc = 100. * correct / total
            history['train_loss'].append(total_loss / len(trainloader))
            history['test_acc'].append(acc)

            scheduler.step()

        return history
    ```

    **Expected results:**

    | Method | Final Accuracy | Accuracy Gap | Convergence |
    |--------|----------------|--------------|-------------|
    | No compression | 94.5-95.5% | Baseline | Normal |
    | 8-bit quantization | 94.0-95.0% | -0.3 to -0.5% | ~Same |
    | Top-1% sparsification | 93.5-94.5% | -0.5 to -1.0% | Slightly slower |
    | PowerSGD rank-4 | 93.0-94.0% | -1.0 to -1.5% | 5-10% slower |

    **Convergence curve observations:**

    | Epoch Range | Behavior |
    |-------------|----------|
    | 1-20 | All methods track closely |
    | 20-50 | Sparsification slightly lags |
    | 50-80 | PowerSGD shows gap |
    | 80-100 | All methods plateau |

    **Key findings:**

    1. **8-bit quantization**: Nearly lossless for ResNet-18
       - 4× compression with <0.5% accuracy loss
       - Convergence rate matches baseline

    2. **Top-1% sparsification**: Aggressive but viable
       - 100× compression with ~1% accuracy loss
       - Error feedback critical for convergence
       - Early epochs show more variance

    3. **PowerSGD rank-4**: Highest approximation error
       - ~100× compression for large weight matrices
       - Struggles with small layers (1D biases)
       - Benefits from warmup period

    **Accuracy vs. compression trade-off:**

    | Method | Compression | Accuracy Loss | Efficiency Score |
    |--------|-------------|---------------|------------------|
    | 8-bit | 4× | 0.3% | 13.3 |
    | Top-1% | 100× | 0.8% | 125 |
    | PowerSGD r4 | ~80× | 1.2% | 67 |

    Efficiency score = Compression / Accuracy Loss

    **Recommendations:**

    $$\boxed{\text{For CIFAR-10/ResNet-18: 8-bit quantization offers best accuracy; Top-1\% best efficiency}}$$

    For larger models (billions of parameters), PowerSGD becomes more attractive as:
    - Large weight matrices benefit most from low-rank
    - Communication savings dominate compute overhead
    - Error feedback accumulates improvements

## Key Takeaways

1. **Gradient redundancy enables compression**: Temporal, spatial, and magnitude redundancies allow 100-1000× compression.

2. **Error feedback is essential**: Without it, biased compression degrades convergence.

3. **Unbiased compression preserves convergence rate**: QSGD, Random-K, and other unbiased methods maintain O(1/√T) convergence.

4. **Different methods for different regimes**: Quantization for moderate compression, sparsification for high compression, low-rank for extreme cases.

5. **Compression overhead matters**: The compute cost of compression must not exceed communication savings.

6. **Warmup helps**: Starting with less compression and increasing during training improves stability.

7. **PowerSGD for large matrices**: Low-rank approximation is ideal for weight gradient matrices in large models.
