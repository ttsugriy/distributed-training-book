---
title: "Investigation Protocol"
subtitle: "A Systematic Approach to Distributed Training Problems"
---

<div class="chapter-opener" markdown>
When training fails at 3 AM across 256 GPUs, you need a methodology, not luck. This chapter provides a systematic protocol for investigating distributed training problems—from initial symptoms to root cause to fix.
</div>

<div class="investigation-question" markdown>
**The Question**: Your training run has stalled. Loss hasn't decreased in hours. Some GPUs show 100% utilization, others 20%. One node is unreachable. Where do you start? In what order do you investigate? How do you isolate the problem?
</div>

## The Investigation Mindset

Distributed training problems are fundamentally different from single-machine bugs:

1. **Non-reproducibility**: Timing-dependent issues may not recur
2. **Partial observability**: You can't see everything at once
3. **Cascading failures**: One failure triggers others
4. **Scale magnification**: Rare events become common at scale

The investigation protocol is designed to handle these challenges systematically.

## The Five Phases

```
┌─────────────────────────────────────────────────────────────┐
│                    INVESTIGATION PROTOCOL                    │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: TRIAGE         → Is this urgent? What's broken?   │
│  Phase 2: LOCALIZATION   → Where is the problem?            │
│  Phase 3: ISOLATION      → What exactly is failing?         │
│  Phase 4: ROOT CAUSE     → Why is it failing?               │
│  Phase 5: RESOLUTION     → How do we fix it and prevent it? │
└─────────────────────────────────────────────────────────────┘
```

Each phase has specific questions, tools, and outputs.

## Phase 1: Triage

**Goal**: Assess severity and categorize the problem.

### The First Five Minutes

```
1. Is training still running?
   ├── Yes → Proceed to performance investigation
   └── No → Proceed to crash investigation

2. If crashed, when did it crash?
   ├── Immediately → Configuration or environment issue
   ├── After warmup → Numerical instability or resource exhaustion
   └── After hours/days → Hardware failure or rare race condition

3. What are the symptoms?
   ├── All ranks crashed → Collective failure or shared resource issue
   ├── One rank crashed → Local hardware or OOM
   ├── Training hangs → Deadlock or straggler
   └── Slow training → Performance regression
```

### Quick Health Check

```python
def triage_check():
    """First-response triage for distributed training."""
    checks = {
        'processes_alive': check_all_ranks_alive(),
        'gpus_visible': check_gpu_visibility(),
        'nccl_initialized': check_nccl_init(),
        'memory_available': check_memory_headroom(),
        'network_reachable': check_inter_node_connectivity(),
        'recent_checkpoints': check_checkpoint_age(),
    }

    critical = []
    warning = []

    for check, status in checks.items():
        if status == 'FAIL':
            critical.append(check)
        elif status == 'WARN':
            warning.append(check)

    return {
        'severity': 'CRITICAL' if critical else ('WARNING' if warning else 'OK'),
        'critical': critical,
        'warning': warning,
    }

def check_all_ranks_alive():
    """Check if all distributed processes are running."""
    try:
        alive_tensor = torch.ones(1, device='cuda')
        dist.all_reduce(alive_tensor, op=dist.ReduceOp.SUM)
        expected = dist.get_world_size()
        if alive_tensor.item() == expected:
            return 'OK'
        else:
            return 'FAIL'
    except Exception as e:
        return 'FAIL'

def check_memory_headroom():
    """Check for sufficient GPU memory."""
    allocated = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    usage = allocated / total

    if usage > 0.95:
        return 'FAIL'
    elif usage > 0.85:
        return 'WARN'
    return 'OK'
```

### Triage Decision Tree

```
START
  │
  ▼
Is training running? ──No──► Did it ever start?
  │                              │
  │Yes                          │No──► Environment issue (Phase 2A)
  │                              │
  │                             │Yes──► Crash analysis (Phase 2B)
  ▼
Is loss decreasing? ──No──► Has loss ever decreased?
  │                              │
  │Yes                          │No──► Initialization issue
  │                              │
  │                             │Yes──► Training stalled (Phase 2C)
  ▼
Is step time stable? ──No──► Performance regression (Phase 2D)
  │
  │Yes
  ▼
Proceed with monitoring
```

## Phase 2: Localization

**Goal**: Determine where the problem is occurring.

### 2A: Environment Issues

Problems that prevent training from starting.

```python
class EnvironmentChecker:
    """Check for environment configuration issues."""

    def check_all(self):
        results = {}

        # CUDA
        results['cuda_available'] = torch.cuda.is_available()
        results['cuda_device_count'] = torch.cuda.device_count()

        # NCCL
        results['nccl_version'] = self._get_nccl_version()

        # Network
        results['master_addr'] = os.environ.get('MASTER_ADDR', 'NOT SET')
        results['master_port'] = os.environ.get('MASTER_PORT', 'NOT SET')
        results['world_size'] = os.environ.get('WORLD_SIZE', 'NOT SET')
        results['rank'] = os.environ.get('RANK', 'NOT SET')

        # File system
        results['checkpoint_dir_writable'] = self._check_checkpoint_dir()

        return results

    def diagnose(self, results):
        """Provide diagnosis based on check results."""
        issues = []

        if not results['cuda_available']:
            issues.append("CUDA not available. Check driver installation.")

        if results['cuda_device_count'] == 0:
            issues.append("No CUDA devices found. Check GPU visibility.")

        if results['master_addr'] == 'NOT SET':
            issues.append("MASTER_ADDR not set. Required for distributed.")

        if not results['checkpoint_dir_writable']:
            issues.append("Cannot write to checkpoint directory.")

        return issues

    def _get_nccl_version(self):
        try:
            return torch.cuda.nccl.version()
        except:
            return "Unknown"

    def _check_checkpoint_dir(self):
        ckpt_dir = os.environ.get('CHECKPOINT_DIR', './checkpoints')
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
            test_file = os.path.join(ckpt_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return True
        except:
            return False
```

### 2B: Crash Analysis

For training that started but crashed.

```python
class CrashAnalyzer:
    """Analyze crash logs and stack traces."""

    KNOWN_PATTERNS = {
        r'CUDA out of memory': {
            'category': 'OOM',
            'likely_cause': 'Batch size too large or memory leak',
            'next_steps': ['Reduce batch size', 'Enable gradient checkpointing',
                          'Check for memory leaks with torch.cuda.memory_snapshot()']
        },
        r'NCCL error|ncclSystemError': {
            'category': 'NETWORK',
            'likely_cause': 'Network timeout or hardware failure',
            'next_steps': ['Check network connectivity', 'Increase NCCL timeout',
                          'Check for hardware errors with nvidia-smi']
        },
        r'RuntimeError: Expected all tensors to be on the same device': {
            'category': 'DEVICE_MISMATCH',
            'likely_cause': 'Tensor on wrong device',
            'next_steps': ['Check .to(device) calls', 'Verify data loading pipeline']
        },
        r'loss.*nan|NaN': {
            'category': 'NUMERICAL',
            'likely_cause': 'Numerical instability',
            'next_steps': ['Check learning rate', 'Enable gradient clipping',
                          'Use loss scaling for mixed precision']
        },
        r'Timeout|deadline exceeded': {
            'category': 'TIMEOUT',
            'likely_cause': 'Collective operation timed out',
            'next_steps': ['Check for stragglers', 'Increase timeout',
                          'Check network bandwidth']
        }
    }

    def analyze_log(self, log_content: str) -> dict:
        """Analyze a crash log for known patterns."""
        import re

        findings = []
        for pattern, info in self.KNOWN_PATTERNS.items():
            if re.search(pattern, log_content, re.IGNORECASE):
                findings.append({
                    'pattern': pattern,
                    **info
                })

        return {
            'findings': findings,
            'unknown': len(findings) == 0,
            'log_snippet': log_content[-1000:]  # Last 1000 chars
        }

    def aggregate_rank_logs(self, log_dir: str) -> dict:
        """Aggregate crash info from all rank logs."""
        from pathlib import Path

        rank_findings = {}
        for log_file in Path(log_dir).glob("rank_*.log"):
            rank = int(log_file.stem.split('_')[1])
            with open(log_file) as f:
                content = f.read()
            rank_findings[rank] = self.analyze_log(content)

        # Find common patterns
        categories = {}
        for rank, findings in rank_findings.items():
            for f in findings['findings']:
                cat = f['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(rank)

        return {
            'per_rank': rank_findings,
            'categories': categories,
            'all_same': len(categories) == 1,
        }
```

### 2C: Training Stalled

Loss not decreasing or training hung.

```python
class StallDetector:
    """Detect and diagnose training stalls."""

    def __init__(self, window_size: int = 100):
        self.loss_history = []
        self.window_size = window_size
        self.last_progress_time = time.time()

    def record_loss(self, loss: float):
        """Record a loss value."""
        self.loss_history.append((time.time(), loss))

        # Keep only recent history
        if len(self.loss_history) > self.window_size * 2:
            self.loss_history = self.loss_history[-self.window_size:]

    def is_stalled(self) -> tuple:
        """Check if training has stalled."""
        if len(self.loss_history) < self.window_size:
            return False, "Insufficient data"

        recent = self.loss_history[-self.window_size:]
        older = self.loss_history[-2*self.window_size:-self.window_size]

        if not older:
            return False, "Insufficient data"

        recent_avg = sum(l for _, l in recent) / len(recent)
        older_avg = sum(l for _, l in older) / len(older)

        # Check for improvement
        improvement = (older_avg - recent_avg) / older_avg

        if improvement < 0.001:  # Less than 0.1% improvement
            return True, f"Loss plateau: {older_avg:.4f} → {recent_avg:.4f}"

        if improvement < 0:  # Loss increasing
            return True, f"Loss increasing: {older_avg:.4f} → {recent_avg:.4f}"

        return False, f"Normal progress: {improvement:.2%} improvement"

    def diagnose_stall(self) -> list:
        """Provide potential causes for a stall."""
        causes = [
            "Learning rate too low or decayed too much",
            "Gradient clipping too aggressive",
            "Batch size changed without LR adjustment",
            "Model capacity insufficient for task",
            "Data loader returning same data (shuffle issue)",
            "Optimizer state corrupted after checkpoint load",
        ]
        return causes
```

### 2D: Performance Regression

Training runs but slower than expected.

```python
class PerformanceRegression:
    """Detect and diagnose performance regressions."""

    def __init__(self, baseline_step_time: float):
        self.baseline = baseline_step_time
        self.step_times = []

    def record_step(self, step_time: float):
        self.step_times.append(step_time)

    def detect_regression(self, threshold: float = 1.1) -> tuple:
        """Detect if current performance is regressed."""
        if len(self.step_times) < 10:
            return False, "Insufficient data"

        current_avg = sum(self.step_times[-10:]) / 10
        ratio = current_avg / self.baseline

        if ratio > threshold:
            return True, f"Step time {ratio:.1%} of baseline ({current_avg:.2f}s vs {self.baseline:.2f}s)"

        return False, f"Normal: {ratio:.1%} of baseline"

    def diagnose_slowdown(self) -> dict:
        """Categorize potential causes of slowdown."""
        return {
            'compute': [
                "Thermal throttling (check GPU temperature)",
                "Power capping (check nvidia-smi)",
                "Memory bandwidth saturation",
                "Kernel launch overhead increase",
            ],
            'communication': [
                "Network congestion from other jobs",
                "Straggler node",
                "NCCL algorithm changed",
                "Bucket size misconfiguration",
            ],
            'memory': [
                "Increased fragmentation",
                "Swap/paging due to CPU memory pressure",
                "Checkpoint saving blocking training",
            ],
            'data': [
                "Data loader bottleneck",
                "Disk I/O contention",
                "Network storage latency",
            ]
        }
```

## Phase 3: Isolation

**Goal**: Isolate the specific component or interaction causing the problem.

### The Bisection Method

Divide and conquer to find the problematic component.

```python
class BisectionDebugger:
    """Systematic isolation through bisection."""

    def bisect_scale(self, working_scale: int, failing_scale: int):
        """
        Find the scale at which training breaks.

        Args:
            working_scale: Known working GPU count
            failing_scale: Known failing GPU count
        """
        print(f"Bisecting between {working_scale} and {failing_scale} GPUs")

        while failing_scale - working_scale > 1:
            mid = (working_scale + failing_scale) // 2
            print(f"  Testing {mid} GPUs...")

            success = self._test_at_scale(mid)

            if success:
                working_scale = mid
                print(f"    → Works at {mid}")
            else:
                failing_scale = mid
                print(f"    → Fails at {mid}")

        print(f"\nBreakpoint: between {working_scale} and {failing_scale} GPUs")
        return working_scale, failing_scale

    def bisect_batch_size(self, working_batch: int, failing_batch: int):
        """Find the batch size at which OOM occurs."""
        print(f"Bisecting batch size between {working_batch} and {failing_batch}")

        while failing_batch - working_batch > 1:
            mid = (working_batch + failing_batch) // 2
            print(f"  Testing batch size {mid}...")

            success = self._test_batch_size(mid)

            if success:
                working_batch = mid
            else:
                failing_batch = mid

        print(f"\nMax batch size: {working_batch}")
        return working_batch

    def bisect_layers(self, model, num_layers: int):
        """Find which layer causes the issue."""
        print(f"Bisecting {num_layers} layers")

        working = 0
        failing = num_layers

        while failing - working > 1:
            mid = (working + failing) // 2
            print(f"  Testing first {mid} layers...")

            # Create partial model
            success = self._test_partial_model(model, mid)

            if success:
                working = mid
            else:
                failing = mid

        print(f"\nProblematic layer: {failing}")
        return failing

    def _test_at_scale(self, num_gpus: int) -> bool:
        """Test if training works at given scale."""
        # Implementation depends on your infrastructure
        pass

    def _test_batch_size(self, batch_size: int) -> bool:
        """Test if batch size fits in memory."""
        try:
            # Allocate test tensor
            dummy = torch.zeros(batch_size, *self.input_shape, device='cuda')
            del dummy
            torch.cuda.empty_cache()
            return True
        except RuntimeError:
            return False

    def _test_partial_model(self, model, num_layers: int) -> bool:
        """Test forward pass with subset of layers."""
        pass
```

### Component Isolation Tests

```python
class ComponentIsolator:
    """Test individual components in isolation."""

    def test_compute_only(self, model, batch):
        """Test forward/backward without communication."""
        # Disable distributed
        model_copy = copy.deepcopy(model)
        for param in model_copy.parameters():
            param.grad = None

        with torch.no_grad():
            output = model_copy(batch)

        return {'success': True, 'output_shape': output.shape}

    def test_communication_only(self, size_bytes: int, num_iterations: int = 10):
        """Test collective communication in isolation."""
        tensor = torch.zeros(size_bytes // 4, dtype=torch.float32, device='cuda')

        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        return {
            'success': True,
            'avg_time_ms': sum(times) / len(times) * 1000,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
        }

    def test_memory_only(self, allocation_gb: float):
        """Test if GPU can allocate specified memory."""
        try:
            size = int(allocation_gb * 1e9 / 4)  # float32
            tensor = torch.zeros(size, dtype=torch.float32, device='cuda')
            del tensor
            torch.cuda.empty_cache()
            return {'success': True, 'allocated_gb': allocation_gb}
        except RuntimeError as e:
            return {'success': False, 'error': str(e)}

    def test_data_loading(self, dataloader, num_batches: int = 10):
        """Test data loading in isolation."""
        times = []
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            start = time.perf_counter()
            if isinstance(batch, torch.Tensor):
                batch = batch.cuda()
            times.append(time.perf_counter() - start)

        return {
            'success': True,
            'avg_time_ms': sum(times) / len(times) * 1000,
            'batches_loaded': len(times),
        }
```

### Network Topology Analysis

```python
class TopologyAnalyzer:
    """Analyze network topology for issues."""

    def map_topology(self):
        """Map the current network topology."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Get hostname
        import socket
        hostname = socket.gethostname()

        # Gather all hostnames
        hostnames = [None] * world_size
        dist.all_gather_object(hostnames, hostname)

        # Group by host
        host_to_ranks = {}
        for r, h in enumerate(hostnames):
            if h not in host_to_ranks:
                host_to_ranks[h] = []
            host_to_ranks[h].append(r)

        return {
            'world_size': world_size,
            'num_hosts': len(host_to_ranks),
            'ranks_per_host': host_to_ranks,
            'local_ranks': host_to_ranks.get(hostname, []),
        }

    def test_pairwise_bandwidth(self, sample_ranks: list = None):
        """Test bandwidth between pairs of ranks."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if sample_ranks is None:
            # Test a few representative pairs
            sample_ranks = [(0, i) for i in range(1, min(4, world_size))]

        results = {}
        tensor_size = 100 * 1024 * 1024  # 100MB

        for src, dst in sample_ranks:
            if rank == src or rank == dst:
                tensor = torch.zeros(tensor_size // 4, dtype=torch.float32, device='cuda')

                if rank == src:
                    tensor.fill_(1.0)

                dist.barrier()
                start = time.perf_counter()

                if rank == src:
                    dist.send(tensor, dst)
                elif rank == dst:
                    dist.recv(tensor, src)

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                bandwidth = tensor_size / elapsed / 1e9  # GB/s

                if rank == 0 or rank == src:
                    results[(src, dst)] = bandwidth

        # Gather results to rank 0
        all_results = [None] * world_size
        dist.all_gather_object(all_results, results)

        if rank == 0:
            combined = {}
            for r in all_results:
                if r:
                    combined.update(r)
            return combined

        return None
```

## Phase 4: Root Cause Analysis

**Goal**: Understand exactly why the problem occurs.

### The Five Whys

```python
class FiveWhysAnalyzer:
    """Structured root cause analysis using Five Whys."""

    def analyze(self, symptom: str) -> list:
        """
        Guide through Five Whys analysis.

        Returns a chain of causes leading to root cause.
        """
        chain = [{'level': 0, 'what': symptom, 'why': None}]

        templates = {
            'OOM': [
                "Memory allocation exceeded GPU capacity",
                "Activation memory larger than expected",
                "Batch size and sequence length combination too large",
                "No gradient checkpointing enabled for deep model",
                "Root: Memory budget not calculated before training"
            ],
            'TIMEOUT': [
                "Collective operation timed out",
                "One or more ranks were slow to participate",
                "Work distribution was uneven across ranks",
                "Data loading was slow on some ranks",
                "Root: No data prefetching or imbalanced dataset sharding"
            ],
            'NUMERICAL': [
                "Loss became NaN or Inf",
                "Gradient explosion during backward pass",
                "Learning rate too high for current loss landscape",
                "No gradient clipping with unstable initialization",
                "Root: Hyperparameters not tuned for model scale"
            ],
        }

        return templates

    def interactive_analysis(self, symptom: str):
        """Interactive Five Whys session."""
        print(f"\n=== Five Whys Analysis ===")
        print(f"Symptom: {symptom}")

        chain = []
        current = symptom

        for i in range(5):
            print(f"\nWhy #{i+1}: Why did '{current}' happen?")
            # In practice, this would be interactive
            # Here we show the pattern

        return chain
```

### Hypothesis Testing

```python
class HypothesisTester:
    """Test hypotheses about failure causes."""

    def __init__(self):
        self.hypotheses = []
        self.results = []

    def add_hypothesis(self, description: str, test_fn, expected_if_true: str):
        """Add a hypothesis to test."""
        self.hypotheses.append({
            'description': description,
            'test': test_fn,
            'expected': expected_if_true,
            'status': 'pending'
        })

    def run_all(self):
        """Run all hypothesis tests."""
        print("\n=== Hypothesis Testing ===\n")

        for h in self.hypotheses:
            print(f"Testing: {h['description']}")
            print(f"  Expected if true: {h['expected']}")

            try:
                result = h['test']()
                h['status'] = 'passed' if result else 'failed'
                h['result'] = result
            except Exception as e:
                h['status'] = 'error'
                h['result'] = str(e)

            print(f"  Result: {h['status']}")
            if h['status'] == 'passed':
                print(f"  → This hypothesis is SUPPORTED")
            print()

        # Summary
        supported = [h for h in self.hypotheses if h['status'] == 'passed']
        print(f"Supported hypotheses: {len(supported)}/{len(self.hypotheses)}")
        for h in supported:
            print(f"  ✓ {h['description']}")

# Example usage
def example_hypothesis_testing():
    tester = HypothesisTester()

    # Hypothesis: OOM is due to batch size
    tester.add_hypothesis(
        "OOM caused by batch size exceeding memory",
        lambda: test_batch_size(current_batch // 2),  # Test with half batch
        "Training succeeds with smaller batch"
    )

    # Hypothesis: Timeout due to straggler
    tester.add_hypothesis(
        "Timeout caused by straggler node",
        lambda: max(get_rank_times()) > 1.5 * mean(get_rank_times()),
        "One rank consistently slower"
    )

    tester.run_all()
```

### Correlation Analysis

```python
class CorrelationAnalyzer:
    """Find correlations between events and failures."""

    def __init__(self):
        self.events = []  # List of (timestamp, event_type, details)
        self.failures = []  # List of (timestamp, failure_type, details)

    def record_event(self, event_type: str, details: dict = None):
        self.events.append((time.time(), event_type, details or {}))

    def record_failure(self, failure_type: str, details: dict = None):
        self.failures.append((time.time(), failure_type, details or {}))

    def find_correlations(self, window_seconds: float = 60) -> list:
        """Find events that frequently precede failures."""
        correlations = {}

        for f_time, f_type, f_details in self.failures:
            # Find events in window before failure
            preceding = [
                (e_type, e_details)
                for e_time, e_type, e_details in self.events
                if f_time - window_seconds < e_time < f_time
            ]

            for e_type, e_details in preceding:
                key = (e_type, f_type)
                if key not in correlations:
                    correlations[key] = 0
                correlations[key] += 1

        # Sort by frequency
        sorted_corr = sorted(correlations.items(), key=lambda x: -x[1])

        return [
            {'event': e, 'failure': f, 'count': c}
            for (e, f), c in sorted_corr
        ]

    def analyze_temporal_patterns(self) -> dict:
        """Analyze temporal patterns in failures."""
        if not self.failures:
            return {'pattern': 'none', 'details': 'No failures recorded'}

        times = [f[0] for f in self.failures]

        # Check for periodicity
        if len(times) >= 3:
            intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
            avg_interval = sum(intervals) / len(intervals)
            variance = sum((i - avg_interval)**2 for i in intervals) / len(intervals)

            if variance < avg_interval * 0.1:  # Low variance = periodic
                return {
                    'pattern': 'periodic',
                    'interval_seconds': avg_interval,
                    'details': f'Failures occur every ~{avg_interval:.0f}s'
                }

        # Check for clustering
        # ... additional analysis

        return {'pattern': 'irregular', 'details': 'No clear pattern'}
```

## Phase 5: Resolution

**Goal**: Fix the problem and prevent recurrence.

### Fix Categories

```python
class FixRegistry:
    """Registry of known fixes for common problems."""

    FIXES = {
        'OOM': {
            'immediate': [
                ('Reduce batch size', 'batch_size //= 2'),
                ('Enable gradient checkpointing', 'model.gradient_checkpointing_enable()'),
                ('Use mixed precision', 'scaler = GradScaler(); autocast()'),
            ],
            'structural': [
                ('Implement ZeRO-3', 'Use FSDP or DeepSpeed ZeRO-3'),
                ('Add activation offloading', 'Offload activations to CPU'),
                ('Model parallelism', 'Split model across GPUs'),
            ]
        },
        'TIMEOUT': {
            'immediate': [
                ('Increase timeout', 'NCCL_TIMEOUT=1800'),
                ('Add barrier before collective', 'dist.barrier()'),
            ],
            'structural': [
                ('Balance workload', 'Ensure equal batch distribution'),
                ('Add straggler detection', 'Monitor per-rank timing'),
                ('Async data loading', 'Use multiple workers with prefetch'),
            ]
        },
        'NUMERICAL': {
            'immediate': [
                ('Reduce learning rate', 'lr *= 0.1'),
                ('Add gradient clipping', 'clip_grad_norm_(model.parameters(), 1.0)'),
                ('Use dynamic loss scaling', 'GradScaler with backoff'),
            ],
            'structural': [
                ('Review initialization', 'Use appropriate init for depth'),
                ('Add layer normalization', 'Normalize between layers'),
                ('Use stable attention', 'Implement attention with numerical guards'),
            ]
        },
        'PERFORMANCE': {
            'immediate': [
                ('Enable overlap', 'bucket_cap_mb=25 for DDP'),
                ('Check thermal throttling', 'Monitor GPU temperature'),
            ],
            'structural': [
                ('Profile and optimize', 'Use Nsight to find bottlenecks'),
                ('Tune bucket size', 'Experiment with different sizes'),
                ('Enable TF32/FP16', 'Use mixed precision training'),
            ]
        }
    }

    @classmethod
    def get_fixes(cls, problem_category: str) -> dict:
        return cls.FIXES.get(problem_category, {})

    @classmethod
    def apply_fix(cls, fix_name: str, context: dict):
        """Apply a known fix."""
        # Implementation depends on fix
        pass
```

### Verification

```python
class FixVerifier:
    """Verify that fixes actually resolve the problem."""

    def __init__(self):
        self.baseline_metrics = None
        self.post_fix_metrics = None

    def capture_baseline(self, train_fn, num_steps: int = 10):
        """Capture metrics before fix."""
        self.baseline_metrics = self._run_and_measure(train_fn, num_steps)

    def verify_fix(self, train_fn, num_steps: int = 10) -> dict:
        """Verify fix by comparing to baseline."""
        self.post_fix_metrics = self._run_and_measure(train_fn, num_steps)

        comparison = {
            'problem_resolved': self._check_resolution(),
            'performance_impact': self._compare_performance(),
            'side_effects': self._check_side_effects(),
        }

        return comparison

    def _run_and_measure(self, train_fn, num_steps):
        metrics = {
            'success': True,
            'step_times': [],
            'memory_peaks': [],
            'losses': [],
        }

        try:
            for i in range(num_steps):
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()

                loss = train_fn()

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                peak_mem = torch.cuda.max_memory_allocated()

                metrics['step_times'].append(elapsed)
                metrics['memory_peaks'].append(peak_mem)
                metrics['losses'].append(loss)

        except Exception as e:
            metrics['success'] = False
            metrics['error'] = str(e)

        return metrics

    def _check_resolution(self) -> bool:
        """Check if the original problem is resolved."""
        if not self.baseline_metrics['success'] and self.post_fix_metrics['success']:
            return True

        # Additional checks based on problem type
        return self.post_fix_metrics['success']

    def _compare_performance(self) -> dict:
        """Compare performance before and after."""
        if not (self.baseline_metrics['success'] and self.post_fix_metrics['success']):
            return {'comparable': False}

        baseline_avg = sum(self.baseline_metrics['step_times']) / len(self.baseline_metrics['step_times'])
        postfix_avg = sum(self.post_fix_metrics['step_times']) / len(self.post_fix_metrics['step_times'])

        return {
            'comparable': True,
            'baseline_step_time': baseline_avg,
            'postfix_step_time': postfix_avg,
            'change_percent': (postfix_avg - baseline_avg) / baseline_avg * 100
        }

    def _check_side_effects(self) -> list:
        """Check for unexpected side effects of the fix."""
        effects = []

        # Memory increase
        if self.post_fix_metrics['success'] and self.baseline_metrics['success']:
            baseline_mem = sum(self.baseline_metrics['memory_peaks']) / len(self.baseline_metrics['memory_peaks'])
            postfix_mem = sum(self.post_fix_metrics['memory_peaks']) / len(self.post_fix_metrics['memory_peaks'])

            if postfix_mem > baseline_mem * 1.1:
                effects.append(f"Memory increased by {(postfix_mem/baseline_mem - 1)*100:.1f}%")

        return effects
```

### Prevention

```python
class PreventionChecklist:
    """Checklist to prevent common issues."""

    CHECKLISTS = {
        'pre_training': [
            "Calculate memory budget: params + grads + optimizer + activations",
            "Test at small scale first (1-2 GPUs)",
            "Verify data loading doesn't bottleneck",
            "Check network bandwidth between nodes",
            "Ensure checkpointing works before long run",
            "Set appropriate NCCL timeout",
            "Enable NCCL debug logging initially",
        ],
        'scaling_up': [
            "Adjust learning rate for new batch size",
            "Extend warmup proportionally",
            "Monitor for stragglers at new scale",
            "Profile communication overhead",
            "Verify reproducibility at small scale first",
        ],
        'long_runs': [
            "Enable automatic checkpointing",
            "Set up monitoring and alerting",
            "Configure automatic restart on failure",
            "Plan checkpoint storage capacity",
            "Enable gradient/loss monitoring",
        ]
    }

    @classmethod
    def get_checklist(cls, phase: str) -> list:
        return cls.CHECKLISTS.get(phase, [])

    @classmethod
    def run_checks(cls, phase: str) -> dict:
        """Run automated checks from checklist."""
        checklist = cls.get_checklist(phase)
        results = {}

        for item in checklist:
            # Some items can be automated
            if 'memory budget' in item.lower():
                results[item] = cls._check_memory_budget()
            elif 'network bandwidth' in item.lower():
                results[item] = cls._check_network()
            else:
                results[item] = 'manual_check_required'

        return results

    @classmethod
    def _check_memory_budget(cls) -> str:
        available = torch.cuda.get_device_properties(0).total_memory
        return f"Available: {available/1e9:.1f}GB"

    @classmethod
    def _check_network(cls) -> str:
        # Simplified network check
        return "Run bandwidth test with NCCL_DEBUG=INFO"
```

## Quick Reference: Decision Trees

### Crash Decision Tree

```
TRAINING CRASHED
      │
      ▼
Check error message
      │
      ├── "CUDA out of memory" ──────────────────────────────────┐
      │                                                          │
      ├── "NCCL" / "Timeout" ────────────────────────────────────┤
      │                                                          │
      ├── "NaN" / "Inf" ─────────────────────────────────────────┤
      │                                                          │
      └── Other ─────────────────────────────────────────────────┤
                                                                 │
                                                                 ▼
                                               ┌─────────────────────────────┐
      ┌────────────────────────────────────────│     DIAGNOSTIC ACTION       │
      │                                        └─────────────────────────────┘
      │
      │  OOM:
      │  1. Check peak memory: torch.cuda.max_memory_allocated()
      │  2. Profile memory breakdown
      │  3. Calculate theoretical requirement
      │  4. Reduce batch or enable checkpointing
      │
      │  NCCL/Timeout:
      │  1. Check which ranks failed: NCCL_DEBUG=INFO
      │  2. Test network: ping, iperf
      │  3. Check for stragglers
      │  4. Increase timeout or fix slow rank
      │
      │  Numerical:
      │  1. Find when NaN first appeared
      │  2. Check gradient norms before NaN
      │  3. Reduce LR or add gradient clipping
      │  4. Check loss scaling settings
      │
      └──────────────────────────────────────────────────────────────────────
```

### Performance Decision Tree

```
SLOW TRAINING
      │
      ▼
Profile step breakdown
      │
      ├── Forward slow ─────────► Check GPU utilization
      │                                   │
      │                          ├── Low ──► Data loading bottleneck
      │                          └── High ─► Memory bandwidth limit
      │
      ├── Backward slow ────────► Check for extra synchronization
      │                                   │
      │                          ├── Yes ──► Remove unnecessary syncs
      │                          └── No ───► Check gradient computation
      │
      ├── Communication slow ───► Check overlap efficiency
      │                                   │
      │                          ├── No overlap ──► Enable bucketing
      │                          └── Has overlap ──► Check bandwidth
      │
      └── Optimizer slow ───────► Check optimizer state size
                                          │
                                 ├── Large ──► Consider ZeRO-1
                                 └── Small ──► Check for Python overhead
```

## Exercises

1. **Build a Triage System**: Implement a complete triage system for your training infrastructure. It should categorize failures into OOM, network, numerical, and other. Test it by artificially inducing each failure type.

??? success "Solution"
    **Exercise 1: Triage System**

    ```python
    import re
    import traceback
    from enum import Enum
    from dataclasses import dataclass
    from typing import Optional, List, Dict
    import torch.distributed as dist

    class FailureCategory(Enum):
        OOM = "out_of_memory"
        NETWORK = "network"
        NUMERICAL = "numerical"
        HARDWARE = "hardware"
        SOFTWARE = "software"
        UNKNOWN = "unknown"

    @dataclass
    class TriageResult:
        category: FailureCategory
        confidence: float  # 0.0 to 1.0
        details: str
        suggested_actions: List[str]
        affected_ranks: List[int]

    class TriageSystem:
        """Automatic failure categorization for distributed training."""

        OOM_PATTERNS = [
            r"CUDA out of memory",
            r"RuntimeError: CUDA error: out of memory",
            r"torch.cuda.OutOfMemoryError",
            r"Tried to allocate .* GiB",
            r"OOM",
        ]

        NETWORK_PATTERNS = [
            r"NCCL error",
            r"Connection refused",
            r"Timeout",
            r"Socket",
            r"ETIMEDOUT",
            r"Connection reset",
            r"NCCL WARN",
        ]

        NUMERICAL_PATTERNS = [
            r"NaN",
            r"Inf",
            r"overflow",
            r"underflow",
            r"Loss is nan",
            r"Gradient overflow",
        ]

        HARDWARE_PATTERNS = [
            r"ECC error",
            r"GPU has fallen off the bus",
            r"Xid",
            r"hardware exception",
            r"NVLink error",
        ]

        def triage(self, error: Exception, logs: str = "") -> TriageResult:
            """Categorize a failure based on exception and logs."""
            error_str = str(error) + "\n" + traceback.format_exc() + "\n" + logs

            # Check patterns in order of specificity
            checks = [
                (self.OOM_PATTERNS, FailureCategory.OOM, self._oom_actions),
                (self.HARDWARE_PATTERNS, FailureCategory.HARDWARE, self._hardware_actions),
                (self.NETWORK_PATTERNS, FailureCategory.NETWORK, self._network_actions),
                (self.NUMERICAL_PATTERNS, FailureCategory.NUMERICAL, self._numerical_actions),
            ]

            for patterns, category, action_fn in checks:
                matches = self._count_matches(error_str, patterns)
                if matches > 0:
                    confidence = min(0.5 + 0.1 * matches, 0.95)
                    return TriageResult(
                        category=category,
                        confidence=confidence,
                        details=self._extract_details(error_str, patterns),
                        suggested_actions=action_fn(),
                        affected_ranks=self._find_affected_ranks(error_str),
                    )

            return TriageResult(
                category=FailureCategory.UNKNOWN,
                confidence=0.3,
                details=str(error)[:500],
                suggested_actions=["Collect full logs", "Check system health", "Review recent changes"],
                affected_ranks=[],
            )

        def _count_matches(self, text: str, patterns: List[str]) -> int:
            return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))

        def _extract_details(self, text: str, patterns: List[str]) -> str:
            for pattern in patterns:
                match = re.search(f".*{pattern}.*", text, re.IGNORECASE)
                if match:
                    return match.group(0)[:200]
            return ""

        def _find_affected_ranks(self, text: str) -> List[int]:
            ranks = set()
            for match in re.finditer(r"rank[:\s]+(\d+)", text, re.IGNORECASE):
                ranks.add(int(match.group(1)))
            return sorted(ranks)

        def _oom_actions(self) -> List[str]:
            return [
                "Reduce batch size",
                "Enable gradient checkpointing",
                "Increase tensor parallelism",
                "Enable ZeRO-3 if using ZeRO-1/2",
                "Check for memory leaks in custom code",
            ]

        def _network_actions(self) -> List[str]:
            return [
                "Check network connectivity between nodes",
                "Verify NCCL_IB_DISABLE setting",
                "Increase NCCL timeout (NCCL_TIMEOUT)",
                "Check for stragglers causing timeouts",
                "Verify firewall rules",
            ]

        def _numerical_actions(self) -> List[str]:
            return [
                "Reduce learning rate",
                "Enable loss scaling if using FP16",
                "Check for data corruption",
                "Add gradient clipping",
                "Verify model initialization",
            ]

        def _hardware_actions(self) -> List[str]:
            return [
                "Run nvidia-smi to check GPU health",
                "Check ECC error counts",
                "Verify GPU temperatures",
                "Consider excluding problematic GPU",
                "Contact infrastructure team",
            ]

    # Test harness to induce failures
    def test_triage_system():
        triage = TriageSystem()

        # Test OOM
        try:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")
        except Exception as e:
            result = triage.triage(e)
            assert result.category == FailureCategory.OOM
            print(f"✓ OOM detected: {result.confidence:.0%} confidence")

        # Test Network
        try:
            raise RuntimeError("NCCL error: unhandled system error, ETIMEDOUT")
        except Exception as e:
            result = triage.triage(e)
            assert result.category == FailureCategory.NETWORK
            print(f"✓ Network error detected: {result.confidence:.0%} confidence")

        # Test Numerical
        try:
            raise ValueError("Loss is nan at step 1000")
        except Exception as e:
            result = triage.triage(e)
            assert result.category == FailureCategory.NUMERICAL
            print(f"✓ Numerical error detected: {result.confidence:.0%} confidence")

        print("\nAll triage tests passed!")
    ```

2. **Bisection Debugging**: A model fails at 64 GPUs but works at 8. Use the bisection method to find the exact scale where it breaks. Document what changes at that scale.

??? success "Solution"
    **Exercise 2: Bisection Debugging**

    ```python
    from dataclasses import dataclass
    from typing import Callable, List, Tuple
    import subprocess

    @dataclass
    class BisectionResult:
        breaking_point: int
        last_working: int
        observations: List[str]

    def bisection_debug(
        test_fn: Callable[[int], bool],
        min_scale: int = 8,
        max_scale: int = 64,
    ) -> BisectionResult:
        """
        Find the exact scale where training breaks.

        Args:
            test_fn: Function that returns True if training works at given scale
            min_scale: Minimum GPU count (known to work)
            max_scale: Maximum GPU count (known to fail)

        Returns:
            BisectionResult with breaking point and observations
        """
        observations = []
        low, high = min_scale, max_scale

        # Verify boundaries
        assert test_fn(low), f"Test should pass at min_scale={low}"
        assert not test_fn(high), f"Test should fail at max_scale={high}"

        observations.append(f"Confirmed: works at {low}, fails at {high}")

        while high - low > 1:
            mid = (low + high) // 2
            # Round to power of 2 if close
            for p2 in [8, 16, 32, 64, 128]:
                if abs(mid - p2) <= 2:
                    mid = p2
                    break

            print(f"Testing scale: {mid} GPUs...")
            if test_fn(mid):
                observations.append(f"✓ Scale {mid}: PASS")
                low = mid
            else:
                observations.append(f"✗ Scale {mid}: FAIL")
                high = mid

        observations.append(f"Breaking point: {low} → {high}")
        return BisectionResult(
            breaking_point=high,
            last_working=low,
            observations=observations,
        )

    def analyze_scale_change(working: int, breaking: int) -> List[str]:
        """Analyze what changes at the breaking scale."""
        changes = []

        # Network topology changes
        if working <= 8 and breaking > 8:
            changes.append("Crossed single-node boundary (NVLink → network)")

        # NCCL algorithm changes
        if working < 16 and breaking >= 16:
            changes.append("NCCL may switch from tree to ring algorithm")

        # Memory per GPU
        changes.append(f"Per-GPU batch size: {1024//breaking} vs {1024//working}")

        # Communication volume
        changes.append(f"AllReduce participants: {working} → {breaking}")
        changes.append(f"Ring allreduce steps: {working-1} → {breaking-1}")

        # Synchronization
        changes.append(f"Barrier sync time increases with more GPUs")

        return changes

    # Example usage
    """
    def run_training_test(num_gpus: int) -> bool:
        result = subprocess.run(
            ["torchrun", f"--nproc_per_node={num_gpus}", "train.py", "--test"],
            capture_output=True,
            timeout=300,
        )
        return result.returncode == 0

    result = bisection_debug(run_training_test, min_scale=8, max_scale=64)
    print(f"Breaking point: {result.breaking_point} GPUs")
    for obs in result.observations:
        print(f"  {obs}")

    changes = analyze_scale_change(result.last_working, result.breaking_point)
    print("\nChanges at breaking point:")
    for change in changes:
        print(f"  - {change}")
    """
    ```

    **Typical findings when going from 8 → 64 GPUs:**

    | Scale | Change | Impact |
    |-------|--------|--------|
    | 8 → 16 | Cross node boundary | NVLink → InfiniBand |
    | 16 → 32 | AllReduce latency | 2× participants |
    | 32 → 64 | Memory pressure | Smaller per-GPU batch |

3. **Five Whys Practice**: Take a recent training failure and apply the Five Whys analysis. Go at least five levels deep. What root cause do you find?

??? success "Solution"
    **Exercise 3: Five Whys Practice**

    ```markdown
    # Five Whys Analysis Template

    ## Incident: Training NaN at step 10,000

    **Why 1: Why did training produce NaN?**
    → Loss exploded to infinity before becoming NaN

    **Why 2: Why did loss explode?**
    → Gradients grew exponentially over ~50 steps

    **Why 3: Why did gradients grow exponentially?**
    → Learning rate warmup completed, full LR was too high for model state

    **Why 4: Why was the learning rate too high?**
    → LR was copied from a paper using different batch size without adjustment

    **Why 5: Why wasn't LR adjusted for batch size?**
    → No documented procedure for scaling hyperparameters with batch size

    ## Root Cause
    Missing documentation/checklist for adapting hyperparameters when
    changing training configuration.

    ## Corrective Actions
    1. Immediate: Reduce LR by sqrt(batch_ratio)
    2. Short-term: Add LR scaling formula to training checklist
    3. Long-term: Implement automatic LR scaling based on batch size
    ```

    ```python
    from dataclasses import dataclass
    from typing import List, Optional

    @dataclass
    class WhyLevel:
        question: str
        answer: str
        evidence: Optional[str] = None

    class FiveWhysAnalysis:
        """Structured Five Whys analysis for training failures."""

        def __init__(self, incident: str):
            self.incident = incident
            self.levels: List[WhyLevel] = []

        def add_why(self, question: str, answer: str, evidence: str = None):
            self.levels.append(WhyLevel(question, answer, evidence))
            return self

        def get_root_cause(self) -> str:
            if len(self.levels) >= 5:
                return self.levels[-1].answer
            return "Analysis incomplete - need at least 5 levels"

        def generate_report(self) -> str:
            report = [f"# Five Whys Analysis\n", f"**Incident**: {self.incident}\n"]

            for i, level in enumerate(self.levels, 1):
                report.append(f"\n**Why {i}**: {level.question}")
                report.append(f"→ {level.answer}")
                if level.evidence:
                    report.append(f"  *Evidence*: {level.evidence}")

            if len(self.levels) >= 5:
                report.append(f"\n## Root Cause\n{self.get_root_cause()}")

            return "\n".join(report)

    # Example
    analysis = FiveWhysAnalysis("GPU 3 consistently 20% slower")
    analysis.add_why(
        "Why is GPU 3 slower?",
        "GPU 3 thermals hit 85°C, triggering throttling",
        "nvidia-smi shows temp at 85°C vs 72°C for others"
    ).add_why(
        "Why is GPU 3 hotter?",
        "Airflow to GPU 3 slot is restricted",
        "Thermal camera shows hot spot"
    ).add_why(
        "Why is airflow restricted?",
        "Adjacent cable bundle blocks intake",
        "Visual inspection confirmed"
    ).add_why(
        "Why is cable bundle there?",
        "Storage expansion installed without cable management",
        "Change log shows storage added 2 weeks ago"
    ).add_why(
        "Why wasn't cable management done?",
        "No thermal verification step in hardware change procedure",
        "Procedure document review"
    )

    print(analysis.generate_report())
    ```

4. **Correlation Analysis**: Instrument your training to record events (checkpoint save, LR change, batch size change) and failures. After a week, analyze correlations. What events predict failures?

??? success "Solution"
    **Exercise 4: Correlation Analysis**

    ```python
    import time
    import json
    from collections import defaultdict
    from dataclasses import dataclass, asdict
    from typing import List, Dict, Optional
    from datetime import datetime
    import numpy as np

    @dataclass
    class TrainingEvent:
        timestamp: float
        event_type: str  # "checkpoint", "lr_change", "batch_size_change", "failure"
        details: Dict
        step: int

    class EventLogger:
        """Log training events for correlation analysis."""

        def __init__(self, log_file: str = "training_events.jsonl"):
            self.log_file = log_file
            self.events: List[TrainingEvent] = []

        def log(self, event_type: str, details: Dict, step: int):
            event = TrainingEvent(
                timestamp=time.time(),
                event_type=event_type,
                details=details,
                step=step,
            )
            self.events.append(event)

            with open(self.log_file, "a") as f:
                f.write(json.dumps(asdict(event)) + "\n")

        def log_checkpoint(self, step: int, path: str):
            self.log("checkpoint", {"path": path}, step)

        def log_lr_change(self, step: int, old_lr: float, new_lr: float):
            self.log("lr_change", {"old": old_lr, "new": new_lr}, step)

        def log_failure(self, step: int, error: str, category: str):
            self.log("failure", {"error": error, "category": category}, step)

    class CorrelationAnalyzer:
        """Analyze correlations between events and failures."""

        def __init__(self, events: List[TrainingEvent]):
            self.events = sorted(events, key=lambda e: e.timestamp)
            self.failures = [e for e in events if e.event_type == "failure"]

        def analyze_precursors(self, window_seconds: float = 300) -> Dict[str, float]:
            """Find events that commonly precede failures."""
            precursor_counts = defaultdict(int)
            total_failures = len(self.failures)

            for failure in self.failures:
                # Look back in time window
                for event in self.events:
                    if event.event_type == "failure":
                        continue
                    time_diff = failure.timestamp - event.timestamp
                    if 0 < time_diff <= window_seconds:
                        precursor_counts[event.event_type] += 1

            # Normalize by failure count
            return {
                event_type: count / total_failures
                for event_type, count in precursor_counts.items()
            }

        def calculate_risk_ratios(self) -> Dict[str, float]:
            """
            Calculate risk ratio: P(failure | event) / P(failure | no event)
            """
            # Time windows after each event type
            WINDOW = 600  # 10 minutes

            event_types = set(e.event_type for e in self.events) - {"failure"}
            risk_ratios = {}

            total_time = self.events[-1].timestamp - self.events[0].timestamp
            base_failure_rate = len(self.failures) / total_time

            for event_type in event_types:
                events_of_type = [e for e in self.events if e.event_type == event_type]

                failures_after_event = 0
                total_window_time = 0

                for event in events_of_type:
                    window_end = event.timestamp + WINDOW
                    total_window_time += WINDOW

                    for failure in self.failures:
                        if event.timestamp < failure.timestamp <= window_end:
                            failures_after_event += 1

                if total_window_time > 0:
                    failure_rate_after = failures_after_event / total_window_time
                    risk_ratios[event_type] = failure_rate_after / base_failure_rate

            return risk_ratios

        def generate_report(self) -> str:
            precursors = self.analyze_precursors()
            risk_ratios = self.calculate_risk_ratios()

            report = ["# Event-Failure Correlation Analysis\n"]
            report.append(f"Total events: {len(self.events)}")
            report.append(f"Total failures: {len(self.failures)}\n")

            report.append("## Precursor Analysis (5-min window)")
            report.append("| Event Type | Preceded Failures |")
            report.append("|------------|-------------------|")
            for event_type, rate in sorted(precursors.items(), key=lambda x: -x[1]):
                report.append(f"| {event_type} | {rate:.0%} |")

            report.append("\n## Risk Ratios")
            report.append("| Event Type | Risk Ratio |")
            report.append("|------------|------------|")
            for event_type, ratio in sorted(risk_ratios.items(), key=lambda x: -x[1]):
                flag = "⚠️" if ratio > 2.0 else ""
                report.append(f"| {event_type} | {ratio:.2f}x {flag} |")

            return "\n".join(report)
    ```

    **Example output after 1 week:**

    | Event Type | Risk Ratio | Interpretation |
    |------------|------------|----------------|
    | checkpoint | 3.2x ⚠️ | Checkpointing may cause OOM pressure |
    | lr_change | 2.1x ⚠️ | LR increases destabilize training |
    | batch_size_change | 1.8x | Batch changes stress memory |
    | eval_start | 0.9x | Eval is safe |

5. **Prevention Checklist**: Create a custom prevention checklist for your specific model and infrastructure. Run through it before your next large training run.

??? success "Solution"
    **Exercise 5: Prevention Checklist**

    ```python
    from dataclasses import dataclass
    from typing import List, Callable, Optional
    from enum import Enum

    class CheckResult(Enum):
        PASS = "✓"
        FAIL = "✗"
        WARN = "⚠"
        SKIP = "○"

    @dataclass
    class CheckItem:
        name: str
        description: str
        check_fn: Optional[Callable[[], CheckResult]] = None
        category: str = "general"
        critical: bool = False

    class PreventionChecklist:
        """Custom prevention checklist for training runs."""

        def __init__(self, model_name: str):
            self.model_name = model_name
            self.items: List[CheckItem] = []
            self.results: dict = {}

        def add_check(self, name: str, description: str,
                      check_fn: Callable = None, category: str = "general",
                      critical: bool = False):
            self.items.append(CheckItem(name, description, check_fn, category, critical))
            return self

        def run_all(self) -> bool:
            """Run all checks, return True if no critical failures."""
            all_passed = True

            for item in self.items:
                if item.check_fn:
                    try:
                        result = item.check_fn()
                    except Exception as e:
                        result = CheckResult.FAIL
                        print(f"Check '{item.name}' raised: {e}")
                else:
                    result = CheckResult.SKIP

                self.results[item.name] = result

                if result == CheckResult.FAIL and item.critical:
                    all_passed = False

            return all_passed

        def generate_report(self) -> str:
            categories = {}
            for item in self.items:
                if item.category not in categories:
                    categories[item.category] = []
                categories[item.category].append(item)

            report = [f"# Pre-Training Checklist: {self.model_name}\n"]

            for category, items in categories.items():
                report.append(f"\n## {category.title()}")
                for item in items:
                    result = self.results.get(item.name, CheckResult.SKIP)
                    critical = " [CRITICAL]" if item.critical else ""
                    report.append(f"- {result.value} {item.name}{critical}")
                    report.append(f"  {item.description}")

            return "\n".join(report)

    # Example checklist for LLaMA-style training
    def create_llama_checklist() -> PreventionChecklist:
        import torch

        checklist = PreventionChecklist("LLaMA-70B")

        # Memory checks
        def check_gpu_memory():
            free = torch.cuda.mem_get_info()[0] / 1e9
            return CheckResult.PASS if free > 70 else CheckResult.FAIL

        checklist.add_check(
            "GPU Memory Available",
            "Verify >70GB free per GPU for 70B model",
            check_gpu_memory, "memory", critical=True
        )

        # Network checks
        checklist.add_check(
            "NCCL Version",
            "Verify NCCL >= 2.18 for optimal performance",
            category="network"
        )

        checklist.add_check(
            "InfiniBand Status",
            "Verify IB links are up: ibstat shows Active",
            category="network", critical=True
        )

        # Numerical checks
        checklist.add_check(
            "Loss Scaling Configured",
            "Verify dynamic loss scaling is enabled for FP16",
            category="numerical"
        )

        checklist.add_check(
            "Gradient Clipping",
            "Verify grad clip norm = 1.0",
            category="numerical"
        )

        # Checkpoint checks
        checklist.add_check(
            "Checkpoint Storage",
            "Verify checkpoint directory has >5TB free",
            category="checkpoint", critical=True
        )

        return checklist

    # Run checklist
    # checklist = create_llama_checklist()
    # if not checklist.run_all():
    #     print("CRITICAL CHECKS FAILED - DO NOT PROCEED")
    # print(checklist.generate_report())
    ```

6. **Fix Verification**: Apply a fix to a known problem. Use the FixVerifier to confirm the problem is resolved without introducing regressions.

??? success "Solution"
    **Exercise 6: Fix Verification**

    ```python
    import time
    import statistics
    from dataclasses import dataclass
    from typing import List, Dict, Callable, Optional
    from enum import Enum

    class VerificationStatus(Enum):
        FIXED = "fixed"
        NOT_FIXED = "not_fixed"
        REGRESSION = "regression"
        INCONCLUSIVE = "inconclusive"

    @dataclass
    class Metric:
        name: str
        value: float
        baseline: float
        tolerance: float  # Acceptable deviation from baseline

        @property
        def is_regression(self) -> bool:
            # Assuming higher is better for throughput, lower for loss
            if "throughput" in self.name.lower() or "speed" in self.name.lower():
                return self.value < self.baseline * (1 - self.tolerance)
            else:
                return self.value > self.baseline * (1 + self.tolerance)

    @dataclass
    class VerificationResult:
        status: VerificationStatus
        problem_resolved: bool
        regressions: List[str]
        metrics: Dict[str, Metric]
        notes: str

    class FixVerifier:
        """Verify that a fix resolves the problem without regressions."""

        def __init__(self, baseline_metrics: Dict[str, float]):
            self.baseline = baseline_metrics
            self.tolerance = 0.05  # 5% tolerance

        def verify(
            self,
            problem_test: Callable[[], bool],  # Returns True if problem exists
            metric_collector: Callable[[], Dict[str, float]],
            num_trials: int = 3,
        ) -> VerificationResult:
            """
            Verify a fix by:
            1. Checking if the original problem is resolved
            2. Collecting metrics to check for regressions
            3. Running multiple trials for statistical significance
            """
            # Test if problem is resolved
            problem_trials = [problem_test() for _ in range(num_trials)]
            problem_resolved = not any(problem_trials)

            # Collect metrics across trials
            all_metrics = [metric_collector() for _ in range(num_trials)]

            # Aggregate metrics
            metrics = {}
            regressions = []

            for metric_name in all_metrics[0].keys():
                values = [m[metric_name] for m in all_metrics]
                avg_value = statistics.mean(values)
                baseline_value = self.baseline.get(metric_name, avg_value)

                metric = Metric(
                    name=metric_name,
                    value=avg_value,
                    baseline=baseline_value,
                    tolerance=self.tolerance,
                )
                metrics[metric_name] = metric

                if metric.is_regression:
                    regressions.append(
                        f"{metric_name}: {baseline_value:.2f} → {avg_value:.2f}"
                    )

            # Determine overall status
            if not problem_resolved:
                status = VerificationStatus.NOT_FIXED
            elif regressions:
                status = VerificationStatus.REGRESSION
            else:
                status = VerificationStatus.FIXED

            return VerificationResult(
                status=status,
                problem_resolved=problem_resolved,
                regressions=regressions,
                metrics=metrics,
                notes=f"Ran {num_trials} trials",
            )

        def generate_report(self, result: VerificationResult) -> str:
            report = ["# Fix Verification Report\n"]

            status_emoji = {
                VerificationStatus.FIXED: "✅",
                VerificationStatus.NOT_FIXED: "❌",
                VerificationStatus.REGRESSION: "⚠️",
                VerificationStatus.INCONCLUSIVE: "❓",
            }

            report.append(f"**Status**: {status_emoji[result.status]} {result.status.value.upper()}")
            report.append(f"**Problem Resolved**: {'Yes' if result.problem_resolved else 'No'}")

            if result.regressions:
                report.append("\n## Regressions Detected")
                for reg in result.regressions:
                    report.append(f"- ⚠️ {reg}")

            report.append("\n## Metrics Comparison")
            report.append("| Metric | Baseline | Current | Status |")
            report.append("|--------|----------|---------|--------|")

            for name, metric in result.metrics.items():
                status = "🔴" if metric.is_regression else "🟢"
                report.append(f"| {name} | {metric.baseline:.2f} | {metric.value:.2f} | {status} |")

            return "\n".join(report)

    # Example usage
    """
    # Baseline before attempting fix
    baseline = {
        "throughput_tokens_sec": 150000,
        "memory_gb": 72,
        "step_time_ms": 450,
    }

    verifier = FixVerifier(baseline)

    def check_oom_problem():
        # Returns True if OOM still occurs
        try:
            run_training_step()
            return False
        except RuntimeError as e:
            return "out of memory" in str(e).lower()

    def collect_metrics():
        return {
            "throughput_tokens_sec": measure_throughput(),
            "memory_gb": get_peak_memory(),
            "step_time_ms": measure_step_time(),
        }

    result = verifier.verify(check_oom_problem, collect_metrics)
    print(verifier.generate_report(result))
    """
    ```

    **Summary of verification workflow:**

    | Step | Action | Purpose |
    |------|--------|---------|
    | 1 | Record baseline metrics | Establish performance reference |
    | 2 | Apply fix | Implement the proposed solution |
    | 3 | Test problem resolution | Confirm original issue is fixed |
    | 4 | Collect new metrics | Measure current performance |
    | 5 | Compare to baseline | Detect any regressions |
    | 6 | Generate report | Document verification results |

## Key Takeaways

1. **Systematic beats ad-hoc**: The five-phase protocol ensures nothing is missed.

2. **Triage quickly**: The first five minutes determine how efficient your investigation will be.

3. **Isolate before fixing**: Don't guess—prove which component is failing.

4. **Root cause, not symptoms**: Five Whys reveals the real issue.

5. **Verify fixes**: A fix that introduces new problems isn't a fix.

6. **Prevent, don't just react**: Checklists and monitoring prevent known issues.

7. **Document everything**: Your investigation today helps the next person tomorrow.

