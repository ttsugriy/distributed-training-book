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

2. **Bisection Debugging**: A model fails at 64 GPUs but works at 8. Use the bisection method to find the exact scale where it breaks. Document what changes at that scale.

3. **Five Whys Practice**: Take a recent training failure and apply the Five Whys analysis. Go at least five levels deep. What root cause do you find?

4. **Correlation Analysis**: Instrument your training to record events (checkpoint save, LR change, batch size change) and failures. After a week, analyze correlations. What events predict failures?

5. **Prevention Checklist**: Create a custom prevention checklist for your specific model and infrastructure. Run through it before your next large training run.

6. **Fix Verification**: Apply a fix to a known problem. Use the FixVerifier to confirm the problem is resolved without introducing regressions.

## Key Takeaways

1. **Systematic beats ad-hoc**: The five-phase protocol ensures nothing is missed.

2. **Triage quickly**: The first five minutes determine how efficient your investigation will be.

3. **Isolate before fixing**: Don't guess—prove which component is failing.

4. **Root cause, not symptoms**: Five Whys reveals the real issue.

5. **Verify fixes**: A fix that introduces new problems isn't a fix.

6. **Prevent, don't just react**: Checklists and monitoring prevent known issues.

7. **Document everything**: Your investigation today helps the next person tomorrow.

