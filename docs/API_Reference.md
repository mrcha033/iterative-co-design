# API Reference Guide

Comprehensive API documentation for the ICD framework, covering core modules, measurement interfaces, and integration patterns.

## Core Modules

### icd.core.solver

The solver module provides optimization algorithms for finding efficient memory permutations.

#### fit_permutation

```python
def fit_permutation(
    W: CSRMatrix,
    clusters: Optional[Sequence[Sequence[int]]] = None,
    time_budget_s: float = 1.0,
    refine_steps: int = 500,
    rng_seed: int = 0,
    method: str = "louvain"
) -> Tuple[List[int], Dict[str, Any]]
```

**Parameters**:
- `W`: Sparse correlation matrix in CSR format
- `clusters`: Pre-computed clusters (optional, auto-generated if None)
- `time_budget_s`: Maximum optimization time in seconds
- `refine_steps`: Number of local refinement iterations
- `rng_seed`: Random seed for deterministic results
- `method`: Clustering method ("louvain", "spectral", "hardware")

**Returns**:
- `permutation`: List of integers representing the optimal ordering
- `stats`: Dictionary with optimization statistics

**Example**:
```python
from icd.core.solver import fit_permutation
from icd.core.graph import CSRMatrix

# Create correlation matrix
W = CSRMatrix(indptr=[0, 2, 4], indices=[1, 2, 0, 2], data=[0.8, 0.6, 0.8, 0.9], shape=(3, 3))

# Find optimal permutation
permutation, stats = fit_permutation(W, time_budget_s=2.0, refine_steps=1000)

print(f"Optimal permutation: {permutation}")
print(f"Final cost: {stats['J_final']}")
print(f"Modularity: {stats['Q_final']}")
```

#### Hardware-Aware Optimization

```python
from icd.core.cost import CostConfig

# Configure hardware topology
config = CostConfig(
    vec_width=4,
    hardware_topology={
        "lanes": [
            {"id": 0, "l2_slice": 0, "memory_channel": 0},
            {"id": 1, "l2_slice": 0, "memory_channel": 0},
            {"id": 2, "l2_slice": 1, "memory_channel": 1},
            {"id": 3, "l2_slice": 1, "memory_channel": 1},
        ]
    }
)

# Hardware-aware permutation
permutation, stats = fit_permutation(W, method="hardware_aware", cost_config=config)
```

### icd.core.cost

Cost function evaluation and configuration.

#### eval_cost

```python
def eval_cost(
    W: CSRMatrix,
    pi: List[int],
    pi_ref: List[int],
    config: CostConfig
) -> Dict[str, float]
```

**Parameters**:
- `W`: Correlation matrix
- `pi`: Current permutation
- `pi_ref`: Reference permutation
- `config`: Cost configuration

**Returns**:
Dictionary with cost metrics:
- `"J"`: Total cost value
- `"J_ref"`: Reference cost
- `"improvement"`: Relative improvement

#### CostConfig

```python
@dataclass
class CostConfig:
    vec_width: int = 1
    hardware_topology: Optional[Dict[str, Any]] = None
    penalty_factor: float = 1.0
```

### icd.graph

Graph construction and correlation analysis.

#### compute_streaming_correlation

```python
def compute_streaming_correlation(
    samples: List[torch.Tensor],
    feature_dim: int,
    cfg: CorrelationConfig
) -> Tuple[CSRMatrix, Dict[str, Any]]
```

**Parameters**:
- `samples`: List of activation tensors
- `feature_dim`: Feature dimension size
- `cfg`: Correlation configuration

**Returns**:
- `correlation_matrix`: Sparse correlation matrix
- `metadata`: Computation metadata

**Example**:
```python
import torch
from icd.graph.streaming_correlation import compute_streaming_correlation
from icd.graph import CorrelationConfig

# Generate sample activations
samples = [torch.randn(64, 256) for _ in range(10)]

# Configure correlation computation
cfg = CorrelationConfig(
    threshold=0.1,
    normalize="sym",
    max_memory_gb=4.0
)

# Compute correlation matrix
csr_matrix, metadata = compute_streaming_correlation(samples, feature_dim=256, cfg=cfg)

print(f"Matrix shape: {csr_matrix.shape}")
print(f"Non-zeros: {csr_matrix.nnz()}")
print(f"Sparsity: {metadata['sparsity']:.3f}")
```

## Measurement Interfaces

### icd.measure.latency

High-precision latency measurement with statistical analysis.

#### LatencyMeasurer

```python
class LatencyMeasurer:
    def __init__(
        self,
        warmup_iter: int = 50,
        repeats: int = 1000,
        fixed_clock: bool = True,
        sync_gpu: bool = True,
    ):
        ...

    def measure(self, model: torch.nn.Module, inputs: Any, device: str | None = None) -> Dict[str, Any]:
        ...

    def measure_callable(self, fn: Callable[[], None]) -> Dict[str, Any]:
        ...
```

**Returned fields** (`Dict[str, Any]`):

- `mean`, `std`, `p50`, `p95`, `p99`: latency statistics in milliseconds.
- `ci95`: `(lower, upper)` tuple (95% confidence interval).
- `outliers`: count of samples outside the Tukey IQR fence (1.5×IQR).
- `raw_samples`: sorted raw latency samples (milliseconds).
- `warmup_iter`, `repeats`, `fixed_clock`: metadata copied from the measurer.
- `device`: populated when using :meth:`measure` with a PyTorch module.

**Example**:
```python
from icd.measure.latency import LatencyMeasurer
import torch

model = torch.nn.Linear(256, 128).cuda()
inputs = torch.randn(32, 256).cuda()

measurer = LatencyMeasurer(warmup_iter=100, repeats=2000)

results = measurer.measure(model, inputs)
print(f"Latency: {results['mean']:.2f} ± {results['std']:.2f} ms")
print(f"95% CI: [{results['ci95'][0]:.2f}, {results['ci95'][1]:.2f}] ms")

# Measuring an arbitrary callable (e.g., TVM GraphModule.run)
tvm_stats = measurer.measure_callable(lambda: tvm_graph_module.run())
print(f"TVM mean latency: {tvm_stats['mean']:.2f} ms")
```

### icd.measure.l2_ncu

NVIDIA Nsight Compute integration for L2 cache analysis.

#### NCUWrapper

```python
class NCUWrapper:
    def __init__(
        self,
        ncu_command: str,
        output_dir: Path,
        metrics: List[str] = None
    ):
        self.ncu_command = ncu_command
        self.output_dir = output_dir
        self.metrics = metrics or ["l2_cache_hit_rate", "dram_throughput"]
```

**Methods**:

##### profile

```python
def profile(
    self,
    executable: Path,
    args: List[str] = None
) -> Dict[str, Any]
```

**Example**:
```python
from icd.measure.l2_ncu import NCUWrapper
from pathlib import Path

# Configure NCU profiler
ncu = NCUWrapper(
    ncu_command="nv-nsight-cu-cli",
    output_dir=Path("runs/profiling"),
    metrics=["l2_cache_hit_rate", "dram__throughput.avg.pct_of_peak_sustained_elapsed"]
)

# Profile executable
results = ncu.profile(
    executable=Path("./my_inference_binary"),
    args=["--model", "bert.pt", "--input", "sample.json"]
)

print(f"L2 hit rate: {results['l2_cache_hit_rate']:.1f}%")
print(f"DRAM utilization: {results['dram_throughput']:.1f}%")
```

## Cross-Vendor Profiling

### icd.measure.rocm_profiler

AMD ROCm profiling integration.

#### ROCmProfiler

```python
class ROCmProfiler:
    def __init__(self, config: ROCmProfilerConfig):
        self.config = config
```

#### ROCmProfilerConfig

```python
@dataclass
class ROCmProfilerConfig:
    binary: Path
    output_dir: Path
    metrics: List[str] = field(default_factory=lambda: ["SQ_WAVES", "VALUUtil"])
    kernel_regex: Optional[str] = None
    additional_args: List[str] = field(default_factory=list)
```

**Example**:
```python
from icd.measure.rocm_profiler import ROCmProfiler, ROCmProfilerConfig
from pathlib import Path

# Configure ROCm profiling
config = ROCmProfilerConfig(
    binary=Path("./inference_app"),
    output_dir=Path("runs/rocm_profile"),
    metrics=["SQ_WAVES", "VALUUtil", "SALUUtil", "L2CacheHit"],
    kernel_regex=".*gemm.*|.*conv.*"
)

# Run profiling
profiler = ROCmProfiler(config)
results = profiler.collect()

# Process results
for metric_data in results["metrics"]:
    print(f"Kernel: {metric_data['KernelName']}")
    print(f"VALU Utilization: {metric_data['VALUUtil']}")
```

### icd.measure.vtune_profiler

Intel VTune profiling integration.

#### VTuneProfiler

```python
class VTuneProfiler:
    def __init__(self, config: VTuneProfilerConfig):
        self.config = config
```

#### VTuneProfilerConfig

```python
@dataclass
class VTuneProfilerConfig:
    binary: Path
    output_dir: Path
    analysis_type: str = "gpu-hotspots"
    result_name: str = "vtune_result"
    env: Dict[str, str] = field(default_factory=dict)
```

**Example**:
```python
from icd.measure.vtune_profiler import VTuneProfiler, VTuneProfilerConfig

# Configure VTune profiling
config = VTuneProfilerConfig(
    binary=Path("./cpu_inference"),
    output_dir=Path("runs/vtune"),
    analysis_type="memory-access",
    env={"OMP_NUM_THREADS": "40"}
)

# Run profiling
profiler = VTuneProfiler(config)
results = profiler.collect()

print(f"Result directory: {results['result_dir']}")
print(f"GPU utilization: {results['summary'].get('gpu_utilization', 'N/A')}")
```

## TVM Integration

### icd.adapters.tvm_export

Complete TVM compilation pipeline with AutoTVM/Ansor support.

#### ExportConfig

```python
@dataclass
class ExportConfig:
    example_input: Any
    target: str = "llvm"
    tuning_trials: int = 0
    tuning_log: Optional[Path] = None
    use_ansor: bool = False
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
```

#### compile_pytorch_module

```python
def compile_pytorch_module(
    module: torch.nn.Module,
    config: ExportConfig,
    artifacts_dir: Optional[Path] = None
) -> Optional[tvm.runtime.GraphModule]
```

**Example**:
```python
from icd.adapters.tvm_export import compile_pytorch_module, ExportConfig
import torch
from pathlib import Path

# Define model
model = torch.nn.Sequential(
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64)
)

# Configure TVM export
config = ExportConfig(
    example_input=torch.randn(1, 256),
    target="cuda",
    tuning_trials=1000,
    tuning_log=Path("autotvm.log"),
    use_ansor=False
)

# Compile with TVM
graph_module = compile_pytorch_module(
    model,
    config,
    artifacts_dir=Path("runs/tvm_artifacts")
)

# Run inference
graph_module.set_input("input_0", torch.randn(1, 256).numpy())
graph_module.run()
output = graph_module.get_output(0).numpy()
```

#### Advanced TVM Features

##### Custom Target Configuration

```python
# Multi-target compilation
targets = ["llvm", "cuda", "rocm"]
compiled_modules = {}

for target in targets:
    config = ExportConfig(
        example_input=torch.randn(1, 256),
        target=target,
        tuning_trials=500
    )

    compiled_modules[target] = compile_pytorch_module(model, config)
```

##### Ansor Auto-scheduler

```python
# Use Ansor for complex models
config = ExportConfig(
    example_input=torch.randn(1, 256),
    target="cuda",
    tuning_trials=3000,
    use_ansor=True,  # Enable auto-scheduler
    tuning_log=Path("ansor_search.log")
)

graph_module = compile_pytorch_module(model, config)
```

## CLI Integration

### Main CLI Interface

#### icd.cli.main

The primary command-line interface for running experiments.

##### run command

```bash
python -m icd.cli.main run -c CONFIG_FILE --out OUTPUT_DIR [OPTIONS]
```

**Options**:
- `--override KEY=VALUE`: Override configuration parameters
- `--dry-run`: Validate configuration without running
- `--print-schema`: Display configuration schema
- `--no-measure`: Skip measurement phase
- `--reuse-perm PATH`: Reuse existing permutation

**Examples**:
```bash
# Basic iterative run
python -m icd.cli.main run -c configs/bert.json --out runs/bert_iter

# Override parameters
python -m icd.cli.main run -c configs/bert.json \
    --override pipeline.mode=linear \
    --override solver.time_budget_s=5.0 \
    --out runs/bert_linear

# Enable profiling
python -m icd.cli.main run -c configs/bert.json \
    --override measure.ncu_enable=true \
    --override measure.power_enable=true \
    --out runs/bert_profiled
```

##### pair command

```bash
python -m icd.cli.main pair -c CONFIG_FILE --out OUTPUT_DIR
```

Automatically runs both linear and iterative modes for comparison.

### Configuration System

#### Configuration Schema

All configurations follow a standardized JSON schema:

```json
{
  "pipeline": {
    "mode": "iterative",           // "linear" | "iterative"
    "repeats": 1000,               // Number of measurement samples
    "warmup_iter": 50,             // Thermal stabilization iterations
    "runner": "module:function"    // Inference runner
  },
  "graph": {
    "source": "mock",              // "mock" | "pytorch" | "onnx"
    "mock": {
      "d": 256,                    // Dimension size
      "blocks": 4,                 // Number of blocks
      "noise": 0.02,               // Noise level
      "seed": 0                    // Random seed
    }
  },
  "solver": {
    "time_budget_s": 1.0,          // Optimization time limit
    "refine_steps": 500,           // Local refinement iterations
    "method": "louvain"            // Clustering method
  },
  "measure": {
    "ncu_enable": false,           // Enable NVIDIA profiling
    "power_enable": false,         // Enable power monitoring
    "tvm_enable": false           // Enable TVM baseline
  }
}
```

### Custom Runners

#### Implementing Custom Runners

```python
def my_custom_runner(
    model: torch.nn.Module,
    config: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Custom inference runner implementation."""

    # Extract configuration
    batch_size = context.get("batch_size", 32)
    sequence_length = context.get("sequence_length", 128)

    # Generate inputs
    inputs = torch.randint(0, 1000, (batch_size, sequence_length))

    # Run inference
    with torch.no_grad():
        outputs = model(inputs)

    # Return timing information (implement actual timing)
    return {
        "latency_ms": 12.34,
        "throughput": batch_size / 0.01234,
        "memory_usage": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    }

# Register runner
# In your_module.py:
def load_custom_model():
    return torch.nn.Embedding(1000, 256)

# Use in configuration:
{
  "pipeline": {
    "runner": "your_module:my_custom_runner",
    "runner_context": {
      "batch_size": 64,
      "sequence_length": 256
    }
  }
}
```

## Error Handling

### Common Exceptions

#### ICD Core Exceptions

```python
from icd.errors import ICDError, SolverError, MeasurementError

try:
    permutation, stats = fit_permutation(W, time_budget_s=1.0)
except SolverError as e:
    print(f"Solver failed: {e}")
    # Handle solver-specific errors

try:
    results = measurer.measure(model, inputs)
except MeasurementError as e:
    print(f"Measurement failed: {e}")
    # Handle measurement errors
```

#### TVM Integration Exceptions

```python
from icd.adapters.tvm_export import DependencyError

try:
    graph_module = compile_pytorch_module(model, config)
except DependencyError as e:
    print(f"TVM not available: {e}")
    # Graceful fallback when TVM not installed
```

### Validation Helpers

#### Configuration Validation

```python
from icd.utils.validation import validate_config

# Validate configuration before running
try:
    validate_config(config_dict)
    print("✓ Configuration valid")
except ValueError as e:
    print(f"✗ Configuration error: {e}")
```

#### Hardware Validation

```python
from icd.utils.hardware import check_gpu_availability

# Check hardware requirements
gpu_available = check_gpu_availability()
if not gpu_available:
    print("⚠ GPU not available, falling back to CPU")
```

This API reference provides comprehensive documentation for integrating and extending the ICD framework across all major components and use cases.