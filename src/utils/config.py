"""
Configuration management with Pydantic validation.
"""
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Configuration for model parameters."""
    name: str = Field(..., description="Model name")
    hf_model_id: Optional[str] = Field(None, description="HuggingFace model ID")
    pretrained_path: Optional[str] = Field(None, description="Path to pretrained model")
    precision: str = Field("float16", description="Model precision")
    
    @validator('name')
    def validate_model_name(cls, v):
        supported_models = ['mamba-3b', 'bert-large', 'resnet-50', 'gcn']
        if v.lower() not in supported_models:
            raise ValueError(f"Model {v} not supported. Supported: {supported_models}")
        return v.lower()
    
    @validator('precision')
    def validate_precision(cls, v):
        supported_precision = ['float32', 'float16', 'bfloat16']
        if v not in supported_precision:
            raise ValueError(f"Precision {v} not supported. Supported: {supported_precision}")
        return v


class DatasetConfig(BaseModel):
    """Configuration for dataset parameters."""
    name: str = Field(..., description="Dataset name")
    path: str = Field("./data/", description="Dataset path")
    sequence_length: int = Field(4096, description="Sequence length for NLP")
    batch_size: int = Field(1, description="Batch size")
    num_samples: int = Field(1000, description="Number of samples for correlation")
    num_benchmark_samples: int = Field(100, description="Number of benchmark samples")
    
    @validator('name')
    def validate_dataset_name(cls, v):
        supported_datasets = ['wikitext-103', 'imagenet', 'ogbn-arxiv']
        if v.lower() not in supported_datasets:
            raise ValueError(f"Dataset {v} not supported. Supported: {supported_datasets}")
        return v.lower()
    
    @validator('sequence_length')
    def validate_sequence_length(cls, v):
        if v <= 0:
            raise ValueError("Sequence length must be positive")
        return v


class IASPConfig(BaseModel):
    """Configuration for IASP (IO-Aware Scan Permutation)."""
    layer_name: str = Field(..., description="Target layer name")
    num_clusters: int = Field(64, description="Number of clusters")
    correlation_threshold: float = Field(0.1, description="Correlation threshold")
    method: str = Field("spectral", description="Permutation method")
    precomputed_path: str = Field("./data/correlation_matrices/", description="Precomputed matrices path")
    
    @validator('num_clusters')
    def validate_num_clusters(cls, v):
        if v <= 0:
            raise ValueError("Number of clusters must be positive")
        return v
    
    @validator('correlation_threshold')
    def validate_correlation_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Correlation threshold must be between 0 and 1")
        return v
    
    @validator('method')
    def validate_method(cls, v):
        supported_methods = ['spectral', 'tsp', 'random']
        if v.lower() not in supported_methods:
            raise ValueError(f"Method {v} not supported. Supported: {supported_methods}")
        return v.lower()


class HDSConfig(BaseModel):
    """Configuration for HDS (Hardware-Native Differentiable Sparsity)."""
    pattern: str = Field("2:4", description="Sparsity pattern")
    learning_rate: float = Field(1e-5, description="Fine-tuning learning rate")
    num_epochs: int = Field(5, description="Number of fine-tuning epochs")
    gumbel_temperature: float = Field(1.0, description="Gumbel temperature")
    sparsity_ratio: float = Field(0.5, description="Target sparsity ratio")
    
    @validator('pattern')
    def validate_pattern(cls, v):
        supported_patterns = ['2:4', '4:8', '1:2']
        if v not in supported_patterns:
            raise ValueError(f"Pattern {v} not supported. Supported: {supported_patterns}")
        return v
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v):
        if v <= 0:
            raise ValueError("Learning rate must be positive")
        return v
    
    @validator('sparsity_ratio')
    def validate_sparsity_ratio(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Sparsity ratio must be between 0 and 1")
        return v


class PTQConfig(BaseModel):
    """Configuration for PTQ (Post-Training Quantization)."""
    bits: int = Field(8, description="Quantization bits")
    scheme: str = Field("symmetric", description="Quantization scheme")
    calibration_samples: int = Field(512, description="Calibration dataset size")
    
    @validator('bits')
    def validate_bits(cls, v):
        supported_bits = [4, 8, 16]
        if v not in supported_bits:
            raise ValueError(f"Bits {v} not supported. Supported: {supported_bits}")
        return v
    
    @validator('scheme')
    def validate_scheme(cls, v):
        supported_schemes = ['symmetric', 'asymmetric']
        if v.lower() not in supported_schemes:
            raise ValueError(f"Scheme {v} not supported. Supported: {supported_schemes}")
        return v.lower()


class ExperimentConfig(BaseModel):
    """Configuration for experiment parameters."""
    strategy: str = Field("iterative_sparsity", description="Experiment strategy")
    num_iterations: int = Field(1, description="Number of co-design iterations")
    output_dir: str = Field("./results/", description="Output directory")
    seed: int = Field(42, description="Random seed")
    save_intermediate: bool = Field(True, description="Save intermediate results")
    
    @validator('strategy')
    def validate_strategy(cls, v):
        supported_strategies = [
            'baseline', 'permute_only', 'sparsity_only', 'linear_sparsity',
            'iterative_sparsity', 'linear_quant_permute_first',
            'linear_quant_quant_first', 'iterative_quant'
        ]
        if v.lower() not in supported_strategies:
            raise ValueError(f"Strategy {v} not supported. Supported: {supported_strategies}")
        return v.lower()
    
    @validator('num_iterations')
    def validate_num_iterations(cls, v):
        if v <= 0:
            raise ValueError("Number of iterations must be positive")
        return v


class BenchmarkConfig(BaseModel):
    """Configuration for benchmarking."""
    warmup_runs: int = Field(10, description="Number of warmup runs")
    num_runs: int = Field(5, description="Number of benchmark runs")
    use_cuda_events: bool = Field(True, description="Use CUDA events for timing")
    cuda_sync: bool = Field(True, description="Synchronize CUDA before timing")
    
    @validator('warmup_runs')
    def validate_warmup_runs(cls, v):
        if v < 0:
            raise ValueError("Warmup runs must be non-negative")
        return v
    
    @validator('num_runs')
    def validate_num_runs(cls, v):
        if v <= 0:
            raise ValueError("Number of runs must be positive")
        return v


class ProfilingConfig(BaseModel):
    """Configuration for profiling."""
    enabled: bool = Field(False, description="Enable profiling")
    tool: str = Field("nsight_compute", description="Profiling tool")
    metrics: List[str] = Field(
        default=[
            "lts__t_sector_hit_rate.pct",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "sm__warps_active.avg.pct_of_peak_sustained_active"
        ],
        description="Metrics to collect"
    )
    
    @validator('tool')
    def validate_tool(cls, v):
        supported_tools = ['nsight_compute', 'pytorch_profiler']
        if v.lower() not in supported_tools:
            raise ValueError(f"Tool {v} not supported. Supported: {supported_tools}")
        return v.lower()


class HardwareConfig(BaseModel):
    """Configuration for hardware settings."""
    device: str = Field("cuda", description="Device type")
    gpu_id: int = Field(0, description="GPU device ID")
    mixed_precision: bool = Field(True, description="Use mixed precision")
    
    @validator('device')
    def validate_device(cls, v):
        supported_devices = ['cuda', 'cpu']
        if v.lower() not in supported_devices:
            raise ValueError(f"Device {v} not supported. Supported: {supported_devices}")
        return v.lower()


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field("INFO", description="Log level")
    file: str = Field("./logs/experiment.log", description="Log file path")
    console: bool = Field(True, description="Log to console")
    rich_formatting: bool = Field(True, description="Use rich formatting")
    
    @validator('level')
    def validate_level(cls, v):
        supported_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if v.upper() not in supported_levels:
            raise ValueError(f"Level {v} not supported. Supported: {supported_levels}")
        return v.upper()


class ReproducibilityConfig(BaseModel):
    """Configuration for reproducibility."""
    deterministic: bool = Field(True, description="Use deterministic algorithms")
    cuda_deterministic: bool = Field(True, description="CUDA deterministic mode")
    warn_non_deterministic: bool = Field(True, description="Warn about non-deterministic ops")


class Config(BaseModel):
    """Main configuration class."""
    model: ModelConfig
    dataset: DatasetConfig
    iasp: IASPConfig
    hds: HDSConfig
    ptq: PTQConfig
    experiment: ExperimentConfig
    benchmark: BenchmarkConfig
    profiling: ProfilingConfig
    hardware: HardwareConfig
    logging: LoggingConfig
    reproducibility: ReproducibilityConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Load configuration from dictionary."""
        return cls(**config_dict)
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict()
    
    def validate_layer_name(self, model_manager) -> None:
        """Validate that the specified layer name exists in the model."""
        if not model_manager.validate_layer_name(self.model.name, self.iasp.layer_name):
            available_layers = model_manager.get_model_layers(self.model.name)
            raise ValueError(
                f"Layer '{self.iasp.layer_name}' not found in model '{self.model.name}'. "
                f"Available layers: {available_layers[:10]}..."
                f"{'(showing first 10)' if len(available_layers) > 10 else ''}"
            )


def load_config(config_path: str) -> Config:
    """
    Load and validate configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated configuration object
    """
    try:
        return Config.from_yaml(config_path)
    except Exception as e:
        raise ValueError(f"Failed to load configuration from {config_path}: {e}")


def create_default_config() -> Config:
    """Create a default configuration."""
    return Config(
        model=ModelConfig(name="mamba-3b"),
        dataset=DatasetConfig(name="wikitext-103"),
        iasp=IASPConfig(layer_name="layers.0.mixer"),
        hds=HDSConfig(),
        ptq=PTQConfig(),
        experiment=ExperimentConfig(),
        benchmark=BenchmarkConfig(),
        profiling=ProfilingConfig(),
        hardware=HardwareConfig(),
        logging=LoggingConfig(),
        reproducibility=ReproducibilityConfig()
    )