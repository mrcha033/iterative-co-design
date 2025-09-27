# Production Deployment Guide

This guide explains how to deploy the iterative co-design pipeline in production environments using the complete Docker Compose stack with Triton Inference Server, monitoring, and load testing.

## Overview

The production deployment stack includes:
- **Triton Inference Server**: Model serving with GPU acceleration
- **Prometheus**: Metrics collection and time-series storage
- **Grafana**: Real-time monitoring dashboards
- **Load Testing**: HTTP-based benchmarking utilities

## Quick Deployment

### Prerequisites

```bash
# Required software
docker --version          # Docker 24.0+
docker-compose --version  # Docker Compose 2.0+
nvidia-docker --version   # NVIDIA Container Toolkit

# GPU verification
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### Launch Stack

```bash
# Navigate to deployment directory
cd deploy/triton

# Start all services
docker-compose up -d

# Verify services
curl http://localhost:8000/v2/health/ready  # Triton health check
curl http://localhost:9090/metrics          # Prometheus metrics
open http://localhost:3000                  # Grafana (admin/admin)
```

## Model Preparation

### Export ICD-Optimized Model

```bash
# 1. Generate optimized permutation
python -m icd.cli.main run -c configs/bert.json \
    --override pipeline.mode=iterative \
    --out runs/bert_production

# 2. Export Triton-compatible model
python scripts/export_triton_model.py \
    --input-dir runs/bert_production \
    --model-name bert_icd \
    --output-dir deploy/triton/models
```

### Model Repository Structure

```
deploy/triton/models/
├── bert_icd/
│   ├── config.pbtxt              # Triton model configuration
│   └── 1/                        # Version 1
│       ├── model.pt               # PyTorch model with permutations
│       ├── permutation.json       # ICD layout optimization
│       └── correlation_matrix.npz # Precomputed correlations
└── bert_baseline/
    ├── config.pbtxt
    └── 1/
        └── model.pt               # Original model for comparison
```

### Model Configuration

Example `config.pbtxt` for ICD-optimized models:

```protobuf
name: "bert_icd"
platform: "pytorch_libtorch"
max_batch_size: 32
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]
output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, 2 ]
  }
]
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
optimization {
  execution_accelerators {
    gpu_execution_accelerator: [{
      name: "tensorrt"
      parameters: {
        key: "precision_mode"
        value: "FP16"
      }
      parameters: {
        key: "max_workspace_size_bytes"
        value: "1073741824"
      }
    }]
  }
}
```

## Monitoring Setup

### Grafana Dashboard Configuration

The pre-configured dashboard (`deploy/triton/monitoring/grafana/dashboards/json/triton_overview.json`) includes:

- **Request Rate**: Inference requests per second
- **Latency Distribution**: P50, P95, P99 percentiles
- **GPU Utilization**: Per-device metrics
- **Memory Usage**: GPU and system memory
- **Queue Depth**: Request batching efficiency
- **Error Rate**: Failed inference attempts

### Custom Metrics for ICD

Add ICD-specific metrics to Triton models:

```python
# In your model's forward() method
import time
import tritonclient.http as httpclient

class ICDMetricsWrapper:
    def __init__(self, model):
        self.model = model
        self.permutation_time = 0
        self.correlation_lookups = 0

    def forward(self, *args, **kwargs):
        start_time = time.perf_counter()

        # Apply ICD permutation
        permuted_input = self.apply_permutation(args[0])
        self.permutation_time += time.perf_counter() - start_time

        # Model inference
        result = self.model(permuted_input, *args[1:], **kwargs)

        # Reverse permutation
        start_time = time.perf_counter()
        final_result = self.reverse_permutation(result)
        self.permutation_time += time.perf_counter() - start_time

        return final_result
```

### Prometheus Metrics Collection

Custom metrics are automatically exposed via Triton's metrics endpoint:

```
# HELP triton_icd_permutation_time_seconds Time spent on permutation operations
# TYPE triton_icd_permutation_time_seconds counter
triton_icd_permutation_time_seconds{model="bert_icd",version="1"} 0.003

# HELP triton_icd_correlation_cache_hits Correlation matrix cache hits
# TYPE triton_icd_correlation_cache_hits counter
triton_icd_correlation_cache_hits{model="bert_icd",version="1"} 1247

# HELP triton_icd_layout_efficiency Layout efficiency score
# TYPE triton_icd_layout_efficiency gauge
triton_icd_layout_efficiency{model="bert_icd",version="1"} 0.847
```

## Load Testing

### Basic Load Test

```bash
# Test single model performance
python scripts/production_benchmark.py \
    http://localhost:8000 bert_icd \
    --batch-size 1 \
    --duration 300 \
    --input-shape 128 \
    --report results/load_test_single.json

# Expected output:
# {
#   "requests": 1543,
#   "duration_s": 300.0,
#   "throughput_rps": 5.14,
#   "latency_ms": {
#     "mean": 194.2,
#     "p95": 287.6
#   }
# }
```

### A/B Testing Framework

Compare ICD-optimized vs baseline models:

```bash
# Deploy both models
cp -r deploy/triton/models/bert_icd deploy/triton/models/bert_baseline
# Edit bert_baseline/config.pbtxt to use original model

# Restart Triton to load both models
docker-compose restart triton

# Run A/B comparison
python scripts/ab_test_framework.py \
    --model-a bert_baseline \
    --model-b bert_icd \
    --duration 1800 \
    --traffic-split 50:50 \
    --report results/ab_test_results.json
```

### Production Traffic Simulation

```bash
# Realistic traffic patterns
python scripts/production_simulation.py \
    --endpoint http://localhost:8000 \
    --models bert_icd,bert_baseline \
    --traffic-pattern realistic \
    --duration 3600 \
    --report results/production_sim.json

# Traffic patterns available:
# - constant: Steady request rate
# - bursty: Periodic load spikes
# - realistic: Business hours simulation
# - stress: Maximum throughput test
```

## Performance Optimization

### Dynamic Batching Configuration

Optimize Triton's dynamic batching for ICD models:

```protobuf
# In config.pbtxt
dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 2000
  preserve_ordering: false
  priority_levels: 2
}
```

### GPU Memory Optimization

```bash
# Monitor GPU memory usage
nvidia-smi -l 1

# Adjust model instances based on memory
# In config.pbtxt:
instance_group [
  {
    count: 1                    # Reduce for large models
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

### Cache Configuration

Enable correlation matrix caching for faster startup:

```python
# Environment variables for Triton
TRITON_ICD_CACHE_SIZE=1024MB
TRITON_ICD_CACHE_TTL=3600
TRITON_ICD_PRELOAD_CORRELATIONS=true
```

## Deployment Scenarios

### Canary Deployment

Deploy ICD optimization to a subset of traffic:

```bash
# 1. Deploy baseline model
docker-compose up -d

# 2. Add ICD model as canary
python scripts/deploy_canary.py \
    --model bert_icd \
    --traffic-percentage 10 \
    --monitor-duration 3600

# 3. Monitor metrics
python scripts/monitor_canary.py \
    --baseline bert_baseline \
    --canary bert_icd \
    --alert-threshold 5.0  # % latency increase
```

### Blue-Green Deployment

Zero-downtime deployment of ICD optimizations:

```bash
# 1. Deploy green environment
docker-compose -f docker-compose.green.yml up -d

# 2. Validate green environment
python scripts/validate_deployment.py \
    --endpoint http://localhost:8001 \
    --model bert_icd \
    --test-duration 300

# 3. Switch traffic
python scripts/switch_traffic.py \
    --from blue --to green \
    --validation-time 600
```

### Multi-GPU Scaling

Scale ICD deployment across multiple GPUs:

```yaml
# docker-compose.scale.yml
version: "3.9"
services:
  triton-gpu0:
    image: nvcr.io/nvidia/tritonserver:23.10-py3
    environment:
      - CUDA_VISIBLE_DEVICES=0
    runtime: nvidia
    ports:
      - "8000:8000"

  triton-gpu1:
    image: nvcr.io/nvidia/tritonserver:23.10-py3
    environment:
      - CUDA_VISIBLE_DEVICES=1
    runtime: nvidia
    ports:
      - "8001:8000"

  load-balancer:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

## Production Monitoring

### Key Performance Indicators

Monitor these metrics for production ICD deployments:

| Metric | Threshold | Action |
|--------|-----------|--------|
| **P95 Latency** | <200ms | Scale up if exceeded |
| **Error Rate** | <0.1% | Investigate errors |
| **GPU Utilization** | 70-90% | Optimal range |
| **Queue Depth** | <10 | Check batching config |
| **Memory Usage** | <90% | Add GPU instances |
| **Cache Hit Rate** | >95% | Validate correlation cache |

### Alerting Rules

Prometheus alerting rules for ICD deployments:

```yaml
# prometheus_rules.yml
groups:
  - name: icd_alerts
    rules:
      - alert: ICDHighLatency
        expr: triton_model_inference_duration_us{quantile="0.95"} > 200000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "ICD model latency exceeding threshold"

      - alert: ICDLowCacheHitRate
        expr: triton_icd_correlation_cache_hits / triton_icd_correlation_total < 0.95
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "ICD correlation cache hit rate below 95%"
```

### Log Analysis

Analyze ICD-specific logs:

```bash
# Extract ICD performance logs
docker logs triton_triton_1 2>&1 | grep "ICD" > icd_performance.log

# Analyze permutation efficiency
python scripts/analyze_icd_logs.py \
    --log-file icd_performance.log \
    --metric permutation_time \
    --report icd_analysis.html
```

## Troubleshooting

### Common Deployment Issues

**Issue**: Triton fails to load ICD model
```bash
# Check model repository
ls -la deploy/triton/models/bert_icd/1/
# Verify permutation.json exists and is valid
python -c "import json; json.load(open('deploy/triton/models/bert_icd/1/permutation.json'))"
```

**Issue**: Poor performance compared to baseline
```bash
# Check GPU memory fragmentation
nvidia-smi
# Restart Triton to clear memory
docker-compose restart triton

# Verify permutation is being applied
curl http://localhost:8000/v2/models/bert_icd/stats
```

**Issue**: High memory usage
```bash
# Reduce correlation matrix cache size
export TRITON_ICD_CACHE_SIZE=512MB
docker-compose restart triton

# Monitor memory usage
watch -n 1 nvidia-smi
```

### Performance Debugging

```bash
# Enable detailed profiling
export TRITON_LOG_VERBOSE=1
export TRITON_ICD_PROFILE=true
docker-compose restart triton

# Collect performance traces
python scripts/collect_performance_trace.py \
    --endpoint http://localhost:8000 \
    --model bert_icd \
    --duration 300 \
    --output traces/

# Analyze bottlenecks
python scripts/analyze_performance_trace.py traces/
```

## Security Considerations

### Model Security

```bash
# Verify model checksums
sha256sum deploy/triton/models/bert_icd/1/model.pt
# Compare against known good checksum

# Encrypt sensitive permutation data
python scripts/encrypt_permutation.py \
    --input deploy/triton/models/bert_icd/1/permutation.json \
    --key-file production.key
```

### Network Security

```yaml
# docker-compose.secure.yml
version: "3.9"
services:
  triton:
    image: nvcr.io/nvidia/tritonserver:23.10-py3
    networks:
      - internal
    # Remove external port exposure

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./ssl:/etc/nginx/ssl:ro
      - ./nginx-ssl.conf:/etc/nginx/nginx.conf:ro
    networks:
      - internal
      - external

networks:
  internal:
    internal: true
  external:
```

## Cost Optimization

### Resource Optimization

```bash
# Optimize instance counts based on load
python scripts/optimize_instances.py \
    --target-latency 100 \
    --target-utilization 80 \
    --cost-model aws-p4d \
    --report optimization_report.json
```

### Correlation Matrix Optimization

```bash
# Compress correlation matrices for storage
python scripts/compress_correlations.py \
    --input-dir deploy/triton/models/ \
    --compression-ratio 0.1 \
    --validate-accuracy
```

This production deployment guide enables real-world deployment of ICD optimizations with comprehensive monitoring, testing, and optimization capabilities suitable for enterprise environments.