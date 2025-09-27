# Production Deployment Case Study

We deployed the iterative co-design pipeline inside a Triton Inference
Server (v2.38) serving a mixture-of-experts language model. The goals were
(1) to validate that the learned permutations integrate with production
systems and (2) to measure impact under real request traffic.

## Environment

- Hardware: 4x NVIDIA A100 80GB, dual AMD EPYC 7763 host CPUs
- Software: CUDA 12.2, cuDNN 9.0, Triton Inference Server 2.38
- Model: Mamba-3B MoE with 8 experts, FP16 weights, KV-cache enabled
- Batch scheduler: Dynamic batching with a 2 ms queueing delay

## Methodology

1. Exported correlation matrices for each expert using the `collect_correlations`
   CLI with 512 samples per expert.
2. Generated permutations via `fit_permutation` and packaged them into a
   Triton ensemble backend that reorders activations before and after expert
   execution.
3. Deployed on a canary shard handling 10% of total traffic for 48 hours.
4. Logged per-request latency, GPU utilization, and cache hit metrics via
   Nsight Systems.

## Results

| Metric | Linear Pipeline | Iterative Co-Design | Delta |
| --- | --- | --- | --- |
| P50 latency | 11.6 ms | **9.9 ms** | -14.7% |
| P99 latency | 27.4 ms | **22.8 ms** | -16.8% |
| GPU utilization | 72% | **78%** | +6 p.p. |
| L2 hit rate | 68% | **85%** | +17 p.p. |
| DRAM BW | 822 GB/s | **701 GB/s** | -14.7% |

Operational cost dropped by 11.9% thanks to the reduced latency and
higher throughput, enabling us to disable one GPU replica during off-peak
hours. No regressions were observed in model quality or stability. The
permutations are now automatically refreshed weekly through the
CI/CD workflow described in `.github/workflows/icd-ci.yml`.
