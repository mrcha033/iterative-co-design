# IASP Configuration Defaults

The iterative correlation pipeline ships with `configs/iasp_defaults.yaml`. The file defines:

- **Correlation**: 16 activation samples captured from `encoder.layer.0.output` with whitening and deterministic seeding (`correlation.whiten: true`).
- **Transfer batching**: Activation tensors are chunked in groups of four to avoid host OOM when staging GPU captures (`correlation.transfer_batch_size: 4`).

```yaml
correlation:
  samples: 16
  layers:
    - encoder.layer.0.output
  whiten: true
  transfer_batch_size: 4
  seed: 123
```
- **Clustering**: Louvain is the primary algorithm. If the measured modularity falls below `0.35` or the solver exceeds `0.5` seconds, the pipeline retries with spectral clustering.

Overrides can be passed through the runtime configuration under `solver.correlation` and `solver.clustering`.
