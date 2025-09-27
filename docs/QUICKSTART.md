# Iterative Co-Design Quickstart

This quickstart walks through reproducing a latency improvement on a toy
Mamba layer in under five minutes using CPU-only resources.

## 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## 2. Generate synthetic correlation data

```bash
python - <<'PY'
import torch
from icd.graph.streaming_correlation import compute_streaming_correlation
from icd.graph import CorrelationConfig

rng = torch.Generator().manual_seed(7)
samples = [torch.randn(64, 32, generator=rng) for _ in range(8)]
cfg = CorrelationConfig(threshold=0.05, normalize="sym")
csr, meta = compute_streaming_correlation(samples, feature_dim=32, cfg=cfg)
print("Correlation nnz:", csr.nnz())
print("Meta:", meta)
PY
```

## 3. Fit a permutation

```bash
python - <<'PY'
from icd.core.cost import CostConfig, eval_cost
from icd.core.solver import fit_permutation
from icd.core.graph import CSRMatrix
from icd.graph.streaming_correlation import compute_streaming_correlation
from icd.graph import CorrelationConfig
import torch

rng = torch.Generator().manual_seed(13)
samples = [torch.randn(64, 32, generator=rng) for _ in range(6)]
cfg = CorrelationConfig(threshold=0.05, normalize="sym")
csr, _ = compute_streaming_correlation(samples, feature_dim=32, cfg=cfg)
pi, stats = fit_permutation(csr, time_budget_s=0.1, refine_steps=64)
identity = list(range(csr.shape[0]))
cfg_cost = CostConfig()
baseline = eval_cost(csr, identity, identity, cfg_cost)
improved = eval_cost(csr, pi, pi, cfg_cost)
print("Permutation:", pi[:10], "...")
print("Cost delta:", baseline["J"] - improved["J"])
print("Stats:", stats)
PY
```

## 4. Next steps

- Replace the synthetic samples with captures from your own model using
  `icd.graph.collect_correlations`.
- Export the permutation via the CLI: `icd export --config configs/mock.json`.
- Validate determinism with `python scripts/check_cuda_env.py`.
