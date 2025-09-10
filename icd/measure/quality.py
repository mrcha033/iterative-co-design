from __future__ import annotations

from typing import Optional


def eval_ppl(model_id: str, dataset: str = "wikitext-103-raw-v1", max_samples: int = 256, max_len: int = 512) -> Optional[float]:
    """CI-safe stub. Return None when unavailable.
    Real implementation can be added for local/GPU runs.
    """
    return None


def eval_acc(model_id: str, dataset: str = "glue/sst2", max_samples: int = 512, max_len: int = 128) -> Optional[float]:
    """CI-safe stub. Return None when unavailable.
    Real implementation can be added for local/GPU runs.
    """
    return None


__all__ = ["eval_ppl", "eval_acc"]

