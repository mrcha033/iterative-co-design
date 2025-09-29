"""Dataset-based quality evaluation helpers."""

from __future__ import annotations

import math
import types
from typing import Iterable, Optional

import numpy as np
from datasets import load_dataset

try:  # pragma: no cover - torch optional in minimal installs
    import torch  # type: ignore[import-not-found]
except Exception as _torch_exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Exception | None = _torch_exc

    def _no_grad():  # type: ignore[misc]
        def decorator(fn):
            return fn

        return decorator

    def _require_torch(op: str) -> None:
        message = (
            f"{op} requires PyTorch. Install the 'experiments' extra (pip install 'repermute[experiments]') "
            "to enable quality evaluation helpers."
        )
        raise RuntimeError(message) from _TORCH_IMPORT_ERROR

else:
    _TORCH_IMPORT_ERROR = None

    def _no_grad():  # pragma: no cover - thin wrapper to please the type checker
        return torch.no_grad()

    def _require_torch(op: str) -> None:  # pragma: no cover - torch already available
        return None

__all__ = ["eval_wt103_ppl", "eval_sst2"]


def _raise_evaluate_placeholder(*args, **kwargs):  # pragma: no cover - replaced
    raise RuntimeError(
        "evaluate shim placeholder – monkeypatch or real import is required"
    )


evaluate = types.SimpleNamespace(load=_raise_evaluate_placeholder)


@_no_grad()
def eval_wt103_ppl(
    model,
    tokenizer,
    *,
    max_length: int = 1024,
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> float:
    """Evaluate WikiText-103 perplexity.

    Parameters
    ----------
    model: Causal LM with ``labels`` support.
    tokenizer: Tokenizer providing ``__call__`` returning ``input_ids``.
    max_length: Sequence truncation length.
    split: Dataset split name.
    max_samples: Optional cap on number of examples for faster smoke tests.
    """

    _require_torch("eval_wt103_ppl")
    dataset = load_dataset("wikitext", "wikitext-103-v1")[split]
    model.eval()

    total_ce = 0.0
    total_tokens = 0
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu") if torch is not None else "cpu"

    count = 0
    for example in dataset:  # type: ignore[assignment]
        if max_samples is not None and count >= max_samples:
            break
        count += 1
        encoded = tokenizer(
            example["text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss.detach().float().item()
        total_ce += loss * (input_ids.numel() - 1)
        total_tokens += int(input_ids.numel() - 1)

    if total_tokens == 0:
        return float("nan")
    ppl = math.exp(total_ce / total_tokens)
    return float(ppl)


@_no_grad()
def eval_sst2(
    model,
    tokenizer,
    *,
    batch_size: int = 64,
    max_length: int = 128,
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> dict[str, float]:
    """Evaluate SST-2 accuracy and F1."""

    _require_torch("eval_sst2")
    dataset = load_dataset("glue", "sst2")[split]

    global evaluate
    try:
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
    except Exception:
        import evaluate as _evaluate

        globals()["evaluate"] = _evaluate
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu") if torch is not None else "cpu"

    processed = 0
    for start in range(0, len(dataset), batch_size):
        if max_samples is not None and processed >= max_samples:
            break
        batch = dataset[start : start + batch_size]
        if isinstance(batch, dict):
            sentences = batch["sentence"]
            labels = batch["label"]
            processed += len(sentences)
        else:
            sentences = [example["sentence"] for example in batch]
            labels = [example["label"] for example in batch]
            processed += len(sentences)
        encoded = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded).logits.detach().cpu()
        preds = torch.argmax(logits, dim=-1).numpy()
        refs = np.asarray(labels)
        accuracy_metric.add_batch(predictions=preds, references=refs)
        f1_metric.add_batch(predictions=preds, references=refs)

    return {
        "accuracy": float(accuracy_metric.compute().get("accuracy", 0.0)),
        "f1": float(f1_metric.compute().get("f1", 0.0)),
    }

# -----------------------------
# Back-compat quality shortcuts
# -----------------------------
def eval_acc(model, tokenizer=None, **kwargs):
    """
    Back-compat: alias for eval_sst2(model, tokenizer, **kwargs).
    Returns a dict that includes "accuracy".
    """
    if tokenizer is None:
        return None
    return eval_sst2(model, tokenizer, **kwargs)

def eval_ppl(model, tokenizer=None, **kwargs):
    """
    Back-compat: alias for eval_wt103_ppl(model, tokenizer, **kwargs).
    Returns a dict that includes "perplexity".
    """
    if tokenizer is None:
        return None
    return eval_wt103_ppl(model, tokenizer, **kwargs)

try:
    __all__
except NameError:
    __all__ = []
for _name in ("eval_acc", "eval_ppl"):
    if _name not in __all__:
        __all__.append(_name)
