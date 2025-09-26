from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from icd.measure.quality import eval_sst2, eval_wt103_ppl


class DummyTokenizer:
    def __call__(self, texts, return_tensors="pt", max_length=16, **_: object):
        if isinstance(texts, (list, tuple)):
            batch = len(texts)
        else:
            texts = [texts]
            batch = 1
        tokens = torch.full((batch, min(max_length, 8)), 1, dtype=torch.long)
        return {"input_ids": tokens}


class DummyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 2, bias=False)

    def forward(self, input_ids):
        logits = torch.zeros(input_ids.shape[0], 2)
        return SimpleNamespace(logits=logits)


class DummyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, labels):
        return SimpleNamespace(loss=torch.tensor(0.5))


def _patch_load_dataset(monkeypatch, data_map):
    def _load_dataset(*args, **kwargs):  # pragma: no cover - simple shim
        return data_map

    monkeypatch.setattr("icd.measure.quality.load_dataset", _load_dataset)


def _patch_evaluate(monkeypatch):
    class DummyMetric:
        def __init__(self, name):
            self.name = name
            self.preds = []
            self.refs = []

        def add_batch(self, predictions, references):
            self.preds.extend(list(predictions))
            self.refs.extend(list(references))

        def compute(self):
            if not self.refs:
                return {self.name: 0.0}
            if self.name == "accuracy":
                correct = sum(int(p == r) for p, r in zip(self.preds, self.refs))
                return {"accuracy": correct / len(self.refs)}
            if self.name == "f1":
                return {"f1": 1.0}
            return {self.name: 0.0}

    def _load(name):  # pragma: no cover - shim
        return DummyMetric(name)

    monkeypatch.setattr("icd.measure.quality.evaluate.load", _load)


def test_eval_sst2_stub(monkeypatch):
    dataset = {
        "validation": [
            {"sentence": "good", "label": 1},
            {"sentence": "bad", "label": 0},
        ]
    }
    _patch_load_dataset(monkeypatch, dataset)
    _patch_evaluate(monkeypatch)
    model = DummyClassifier()
    tokenizer = DummyTokenizer()
    result = eval_sst2(model, tokenizer, batch_size=2, max_samples=2)
    assert "accuracy" in result
    assert "f1" in result


def test_eval_wt103_stub(monkeypatch):
    dataset = {"validation": [{"text": "hello world"}, {"text": "another sample"}]}
    _patch_load_dataset(monkeypatch, dataset)
    model = DummyLM()
    tokenizer = DummyTokenizer()
    ppl = eval_wt103_ppl(model, tokenizer, max_samples=2)
    assert ppl > 0
