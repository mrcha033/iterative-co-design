from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

from icd.utils import imports as imports_mod


def test_candidate_module_paths_and_loader(tmp_path, monkeypatch):
    module_path = tmp_path / "dummy_mod.py"
    module_path.write_text("value = 42\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    candidates = list(imports_mod._candidate_module_paths("dummy_mod"))
    assert module_path in candidates

    loaded = imports_mod._load_module_from_path("dummy_mod")
    assert isinstance(loaded, ModuleType)
    assert getattr(loaded, "value") == 42

    obj = imports_mod.load_object("dummy_mod.value")
    assert obj == 42


def test_load_object_errors(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        imports_mod.load_object("")

    module_path = tmp_path / "modpkg"
    module_path.mkdir()
    (module_path / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError):
        imports_mod.load_object("modpkg")
