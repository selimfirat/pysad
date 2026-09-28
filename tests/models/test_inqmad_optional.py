"""Regression tests for optional JAX import of Inqmad (#174)."""
import builtins
import importlib
import sys

import pytest


def _reload_inqmad_without_jax(monkeypatch):
    """Re-import pysad.models.inqmad with jax imports blocked."""
    for name in list(sys.modules):
        if name == "jax" or name.startswith("jax.") or name.endswith(".inqmad") or name == "pysad.models.inqmad":
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jax" or name.startswith("jax."):
            raise ImportError("simulated missing jax")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return importlib.import_module("pysad.models.inqmad")


def test_inqmad_module_imports_without_jax(monkeypatch):
    mod = _reload_inqmad_without_jax(monkeypatch)
    assert mod.JAX_AVAILABLE is False
    assert hasattr(mod, "Inqmad")


def test_inqmad_init_raises_import_error_without_jax(monkeypatch):
    mod = _reload_inqmad_without_jax(monkeypatch)
    with pytest.raises(ImportError, match=r"pysad\[inqmad\]"):
        mod.Inqmad(input_shape=2, dim_x=4, gamma=1.0)
