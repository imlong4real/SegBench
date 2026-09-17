"""Tests for the thin Cirro-only Segger launcher."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[1] / "workflows" / "cirro" / "bin"
          / "seeded_segger_cli.py")


def load_launcher():
    spec = importlib.util.spec_from_file_location("seeded_segger_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_force_num_workers_overrides_hidden_upstream_default(monkeypatch):
    class FakeDataModule:
        def __init__(self, *, num_workers=8):
            self.num_workers = num_workers

    segger = types.ModuleType("segger")
    data = types.ModuleType("segger.data")
    data.ISTDataModule = FakeDataModule
    segger.data = data
    monkeypatch.setitem(sys.modules, "segger", segger)
    monkeypatch.setitem(sys.modules, "segger.data", data)

    launcher = load_launcher()
    launcher.force_num_workers(0)

    assert FakeDataModule().num_workers == 0
    assert FakeDataModule(num_workers=8).num_workers == 0


def test_force_num_workers_rejects_negative_values():
    launcher = load_launcher()
    try:
        launcher.force_num_workers(-1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative worker count was accepted")
