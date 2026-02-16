from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime_editing_fast.editor import FastFlux2Config, FastFlux2RealtimeEditor


class _DummyPipe:
    class _DummyTransformer:
        def set_attention_backend(self, backend: str) -> None:
            _ = backend

    def __init__(self) -> None:
        self.transformer = self._DummyTransformer()


def _new_editor() -> FastFlux2RealtimeEditor:
    cfg = FastFlux2Config(verbose=False)
    return FastFlux2RealtimeEditor(cfg)


def test_auto_select_prefers_fa3_over_sage_and_native(monkeypatch) -> None:
    editor = _new_editor()
    pipe = _DummyPipe()
    called: list[str] = []

    monkeypatch.setattr(editor, "_maybe_patch_diffusers_fa3_dispatch", lambda: (True, "patched"))
    monkeypatch.setattr(editor, "_probe_fa3_interface_compatibility", lambda runtime_device, dtype: (True, "ok"))

    def _fake_try_set(_pipe, backend: str):
        called.append(backend)
        return True, "ok"

    monkeypatch.setattr(editor, "_try_set_attention_backend", _fake_try_set)

    selected = editor._auto_select_attention_backend(pipe, runtime_device="cuda:0", dtype=torch.bfloat16)
    assert selected == "_flash_3"
    assert called == ["_flash_3"]


def test_auto_select_falls_back_to_sage_when_fa3_unavailable(monkeypatch) -> None:
    editor = _new_editor()
    pipe = _DummyPipe()
    called: list[str] = []

    monkeypatch.setattr(editor, "_maybe_patch_diffusers_fa3_dispatch", lambda: (True, "patched"))
    monkeypatch.setattr(editor, "_probe_fa3_interface_compatibility", lambda runtime_device, dtype: (False, "nope"))

    def _fake_try_set(_pipe, backend: str):
        called.append(backend)
        return (backend == "sage"), "ok" if backend == "sage" else "fail"

    monkeypatch.setattr(editor, "_try_set_attention_backend", _fake_try_set)

    selected = editor._auto_select_attention_backend(pipe, runtime_device="cuda:0", dtype=torch.bfloat16)
    assert selected == "sage"
    assert called == ["sage"]


def test_auto_select_falls_back_to_native_when_fa3_and_sage_fail(monkeypatch) -> None:
    editor = _new_editor()
    pipe = _DummyPipe()
    called: list[str] = []

    monkeypatch.setattr(editor, "_maybe_patch_diffusers_fa3_dispatch", lambda: (True, "patched"))
    monkeypatch.setattr(editor, "_probe_fa3_interface_compatibility", lambda runtime_device, dtype: (False, "nope"))

    def _fake_try_set(_pipe, backend: str):
        called.append(backend)
        return (backend == "native"), "ok" if backend == "native" else "fail"

    monkeypatch.setattr(editor, "_try_set_attention_backend", _fake_try_set)

    selected = editor._auto_select_attention_backend(pipe, runtime_device="cuda:0", dtype=torch.bfloat16)
    assert selected == "native"
    assert called == ["sage", "native"]
