from __future__ import annotations

import os
from pathlib import Path

from harness_evolution.harnesses.opencode.build import _settle_native_optional_dependencies


def test_settle_native_optional_dependencies_no_matches_is_a_noop(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "node_modules").mkdir(parents=True)
    _settle_native_optional_dependencies(workspace)  # must not raise


def test_settle_native_optional_dependencies_resolves_existing_symlink(tmp_path):
    workspace = tmp_path / "ws"
    real = workspace / "node_modules" / ".bun" / "native-preview-linux-x64@1" / "node_modules" / "@typescript" / "native-preview-linux-x64"
    (real / "lib").mkdir(parents=True)
    link_dir = workspace / "node_modules" / "@typescript"
    link_dir.mkdir(parents=True)
    os.symlink(os.path.relpath(real, link_dir), link_dir / "native-preview-linux-x64")
    _settle_native_optional_dependencies(workspace)  # must not raise, resolves immediately


def test_settle_native_optional_dependencies_gives_up_on_broken_symlink(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    link_dir = workspace / "node_modules" / "@typescript"
    link_dir.mkdir(parents=True)
    os.symlink("/does/not/exist", link_dir / "native-preview-linux-x64")
    import harness_evolution.harnesses.opencode.build as build_mod

    monkeypatch.setattr(build_mod.time, "sleep", lambda _: None)
    _settle_native_optional_dependencies(workspace)  # must not raise even when it never resolves


def test_settle_native_optional_dependencies_waits_for_delayed_target(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    scope = workspace / "node_modules" / "@typescript"
    scope.mkdir(parents=True)
    target = workspace / "node_modules" / ".bun" / "delayed-native"
    (scope / "native-preview-linux-x64").symlink_to(target)
    waits = []

    def expose_dependency(_seconds):
        waits.append(True)
        (target / "lib").mkdir(parents=True)

    import harness_evolution.harnesses.opencode.build as build_mod
    monkeypatch.setattr(build_mod.time, "sleep", expose_dependency)
    _settle_native_optional_dependencies(workspace)
    assert waits == [True]
    assert (scope / "native-preview-linux-x64" / "lib").is_dir()
