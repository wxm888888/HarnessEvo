from pathlib import Path
import importlib
import types

import pytest

from harness_dist import afs
from harness_evolution.evaluation.backends.opensandbox_settings import publish_side_binaries
from terminal_bench_opensandbox import afs_publish


def test_real_publisher_preserves_custom_destination(tmp_path, monkeypatch):
    source = tmp_path / "opencode"
    source.write_bytes(b"binary")
    destination = "/sandbox-data/duchenyu/opencode/wxh/test-sample/opencode"
    captured = []
    class Runtime:
        pass
    monkeypatch.setattr(afs_publish, "_runtime", lambda *args: Runtime())
    async def publish(runtime, binary, target, log):
        captured.append((binary, target))
        return target
    monkeypatch.setattr(afs_publish, "_publish", publish)
    monkeypatch.setenv("HARNESS_AFS_PUBLISH", "1")
    monkeypatch.setenv("HARNESS_AFS_PUBLISH_BACKEND", "opensandbox")
    publish_side_binaries([("sample", "a" * 40, str(source), destination)], env_file="credentials.env")
    assert captured == [(source, destination)]


def test_default_publisher_keeps_commit_layout(tmp_path, monkeypatch):
    source = tmp_path / "opencode"
    source.write_bytes(b"binary")
    captured = []
    monkeypatch.setattr(afs_publish, "_runtime", lambda *args: object())
    async def publish(runtime, binary, target, log):
        captured.append(target)
        return target
    monkeypatch.setattr(afs_publish, "_publish", publish)
    monkeypatch.setenv("HARNESS_AFS_PUBLISH_BACKEND", "opensandbox")
    afs.publish_binary(source, "a" * 40, env_file="credentials.env", afs_root="/sandbox-data/test")
    assert captured == ["/sandbox-data/test/aaaaaaa/opencode"]


def test_custom_path_rejects_traversal(tmp_path, monkeypatch):
    source = tmp_path / "opencode"
    source.write_bytes(b"binary")
    monkeypatch.setenv("HARNESS_AFS_PUBLISH_BACKEND", "opensandbox")
    with pytest.raises(ValueError):
        afs.publish_binary(source, "a" * 40, afs_path="/sandbox-data/test/../opencode")
