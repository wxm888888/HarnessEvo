from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from harness_evolution.harnesses.opencode import build


def test_commit_recipe_registry_keeps_first_nondeterministic_build(tmp_path):
    config = SimpleNamespace(
        build={"skip_embed_web_ui": True},
        workspace_root=tmp_path / "workspace",
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    dist = tmp_path / "dist" / "opencode"
    dist.parent.mkdir()
    dist.write_bytes(b"first-build-with-timestamp-a")

    with mock.patch.object(build, "_git_head_commit", return_value="a" * 40):
        registered = build.register_built_binary(config, repo, dist)
        dist.write_bytes(b"second-build-with-timestamp-b")
        same_registered = build.register_built_binary(config, repo, dist)

    assert same_registered == registered
    assert registered.read_bytes() == b"first-build-with-timestamp-a"
