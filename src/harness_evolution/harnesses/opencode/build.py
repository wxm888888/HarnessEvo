"""Build OpenCode binaries from a source checkout and keep a commit-keyed binary registry."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from harness_evolution.core.types import RuntimeConfig
from harness_evolution.core.config import _bool_value, _repo_relative_path


def _current_binary_name() -> str:
    os_name = platform.system().lower()
    if os_name == "darwin":
        os_part = "darwin"
    elif os_name == "linux":
        os_part = "linux"
    elif os_name.startswith("win"):
        os_part = "windows"
    else:
        os_part = os_name
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        arch_part = "x64"
    elif machine in {"aarch64", "arm64"}:
        arch_part = "arm64"
    else:
        arch_part = machine
    suffix = ".exe" if os_part == "windows" else ""
    return f"opencode-{os_part}-{arch_part}/bin/opencode{suffix}"


def _configured_bun_bin(config: RuntimeConfig) -> Path | None:
    """Return the configured Bun executable, if one was supplied.

    ``BUN_BIN`` used to be passed only to the OpenCode launcher.  Static
    validation runs directly from Python, so it must receive the same
    executable through ``PATH`` as well.
    """
    evolver = config.opencode_evolver or {}
    client_init = evolver.get("client_init") if isinstance(evolver, dict) else None
    client_env = client_init.get("env") if isinstance(client_init, dict) else None
    dev_run_env = client_init.get("dev_run_env") if isinstance(client_init, dict) else None
    raw = client_env.get("BUN_BIN") if isinstance(client_env, dict) else None
    if not raw:
        raw = dev_run_env.get("BUN_BIN") if isinstance(dev_run_env, dict) else None
    if not raw:
        raw = evolver.get("bun_bin") if isinstance(evolver, dict) else None
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    return path if path.is_file() else None


def bun_command_env(env: dict[str, str], bun_bin: Path | None = None) -> dict[str, str]:
    """Prepare the same JS runtime for agent tools and controller checks.

    tsgo's JS launcher catches a missing import.meta.resolve API on Node 16
    and reports the installed native package as missing. Use Bun as node only
    when the inherited Node cannot run that launcher. Child shells inherit
    this PATH, including Bun's additional node_modules/.bin entries.
    """
    env = dict(env)
    if bun_bin is None and env.get("BUN_BIN"):
        configured = Path(env["BUN_BIN"]).expanduser()
        if configured.is_file():
            bun_bin = configured
    if bun_bin is None:
        discovered_bun = shutil.which("bun", path=env.get("PATH"))
        bun_bin = Path(discovered_bun) if discovered_bun else None
    if bun_bin is not None:
        env["BUN_BIN"] = str(bun_bin)
        env["PATH"] = f"{bun_bin.parent}{os.pathsep}{env.get('PATH', '')}"
        # Some CPU rjob images still provide Node 16.  The native TypeScript
        # preview launcher uses ``import.meta.resolve`` and misleadingly
        # reports its optional linux package as missing when run by that old
        # Node, even though the package and tsgo binary are present.  Bun can
        # execute the same launcher correctly.  Add a process-local node shim
        # only when the installed Node lacks the required API.
        node_bin = shutil.which("node", path=env.get("PATH"))
        node_compatible = False
        if node_bin:
            probe = subprocess.run(
                [
                    node_bin,
                    "--input-type=module",
                    "-e",
                    "if (typeof import.meta.resolve !== 'function') process.exit(1)",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                check=False,
            )
            node_compatible = probe.returncode == 0
        if not node_compatible:
            shim_dir = Path(env.get("TMPDIR") or "/tmp") / "harnessrl-bun-node-shim"
            shim_dir.mkdir(parents=True, exist_ok=True)
            shim = shim_dir / "node"
            if shim.is_symlink() and shim.resolve(strict=False) != bun_bin.resolve():
                shim.unlink()
            if not shim.exists():
                try:
                    shim.symlink_to(bun_bin.resolve())
                except FileExistsError:
                    pass
            env["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"
    return env



def _command_env(config: RuntimeConfig, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = bun_command_env(os.environ.copy(), _configured_bun_bin(config))
    if extra:
        env.update(extra)
    return env



def ensure_bun_install(opencode_repo_path: Path, env: dict[str, str] | None = None) -> None:
    marker = opencode_repo_path / "node_modules" / ".opencode_bun_install.ok"
    if marker.exists():
        return
    if (opencode_repo_path / "node_modules").exists():
        marker.write_text(json.dumps({"note": "existing node_modules reused"}) + "\n", encoding="utf-8")
        return
    result = subprocess.run(
        ["bun", "install"],
        cwd=str(opencode_repo_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**(env or os.environ), "HUSKY": "0"},
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"bun install failed in {opencode_repo_path}\n{result.stdout}")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"installed_at": datetime.now(timezone.utc).isoformat()}) + "\n", encoding="utf-8")


def _workspace_member_links(
    source_node_modules_dir: Path, source_repo: Path
) -> list[tuple[Path, Path]]:
    """Return (link path, package path) pairs relative to their two roots."""
    packages_root = (source_repo / "packages").resolve()
    links: list[tuple[Path, Path]] = []
    for dirpath, dirnames, _files in os.walk(source_node_modules_dir, followlinks=False):
        for name in list(dirnames):
            entry = Path(dirpath) / name
            if not entry.is_symlink():
                continue
            dirnames.remove(name)
            try:
                package_relative = entry.resolve().relative_to(packages_root)
            except ValueError:
                continue
            links.append((entry.relative_to(source_node_modules_dir), package_relative))
    return links


def _relink_workspace_members(
    node_modules_dir: Path,
    workspace: Path,
    workspace_member_links: list[tuple[Path, Path]],
) -> None:
    """Repoint cloned workspace-member links at this candidate worktree.

    Those are bun's per-workspace-member symlinks (e.g.
    node_modules/@opencode-ai/core -> ../../../core), possibly nested
    arbitrarily deep -- not just the shallow node_modules/@scope/pkg case, but
    also third-party packages that themselves depend on a workspace package
    (e.g. opencode-gitlab-auth depends on @opencode-ai/plugin) via bun's
    isolated-install `.bun/<pkg>@<version>/node_modules/...` store, and even
    *published .d.ts files* physically inside such a package that do their
    own directory-walking node_modules lookup from their real (baseline) path.
    `cp -al` preserves symlinks as symlinks (only regular files get
    hardlinked), so any such link -- at any depth -- is still exactly the one
    baseline had, and still carries the *relative* target that resolves back
    to source_repo/packages/<name> according to wherever it physically lives.
    Discovering links recursively from the image-local source tree remains
    depth- and scheme-agnostic, but avoids walking tens of thousands of GPFS
    entries once per candidate after cloning. That GPFS walk added several
    minutes to a two-candidate standalone smoke.
    """
    for link_relative, package_relative in workspace_member_links:
        entry = node_modules_dir / link_relative
        if not entry.is_symlink():
            raise RuntimeError(f"expected cloned workspace-member symlink is missing: {entry}")
        entry.unlink()
        entry.symlink_to(
            workspace / "packages" / package_relative,
            target_is_directory=True,
        )


_NODE_MODULES_CACHE_LOCK = threading.Lock()


def _copy_node_modules_tree(source: Path, target: Path) -> None:
    """Copy a dependency tree across filesystems, preferring CoW clones."""
    result = subprocess.run(
        [
            "cp",
            "-a",
            "--reflink=auto",
            "--no-preserve=ownership,xattr",
            str(source),
            str(target),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to copy dependency tree {source} -> {target}:\n{result.stdout.strip()}"
        )


def _remove_partial_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def link_shared_node_modules(workspace: Path, source_repo: Path) -> list[str]:
    """Point a fresh worktree at an already-installed node_modules.

    Worktrees are created empty, so validation (tsgo) and the build would each
    need their own `bun install` per candidate. The installed dependency trees
    may live in the container image while worktrees live on GPFS, where a
    direct `cp -al` fails because hardlinks cannot cross filesystems. Copy each
    source tree once into a run-local cache next to the worktrees, then
    hardlink-clone that same-filesystem cache into every candidate. The first
    copy prefers reflinks and falls back to a normal copy automatically.

    A directory tree shared via one top-level symlink instead (an earlier
    version of this function) is *not* equivalent: any file whose physical
    location is still source_repo (which, for third-party deps, is
    everything -- e.g. a published .d.ts) resolves its own `node_modules`
    lookups by walking up from that physical path, landing back on
    source_repo's own node_modules/@opencode-ai/* siblings regardless of
    which worktree you reached the file through. A real directory clone
    sidesteps that
    entirely: cloned directories are real, so the walk-up lands in *this*
    worktree's node_modules -- see `_relink_workspace_members` for the one
    remaining piece, repointing workspace-member symlinks (which the clone
    preserves as symlinks pointing at source_repo/packages/<name>) at the
    worktree's own edited copy.
    """
    workspace = workspace.resolve()
    source_repo = source_repo.resolve()
    relatives = ["node_modules"]
    # Workspace packages nest to different depths (packages/core, but also
    # packages/console/function), so walk while pruning node_modules itself.
    for root, dirnames, _ in os.walk(source_repo / "packages"):
        if "node_modules" in dirnames:
            relatives.append(str((Path(root) / "node_modules").relative_to(source_repo)))
        dirnames[:] = [name for name in dirnames if name != "node_modules" and not name.startswith(".")]
    # GitHarnessManager lays candidates out as
    #   <run-group>/worktrees/<candidate>
    # Keep the cache under <run-group>, on the exact filesystem that contains
    # the candidate worktrees. The run group is recreated for every evolver
    # batch, so the cache can never outlive or mismatch its source checkout.
    cache_root = workspace.parent.parent / ".node_modules_cache"
    linked = []
    for relative in relatives:
        source = source_repo / relative
        target = workspace / relative
        if not source.exists() or target.exists() or target.is_symlink():
            continue
        workspace_member_links = _workspace_member_links(source, source_repo)
        cached = cache_root / relative
        # Rollouts clone dependencies concurrently. Publish a cache entry only
        # after its cross-filesystem copy is complete so nobody can hardlink a
        # partially populated tree.
        with _NODE_MODULES_CACHE_LOCK:
            if not cached.exists():
                cached.parent.mkdir(parents=True, exist_ok=True)
                temporary = cached.with_name(
                    f".{cached.name}.tmp-{os.getpid()}-{threading.get_ident()}"
                )
                _remove_partial_tree(temporary)
                try:
                    _copy_node_modules_tree(source, temporary)
                    temporary.replace(cached)
                except Exception:
                    _remove_partial_tree(temporary)
                    raise
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["cp", "-al", str(cached), str(target)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            # Hardlinks can still be unavailable on unusual mounts. Keep the
            # evolver usable (at the cost of one full copy per candidate) and
            # remove cp's partial destination before retrying.
            _remove_partial_tree(target)
            _copy_node_modules_tree(cached, target)
        _relink_workspace_members(target, workspace, workspace_member_links)
        linked.append(relative)
    _settle_native_optional_dependencies(workspace)
    return linked


def _settle_native_optional_dependencies(workspace: Path) -> None:
    """Wait for platform-specific optional packages to resolve after cloning.

    bun installs each native/platform package (e.g. tsgo's
    ``@typescript/native-preview-<platform>-<arch>``) through a chain of
    symlinks nested inside ``node_modules/.bun``.  ``cp -al`` hardlinks the
    whole ``node_modules`` tree in one call and returns only once that call
    exits, but on GPFS the directory entries it just created are not always
    immediately visible to a *different* process's Node-style
    ``import.meta.resolve`` walk -- the walk can briefly see a broken link
    and raise ``Unable to resolve ...`` even though every symlink is already
    on disk and correct.  Retrying a cheap existence check a few times lets
    that window pass before the evolver agent gets a chance to run
    `bun run typecheck` and treat the transient failure as a real error.
    """
    for pattern in ("@typescript/native-preview-*",):
        for candidate in workspace.glob(f"node_modules/{pattern}"):
            if not candidate.is_symlink() and not candidate.exists():
                continue
            for attempt in range(10):
                try:
                    resolved = candidate.resolve(strict=True)
                    if (resolved / "lib").exists() or any(resolved.iterdir()):
                        break
                except (FileNotFoundError, NotADirectoryError, OSError):
                    pass
                time.sleep(0.5)



def _git_head_commit(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_path),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to resolve HEAD commit in {repo_path}\n{result.stdout}")
    return result.stdout.strip()


def register_built_binary(config: RuntimeConfig, opencode_repo_path: Path, binary: Path) -> Path:
    """Copy a freshly built binary into a commit-keyed registry.

    `bun run script/build.ts` always writes to the same dist/ path, so every
    build overwrites the previous one there. Nothing else records which
    commit produced the binary sitting at that path, so a second build (e.g.
    for a different candidate, or a rebuild of the same commit later) makes
    the earlier artifact unrecoverable. Every entry is keyed by
    (commit, build recipe).

    Bun embeds a build timestamp and randomized chunk names, so rebuilding an
    unchanged commit is not byte-for-byte reproducible. AFS publication is
    intentionally immutable at the commit path. Keep the first complete
    registry artifact for a key so a gate retry publishes the same bytes
    instead of conflicting with its own earlier upload.
    """
    commit = _git_head_commit(opencode_repo_path)
    recipe = {
        "skip_embed_web_ui": _bool_value(config.build, "skip_embed_web_ui", True),
        "binary_name": _current_binary_name(),
    }
    recipe_fingerprint = sha1(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:12]
    registry_root = config.workspace_root / "artifacts" / "binaries"
    entry_dir = registry_root / f"{commit}-{recipe_fingerprint}"
    registered_binary = entry_dir / binary.name
    metadata_path = entry_dir / "metadata.json"
    missing = not registered_binary.is_file() or registered_binary.stat().st_size == 0
    if missing:
        entry_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(binary, registered_binary)
        registered_binary.chmod(registered_binary.stat().st_mode | 0o111)
        metadata_path.write_text(
            json.dumps(
                {
                    "harness_commit": commit,
                    "recipe": recipe,
                    "recipe_fingerprint": recipe_fingerprint,
                    "source_repo_path": str(opencode_repo_path),
                    "built_from_dist_path": str(binary),
                    "registered_at": datetime.now(timezone.utc).isoformat(),
                    "size_bytes": registered_binary.stat().st_size,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    from harness_dist.skills import write_bundle
    archive = write_bundle(opencode_repo_path, registered_binary, commit)
    metadata = json.loads(metadata_path.read_text())
    metadata["skills_bundle"] = archive.name
    metadata["skills_commit"] = commit
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return registered_binary


def build_opencode_binary_from_source(config: RuntimeConfig, opencode_repo_path: Path) -> Path:
    package_dir = opencode_repo_path / "packages" / "opencode"
    if not package_dir.exists():
        raise ValueError(f"not an OpenCode source checkout: {opencode_repo_path}")
    build_env = _command_env(config)
    ensure_bun_install(opencode_repo_path, env=build_env)
    build_command = ["bun", "run", "script/build.ts", "--single", "--skip-install"]
    if _bool_value(config.build, "skip_embed_web_ui", True):
        build_command.append("--skip-embed-web-ui")
    models_snapshot = package_dir / "test" / "tool" / "fixtures" / "models-api.json"
    build_env.pop("MODELS_DEV_API_JSON", None)
    if config.build.get("models_dev_api_json"):
        build_env["MODELS_DEV_API_JSON"] = str(_repo_relative_path(str(config.build["models_dev_api_json"])))
    elif models_snapshot.exists():
        build_env["MODELS_DEV_API_JSON"] = str(models_snapshot)
    xdg_root = (
        _repo_relative_path(str(config.build["xdg_root"]))
        if config.build.get("xdg_root")
        else Path("/ram/tmp/opencode_harness_evolver_xdg") / sha1(str(opencode_repo_path).encode()).hexdigest()
    )
    for key, child in {
        "XDG_DATA_HOME": "data",
        "XDG_CONFIG_HOME": "config",
        "XDG_CACHE_HOME": "cache",
        "XDG_STATE_HOME": "state",
    }.items():
        target = xdg_root / child
        target.mkdir(parents=True, exist_ok=True)
        build_env[key] = str(target)
    result = subprocess.run(
        build_command,
        cwd=str(package_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=build_env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"opencode build failed in {package_dir}\n{result.stdout}")
    binary = package_dir / "dist" / _current_binary_name()
    if not binary.exists():
        raise RuntimeError(f"expected built opencode binary was not found: {binary}")
    binary.chmod(binary.stat().st_mode | 0o111)
    return register_built_binary(config, opencode_repo_path, binary)


def resolve_local_opencode_binary(config: RuntimeConfig, current_harness_path: str | Path) -> Path:
    path = Path(current_harness_path).expanduser().resolve()
    if path.is_file() and os.access(path, os.X_OK):
        return path
    if not path.is_dir():
        raise ValueError(f"current_harness_path is neither an executable nor a directory: {path}")
    if config.build.get("source_binary_relative_path"):
        candidate = path / str(config.build["source_binary_relative_path"])
        if candidate.is_file() and os.access(candidate, os.X_OK):
            from harness_dist.skills import write_bundle
            write_bundle(path, candidate, _git_head_commit(path))
            return candidate.resolve()
    if not _bool_value(config.build, "build_for_tb", True):
        existing = path / "packages" / "opencode" / "dist" / _current_binary_name()
        if existing.is_file() and os.access(existing, os.X_OK):
            from harness_dist.skills import write_bundle
            write_bundle(path, existing, _git_head_commit(path))
            return existing.resolve()
        raise RuntimeError(f"build.build_for_tb is false but no binary was found at {existing}")
    return build_opencode_binary_from_source(config, path)
