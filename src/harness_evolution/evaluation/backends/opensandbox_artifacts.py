"""Download trace artifacts from an OpenSandbox session and promote the final attempt's files."""
from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from harness_evolution.evaluation.backends.opensandbox_settings import _log


def _keep_raw_trace_wire() -> bool:
    """Return whether to retain raw proxy request/response wire files.

    The ``*-resp.sse`` files are useful for low-level debugging, but they are
    not needed by the evolution evidence pipeline after normalized context and
    OpenAI-compatible message files have been captured. They are therefore
    removed after temporary fallback materialization by default. Set
    ``HARNESSRL_KEEP_RAW_TRACE_WIRE=1`` for an explicit debugging run that
    should retain the raw files as well.
    """
    value = os.environ.get("HARNESSRL_KEEP_RAW_TRACE_WIRE", "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


async def _download_trace_wire_files(runtime: Any, local_dir: Path) -> int:
    """Download proxy wire records one file at a time.

    A single ``tar | base64`` response is fragile for long OpenCode contexts:
    the combined response can exceed the execd/proxy output limit and leave
    the caller waiting forever.  The proxy writes small, independently useful
    request/response/meta files, so enumerate them with one short command and
    fetch each file through the runtime file endpoint instead.
    """
    download_file = getattr(runtime, "download_file", None)
    if not callable(download_file):
        return 0
    listing = await runtime.exec(
        "find /logs/agent/llm-reqs -maxdepth 1 -type f -printf '%f\\n' 2>/dev/null || true",
        cwd="/",
        timeout_sec=60,
    )
    names = [
        line.strip()
        for line in str(listing.stdout or "").splitlines()
        if line.strip() and "/" not in line and line.strip() not in {".", ".."}
    ]
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for name in sorted(set(names)):
        try:
            await download_file(f"/logs/agent/llm-reqs/{name}", str(local_dir / name))
            downloaded += 1
        except Exception as exc:  # noqa: BLE001
            _log(f"trace wire file {name} unavailable: {exc}")
    return downloaded


async def _download_trace_bundle(runtime: Any, session_dir: Path) -> bool:
    """Pack and download the complete remote agent log tree.

    This follows the working 2026-10-01 collector: compression happens inside
    the sandbox and the host downloads one archive through the file endpoint.
    It avoids the large stdout response produced by ``tar | base64`` and keeps
    ``llm-reqs`` plus verifier logs in one consistent snapshot.
    """
    download_file = getattr(runtime, "download_file", None)
    if not callable(download_file):
        return False
    remote_archive = "/tmp/polar-agent-artifacts.tgz"
    packed = await runtime.exec(
        "rm -f /tmp/polar-agent-artifacts.tgz && "
        "tar -C /logs -czf /tmp/polar-agent-artifacts.tgz agent verifier",
        cwd="/",
        timeout_sec=120,
    )
    if packed.return_code not in (0, None):
        _log(f"trace bundle pack failed: {packed.stderr or packed.stdout}")
        return False
    with tempfile.NamedTemporaryFile(prefix="polar-agent-artifacts-", suffix=".tgz", delete=False) as tmp:
        archive_local = Path(tmp.name)
    try:
        await download_file(remote_archive, str(archive_local))
        with tarfile.open(archive_local, mode="r:gz") as archive:
            members = archive.getmembers()
            root = session_dir.resolve()
            for member in members:
                target = (session_dir / member.name).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"unsafe trace bundle member: {member.name}")
            archive.extractall(session_dir)
        return True
    finally:
        archive_local.unlink(missing_ok=True)


async def _download_raw_trace_for_materialization(
    runtime: Any,
    session_dir: Path,
    raw_wire_dir: Path,
) -> bool:
    """Temporarily fetch raw proxy wire when normalized context is absent.

    Some task images expose the proxy's ``llm-reqs`` directory but do not
    expose a prebuilt ``llm-context.jsonl``.  The derived context can be
    reconstructed locally from the request metadata and response ``.sse``
    files.  This helper deliberately downloads those files only for the
    duration of materialization; the caller removes them unless raw-wire
    debugging was explicitly requested.
    """
    bundle_ok = False
    try:
        bundle_ok = await _download_trace_bundle(runtime, session_dir)
    except Exception as bundle_exc:  # noqa: BLE001
        _log(f"trace bundle unavailable during fallback: {bundle_exc}")
    if not bundle_ok or not any(raw_wire_dir.glob("*-meta.json")):
        try:
            n_wire_files = await _download_trace_wire_files(runtime, raw_wire_dir)
            if n_wire_files == 0:
                _log("temporary raw wire fallback returned no files")
        except Exception as raw_exc:  # noqa: BLE001
            _log(f"temporary raw wire fallback unavailable: {raw_exc}")
    return any(raw_wire_dir.glob("*-meta.json"))


def _promote_final_trial_artifacts(session_dir: Path) -> Path:
    """Promote the final retry's agent artifacts into the canonical trial dir.

    ``run_trial`` stores retries in sibling directories (``*.retry-N``) so a
    failed sandbox is never mixed with a fresh retry.  The downstream
    collector, however, consumes the canonical trial directory.  Once the
    retry loop has finished, copy the final attempt's agent tree back there so
    trajectory and normalized trace files describe the attempt whose outcome
    is recorded in ``trial.json``.
    """
    history_path = session_dir / "retry_history.json"
    if not history_path.is_file():
        return session_dir
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return session_dir
    if not isinstance(history, list) or len(history) <= 1:
        return session_dir

    final_attempt = session_dir.parent / f"{session_dir.name}.retry-{len(history) - 1}"
    source_agent = final_attempt / "agent"
    if not source_agent.is_dir():
        return session_dir

    target_agent = session_dir / "agent"
    if target_agent.exists() or target_agent.is_symlink():
        if target_agent.is_dir() and not target_agent.is_symlink():
            shutil.rmtree(target_agent)
        else:
            target_agent.unlink()
    shutil.copytree(source_agent, target_agent)
    source_remote = final_attempt / "agent.remote.jsonl"
    target_remote = session_dir / "agent.remote.jsonl"
    if source_remote.is_file():
        shutil.copy2(source_remote, target_remote)
        events: list[dict[str, Any]] = []
        for raw in target_remote.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
        (target_agent / "trajectory.json").write_text(
            json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    _log(f"promoted final retry artifacts {final_attempt} -> {session_dir}")
    return session_dir
