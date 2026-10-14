"""Held-out backend selection, OpenSandbox settings and the binaries each side runs."""
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from harness_evolution.core.paths import ensure_external_paths

ensure_external_paths()

from harness_dist import afs as harness_afs
from harness_evolution.tasks.catalog import _first_nonempty


def _log(msg: str) -> None:
    print(f"[gate.opensandbox {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def custom_afs_binary_path(
    root: str,
    *,
    run_tag: str,
    side_label: str,
    commit: str,
) -> str:
    """Return an immutable, per-run/per-side OpenSandbox AFS path.

    The default `{root}/{commit[:7]}/opencode` layout remains supported for
    callers that do not need isolation. Evolution gates use this helper so
    each step/candidate has the explicit layout
    ``{root}/wxh/{time}-step-{n}-candidate-{n}/opencode``.
    """
    root = str(root or "/sandbox-data/duchenyu/opencode").rstrip("/")
    if not root.startswith("/sandbox-data/") or ".." in Path(root).parts:
        raise ValueError("OpenSandbox AFS root must be below /sandbox-data/")
    commit = str(commit or "").strip()
    if not commit:
        raise ValueError("custom AFS path requires a commit")

    def safe(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip())
        return cleaned.strip("-.") or "item"

    return f"{root}/wxh/{safe(run_tag)}-{safe(side_label)}/opencode"


def heldout_backend(heldout: dict[str, Any]) -> str:
    """Which held-out runner to use.

    Explicit ``heldout_gate.backend`` / ``HARNESS_GATE_BACKEND`` wins.
    Default is OpenSandbox; Harbor only if the yaml still points at it
    and nothing asked for OpenSandbox.
    """
    explicit = _first_nonempty(heldout.get("backend"), os.environ.get("HARNESS_GATE_BACKEND")).lower()
    if explicit:
        return explicit
    if heldout.get("opensandbox") is not None:
        return "opensandbox"
    if heldout.get("harbor_template"):
        return "harbor"
    return "opensandbox"


def opensandbox_settings(heldout: dict[str, Any]) -> dict[str, Any]:
    osb = dict(heldout.get("opensandbox") or {})

    def pick(key: str, *envs: str, default: str = "") -> str:
        return _first_nonempty(osb.get(key), *(os.environ.get(env, "") for env in envs), default=default)

    create_rate = osb.get("create_rate")
    if create_rate in (None, ""):
        create_rate = os.environ.get("SANDBOX_CREATES_PER_SEC", "0.2")
    create_burst = osb.get("create_burst")
    if create_burst in (None, ""):
        create_burst = os.environ.get("SANDBOX_CREATES_BURST", "1")
    agent_timeout = osb.get("agent_timeout_sec")
    if agent_timeout in (None, ""):
        agent_timeout = os.environ.get("POLAR_AGENT_TIMEOUT_SEC", "1800")
    model = pick("agent_model_name", "AGENT_MODEL_NAME", "LLM_PROXY_MODEL_NAME")
    model_names_value = osb.get("agent_model_names")
    if model_names_value in (None, ""):
        model_names_value = os.environ.get("AGENT_MODEL_NAMES") or os.environ.get("LLM_REPLICA_MODELS") or ""
    if isinstance(model_names_value, (list, tuple)):
        model_names = [str(item).strip() for item in model_names_value if str(item).strip()]
    else:
        model_names = [item.strip() for item in str(model_names_value).split(",") if item.strip()]
    if not model_names and model:
        model_names = [model]
    # This is a gate-wide deadline shared by baseline and all candidate
    # trials.  An environment override is useful for a frozen/resumed run:
    # changing the timeout must not require rewriting its copied gate.yaml.
    gate_timeout = os.environ.get("HARNESSRL_GATE_TIMEOUT_SEC") or osb.get(
        "gate_timeout_sec", heldout.get("gate_timeout_sec", 0)
    )
    gate_cleanup_grace = os.environ.get("HARNESSRL_GATE_CLEANUP_GRACE_SEC") or osb.get(
        "gate_cleanup_grace_sec", heldout.get("gate_cleanup_grace_sec", 60)
    )
    # OpenSandbox now disables the implicit project mount when a request
    # contains a `volumes` field.  Keep the historical project view explicit
    # so AFS paths under /sandbox-data remain visible in every task sandbox.
    # A gate config may provide a custom list for a deliberate minimal view.
    volumes = osb.get("volumes")
    if volumes is None:
        volumes = [{"mountPath": "/sandbox-data"}]
    elif not isinstance(volumes, list):
        raise ValueError("opensandbox.volumes must be a list when provided")
    return {
        "env_file": pick("env_file", "OPEN_SANDBOX_SECRET_ENV"),
        "base_url": pick("base_url", "OPEN_SANDBOX_BASE", default="http://10.12.111.143"),
        "project": pick("project", "OPEN_SANDBOX_PROJECT", default="ailab-safethm"),
        "docker_registry": pick(
            "docker_registry", "OPEN_SANDBOX_DOCKER_REGISTRY", default="registry-v2.h.pjlab.org.cn"
        ),
        "agent_gateway_url": pick("agent_gateway_url", "AGENT_GATEWAY_URL"),
        "agent_model_name": model or (model_names[0] if model_names else ""),
        "agent_model_names": model_names,
        "opencode_afs_root": pick(
            "opencode_afs_root", "OPENCODE_AFS_ROOT", default="/sandbox-data/duchenyu/opencode"
        ),
        "volumes": volumes,
        "opencode_setup": _first_nonempty(
            os.environ.get("HARNESSRL_OPENCODE_SETUP"),
            pick("opencode_setup", "OPENCODE_SETUP", default="afs_bin"),
        ).lower(),
        "cpu": pick("cpu", "SANDBOX_CPU", default="1"),
        "memory": pick("memory", "SANDBOX_MEMORY", default="4Gi"),
        "create_rate": float(create_rate),
        "create_burst": int(create_burst),
        "create_retries": int(osb.get("create_retries") or os.environ.get("SANDBOX_CREATE_RETRIES", "5")),
        # A trial-level retry always creates a fresh sandbox and reruns the
        # complete agent+verifier flow.  This is deliberately separate from
        # command-receipt recovery: an unknown command result must never be
        # replayed inside the same sandbox.
        "trial_retries": int(osb.get("trial_retries") or os.environ.get("SANDBOX_TRIAL_RETRIES", "2")),
        "create_timeout": int(osb.get("create_timeout") or os.environ.get("SANDBOX_CREATE_TIMEOUT", "300")),
        "agent_timeout_sec": int(agent_timeout),
        "verifier_timeout_sec": int(osb.get("verifier_timeout_sec") or 300),
        "gate_timeout_sec": max(0, int(gate_timeout or 0)),
        "gate_cleanup_grace_sec": max(0.0, float(gate_cleanup_grace or 0)),
        "ttl_seconds": int(osb.get("ttl_seconds") or 21600),
        "workdir": pick("workdir", default="/home/user"),
        # Match the TB2.1 runtime defaults.  A smaller fixed context/output
        # budget silently truncates difficult tasks and changes the score.
        "context_limit": int(osb.get("opencode_context_limit") or os.environ.get("OPENCODE_CONTEXT_LIMIT", "262144")),
        "output_limit": int(osb.get("opencode_output_limit") or os.environ.get("OPENCODE_OUTPUT_LIMIT", "32768")),
    }


def _should_publish_for_eval() -> bool:
    flag = os.environ.get("HARNESS_AFS_PUBLISH", "").strip()
    if flag == "0":
        return False
    return True


def publish_side_binaries(
    sides: Iterable[tuple[str, str, str] | tuple[str, str, str, str]],
    *,
    publish: Callable[..., str] | None = None,
    env_file: str = "",
) -> None:
    """Upload each unique (commit, binary) so sandboxes can copy it.

    Held-out eval needs the candidate on AFS *before* promote; this is not
    gated on harness_dist.afs.should_publish() (that's the ACCEPT path).
    """
    if not _should_publish_for_eval():
        _log("skipping AFS publish (HARNESS_AFS_PUBLISH=0); sandboxes must already see the binaries")
        return
    custom_publisher = publish is not None
    publisher = publish or harness_afs.publish_binary
    seen: set[str] = set()
    for side_info in sides:
        _label, commit, binary = side_info[:3]
        afs_bin = side_info[3] if len(side_info) >= 4 else ""
        if commit in seen:
            continue
        seen.add(commit)
        if not binary or not Path(binary).exists():
            _log(f"WARNING: no local binary for {commit[:12]}; assuming AFS already has {commit[:7]}/opencode")
            continue
        publish_kwargs: dict[str, Any] = {"log": _log}
        if env_file and not custom_publisher:
            # The AFS publisher starts a short-lived OpenSandbox transport
            # sandbox. Pass the same credential file used by evaluation
            # runtimes; relying only on OPEN_SANDBOX_ENV_FILE leaves the
            # publisher runtime without access-key variables.
            publish_kwargs["env_file"] = env_file
        if afs_bin:
            publish_kwargs["afs_path"] = afs_bin
        dest = publisher(binary, commit, **publish_kwargs)
        _log(f"published {commit[:12]} -> {dest}")
