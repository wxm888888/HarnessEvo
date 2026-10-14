"""Legacy Harbor/E2B held-out backend: bake binaries into task images and run harbor jobs."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from harness_evolution.core.paths import REPO_ROOT
from harness_evolution.evaluation.gate_common import _log


def _image_prep_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config["heldout_gate"]["image_prep"]


def _relay(config: dict[str, Any]):
    """Configure and return the shared image relay module."""
    prep = _image_prep_cfg(config)
    os.environ["HARNESS_RELAY_SSH_HOST"] = str(prep.get("ssh_host") or "")
    os.environ["HARNESS_RELAY_STAGE_DIR"] = str(prep["stage_dir"])
    remote_python = str(prep.get("remote_python") or "")
    if remote_python:
        os.environ["HARNESS_RELAY_REMOTE_PYTHON"] = remote_python
    else:
        os.environ.pop("HARNESS_RELAY_REMOTE_PYTHON", None)
    from harness_dist import relay

    return relay


def _open_ssh_master(config: dict[str, Any]) -> None:
    _relay(config).open_ssh_master()


def _attach_one_image(config: dict[str, Any], commit: str, binary_path: str, base_image: str, tag_log: str) -> str:
    return _relay(config).attach_one(commit, binary_path, base_image, tag_log)


def _cleanup_tags(config: dict[str, Any], tag_log: str) -> None:
    _relay(config).cleanup_tags(tag_log, log=lambda m: _log(f"tag cleanup: {m}"))


def _resolve_task_images(config: dict[str, Any], task_names: list[str]) -> dict[str, str]:
    """task name -> registry-qualified base image ref, using the same rewrite
    rule the E2B adapter applies (HARBOR_E2B_IMAGE_REWRITE)."""
    import tomllib

    heldout = config["heldout_gate"]
    tasks_dir = Path(heldout["tasks_dir"])
    rewrite = str(heldout.get("image_rewrite", ""))
    prefix_from, _, prefix_to = rewrite.partition("=")
    images: dict[str, str] = {}
    for name in task_names:
        task_toml = tasks_dir / name / "task.toml"
        data = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        image = None
        for section in (data, data.get("environment", {}), data.get("environments", {}).get("docker", {})):
            if isinstance(section, dict) and section.get("docker_image"):
                image = str(section["docker_image"])
                break
        if not image:
            raise ValueError(f"no docker_image found in {task_toml}")
        if prefix_from and image.startswith(prefix_from):
            image = prefix_to + image[len(prefix_from):]
        images[name] = image
    return images


def _stage_binary(config: dict[str, Any], commit: str, binary_path: str) -> str:
    """Copy a locally built binary to the shared stage dir so the SSH relay
    host can read it when it needs to (re)create the carrier image."""
    prep = _image_prep_cfg(config)
    dest_dir = Path(prep["stage_dir"]) / "binaries" / commit[:12]
    dest = dest_dir / "opencode"
    if not dest.exists() or dest.stat().st_size != Path(binary_path).stat().st_size:
        dest_dir.mkdir(parents=True, exist_ok=True)
        # 0777: shared between container root and the test1714 user on relay
        # hosts; both may write layer tars next to the binary.
        os.chmod(dest_dir, 0o777)
        shutil.copy2(binary_path, dest)
        os.chmod(dest, 0o755)
    return str(dest)


def _prebuild_side(
    config: dict[str, Any],
    side: str,
    commit: str,
    binary_stage_path: str,
    base_images: list[str],
    tag_log: str,
) -> None:
    concurrency = int(_image_prep_cfg(config).get("concurrency", 8))
    t0 = time.monotonic()
    _log(f"[{side}] baking {len(base_images)} images with harness {commit} (concurrency={concurrency})")
    errors: list[str] = []
    # The first attach runs alone: for a commit the registry has never seen,
    # ensure_carrier() on the relay host creates the carrier image and its
    # local layer tar, and concurrent first-time calls would race on both.
    first, rest = base_images[0], base_images[1:]
    new_ref = _attach_one_image(config, commit, binary_stage_path, first, tag_log)
    _log(f"[{side}] 1/{len(base_images)} {first.rsplit('/', 1)[-1]} -> {new_ref.rsplit('/', 1)[-1]} (carrier ready)")
    done = 1
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(_attach_one_image, config, commit, binary_stage_path, ref, tag_log): ref
            for ref in rest
        }
        for fut in as_completed(futures):
            ref = futures[fut]
            try:
                new_ref = fut.result()
                done += 1
                _log(f"[{side}] {done}/{len(base_images)} {ref.rsplit('/', 1)[-1]} -> {new_ref.rsplit('/', 1)[-1]}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{ref}: {exc}")
    if errors:
        raise RuntimeError(f"[{side}] {len(errors)} image(s) failed to bake:\n" + "\n".join(errors[:5]))
    _log(f"[{side}] all images ready in {time.monotonic() - t0:.1f}s")


def _load_env_file(path: str | Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _write_job_config(config: dict[str, Any], run_id: str, side: str, task_names: list[str]) -> tuple[Path, str]:
    heldout = config["heldout_gate"]
    template = json.loads(Path(heldout["harbor_template"]).read_text(encoding="utf-8"))
    jobs_dir = Path(heldout["jobs_dir"])
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_name = f"gate-{run_id}-{side}"
    tasks_dir = Path(heldout["tasks_dir"])
    template["job_name"] = job_name
    template["jobs_dir"] = str(jobs_dir)
    template["n_attempts"] = int(heldout.get("n_attempts", 1))
    template["orchestrator"]["n_concurrent_trials"] = int(heldout.get("n_concurrent", 4))
    template["tasks"] = [{"path": str(tasks_dir / name)} for name in task_names]
    filebridge = heldout.get("filebridge") or {}
    if filebridge.get("enabled"):
        # E2B file bridge instead of whatever baseURL the template carries
        # (historically a hand-started trycloudflare tunnel). The agent boots a
        # sandbox-local OpenAI endpoint; a host-side watcher relays requests to
        # the real router via E2B file sync, so no public URL is required.
        bridge_base_url = str(filebridge.get("bridge_base_url", "http://127.0.0.1:30000/v1"))
        agent = template["agents"][0]
        # harbor resolves a recognized agent *name* before import_path
        # (AgentFactory.create_agent_from_config), and the template names the
        # built-in "opencode" agent -- which npm-installs the official binary
        # over the baked harness. name must be cleared for import_path to win.
        agent["name"] = None
        agent["import_path"] = str(filebridge["agent_import_path"])
        providers = agent.get("kwargs", {}).get("opencode_config", {}).get("provider", {})
        for provider_cfg in providers.values():
            provider_cfg.setdefault("options", {})["baseURL"] = bridge_base_url
    config_path = jobs_dir / f"{job_name}.json"
    config_path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    return config_path, job_name


def _start_filebridge_watcher(
    filebridge: dict[str, Any], jobs_dir: Path, job_name: str, env: dict[str, str]
) -> subprocess.Popen | None:
    """Host-side relay: watches the job dir for e2b-sandbox-id.txt markers and
    starts one file relay per sandbox, forwarding the sandbox-local bridge to
    the real router URL. Watches jobs_dir (not the job subdir, which harbor
    itself creates) -- pre-existing markers are ignored by the watcher."""
    root = Path(filebridge["harbor_e2b_root"])
    watcher_script = root / "watch_e2b_file_relays.py"
    if not watcher_script.is_file():
        raise RuntimeError(f"filebridge watcher not found: {watcher_script}")
    log_path = jobs_dir / f"{job_name}.filebridge.log"
    cmd = [
        sys.executable,
        str(watcher_script),
        "--runs-dir", str(jobs_dir),
        "--sglang-base-url", str(filebridge["sglang_base_url"]),
    ]
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    _log(f"filebridge watcher started (pid {proc.pid}, upstream {filebridge['sglang_base_url']}, log: {log_path})")
    return proc


def _stop_filebridge_watcher(watcher: subprocess.Popen | None) -> None:
    if watcher is None or watcher.poll() is not None:
        return
    watcher.terminate()
    try:
        watcher.wait(timeout=15)
    except subprocess.TimeoutExpired:
        watcher.kill()


def _harbor_env(config: dict[str, Any], commit: str) -> dict[str, str]:
    heldout = config["heldout_gate"]
    env = {**os.environ, **_load_env_file(heldout["e2b_env_file"])}
    env["E2B_DOMAIN"] = str(heldout.get("e2b_domain", env.get("E2B_DOMAIN", "")))
    env["PYTHONPATH"] = str(heldout["adapter_pythonpath"])
    env["HARBOR_E2B_IMAGE_REWRITE"] = str(heldout.get("image_rewrite", ""))
    env["HARBOR_E2B_IMAGE_TAG_SUFFIX"] = f"-harness-{commit[:7]}"
    # E2B control plane and server-side image pull both ride the internal
    # network; an inherited proxy only breaks them.
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        env.pop(proxy_var, None)
    # Match training (run_tb_train_harbor.sh): e2b-sdk caches ONE http2
    # transport per event loop, so the default keepalive pool is the ceiling on
    # every concurrent sandbox operation in a harbor job -- sandbox creation as
    # well as the log up/download around verification.
    env.setdefault("E2B_MAX_KEEPALIVE_CONNECTIONS", "192")
    filebridge = heldout.get("filebridge") or {}
    if filebridge.get("enabled"):
        # The FileBridgeBakedOpenCode agent lives under <harbor_e2b_root>/examples;
        # harness_evolution.evaluation.backends.harbor_filebridge subclasses it.
        env["PYTHONPATH"] = f"{env['PYTHONPATH']}:{filebridge['harbor_e2b_root']}:{REPO_ROOT / 'src'}"
        env.setdefault("OPENAI_API_KEY", "dummy")
    return env


def _run_harbor(
    config: dict[str, Any],
    job_config: Path,
    job_name: str,
    commit: str,
    *,
    shared_watcher: subprocess.Popen | None = None,
) -> Path:
    """Run one side (baseline or candidate) of the held-out eval.

    When shared_watcher is None, this manages its own FileBridge watcher
    (backward-compatible, sequential mode). When a shared_watcher is passed
    in, the caller owns its lifecycle -- this lets baseline and candidate
    run concurrently against one watcher that rglobs the whole jobs_dir for
    e2b-sandbox-id.txt markers, instead of racing two watcher processes over
    the same directory.
    """
    heldout = config["heldout_gate"]
    jobs_dir = Path(heldout["jobs_dir"])
    env = _harbor_env(config, commit)
    filebridge = heldout.get("filebridge") or {}
    own_watcher: subprocess.Popen | None = None
    if shared_watcher is None and filebridge.get("enabled"):
        own_watcher = _start_filebridge_watcher(filebridge, jobs_dir, job_name, env)
    log_path = jobs_dir / f"{job_name}.log"
    cmd = [
        str(heldout.get("harbor_bin", "harbor")),
        "run",
        "-c", str(job_config),
        "--job-name", job_name,
        "--n-concurrent", str(int(heldout.get("n_concurrent", 4))),
        "--n-attempts", str(int(heldout.get("n_attempts", 1))),
        "-y",
    ]
    # Backstop only: every phase harbor runs has its own timeout, but the log
    # up/download around verification does not, so a stalled transfer used to
    # be able to hang a round indefinitely. Sized to never fire in normal
    # operation (a healthy round is well under an hour) -- it exists so a hang
    # fails the evolution round instead of stalling training overnight.
    job_timeout_sec = float(heldout.get("job_timeout_sec", 10800))
    _log(f"harbor run start: {job_name} (tag suffix -harness-{commit[:7]}, log: {log_path})")
    t0 = time.monotonic()
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            # start_new_session: harbor spawns its own children (e2b helpers,
            # verifier), so the backstop has to kill the process group.
            proc = subprocess.Popen(
                cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
            try:
                returncode = proc.wait(timeout=job_timeout_sec if job_timeout_sec > 0 else None)
            except subprocess.TimeoutExpired:
                _log(f"harbor run BACKSTOP TIMEOUT: {job_name} exceeded {job_timeout_sec:.0f}s, killing")
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(proc.pid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        returncode = proc.wait(timeout=60)
                        break
                    except subprocess.TimeoutExpired:
                        returncode = -1
    finally:
        _stop_filebridge_watcher(own_watcher)
    _log(f"harbor run done: {job_name} exit {returncode} in {time.monotonic() - t0:.0f}s")
    job_dir = jobs_dir / job_name
    if not (job_dir / "result.json").exists():
        raise RuntimeError(f"harbor produced no result.json for {job_name}; see {log_path}")
    return job_dir


def _parse_task_passes(job_dir: Path) -> tuple[dict[str, int], int]:
    """Return ({task_name: pass_count}, n_errored_trials)."""
    data = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
    evals = data["stats"]["evals"]
    passes: dict[str, int] = {}
    for eval_stats in evals.values():
        reward_map = (eval_stats.get("reward_stats") or {}).get("reward") or {}
        for value, trials in reward_map.items():
            for trial in trials:
                task = trial.split("__", 1)[0]
                passes.setdefault(task, 0)
                if float(value) > 0.5:
                    passes[task] += 1
    return passes, int(data["stats"].get("n_errored_trials", 0))
