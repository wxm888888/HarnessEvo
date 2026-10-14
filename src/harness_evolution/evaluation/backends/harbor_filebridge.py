"""Held-out gate's opencode agent: FileBridgeBakedOpenCode, minus the relay
trace sitting in harbor's round-tripped log directory.

watch_e2b_file_relays.py points each relay's trace directory at
<trial>/agent/relay-trace, i.e. *inside* the agent log directory that harbor's
Trial uploads back into the sandbox before verification (see
Trial._maybe_upload_agent_logs -- it exists so the verifier can read files the
agent produced host-side, e.g. trajectory.json). That trace is the full
request/response log of every model call: ~140 MB on a long task, versus a few
hundred KB for everything else in there. Neither the download nor the upload
step has a timeout, and with a whole held-out job's trials uploading at once
the measured throughput was ~39 KB/s -- so a big trace bought 40-60 minutes of
dead wait *after* the agent had already finished, which is why a gate round
took ~75 minutes while training's own 256-trajectory rollout takes ~30 (the
training path runs the verifier through polar's runtime and never round-trips
agent logs at all).

Moving the trace one level up, to <trial>/relay-trace, keeps every byte of it
for debugging while leaving harbor only the small files it actually needs to
send back.
"""
from __future__ import annotations

import os

from examples.terminal_bench.harbor_file_bridge_baked_opencode import FileBridgeBakedOpenCode

_TRACE_DIRNAME = "relay-trace"


class TraceOutsideLogsBakedOpenCode(FileBridgeBakedOpenCode):
    async def run(self, instruction, environment, context):
        # finally, not a plain post-call: on the agent-timeout path harbor
        # still downloads and re-uploads the log directory (Trial.run's
        # AgentTimeoutError branch), so a timed-out trial would otherwise pay
        # the full upload cost -- and those are exactly the trials with the
        # largest traces.
        try:
            return await super().run(instruction, environment, context)
        finally:
            self._externalize_relay_trace()

    def _externalize_relay_trace(self) -> None:
        source = self.logs_dir / _TRACE_DIRNAME
        if not source.is_dir():
            return
        target = self.logs_dir.parent / _TRACE_DIRNAME
        try:
            if target.exists():
                target = target.with_name(f"{_TRACE_DIRNAME}-{os.getpid()}")
            # Renaming a directory the relay process still has files open in is
            # safe: its open descriptors follow the inode. If it appends by
            # reopening, it recreates the original path -- but only with
            # whatever arrives after the agent already stopped calling the
            # model, i.e. nothing worth uploading.
            os.replace(source, target)
        except OSError as exc:
            print(f"[gate.agent] could not move relay trace out of {source}: {exc}", flush=True)
