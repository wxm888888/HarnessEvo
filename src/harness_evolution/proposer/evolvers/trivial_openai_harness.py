from __future__ import annotations

from harness_evolution.proposer.evolvers.base import BaseEvolver
from harness_evolution.proposer.evolvers.protocol import EvolverRequest, EvolverResult
from harness_evolution.proposer.evolvers.registry import EVOLVER_REGISTRY


@EVOLVER_REGISTRY.register("trivial_openai_harness")
class TrivialOpenAIHarnessEvolver(BaseEvolver):
    """Trivial evolver for OpenAI harness code."""

    def evolve(self, request: EvolverRequest) -> EvolverResult:
        workspace = request.workspace_resolved()
        target_file = request.require_file_target()
        _ = request.requirement
        file_path = (workspace / target_file).resolve()
        if not str(file_path).startswith(str(workspace)):
            raise ValueError(f"target path escapes repository root: {target_file}")
        if not file_path.exists():
            raise ValueError(f"target file does not exist: {target_file}")

        original = file_path.read_text(encoding="utf-8")
        updated = original

        if 'SYSTEM_PROMPT = "You are a concise task execution assistant."' in updated:
            updated = updated.replace(
                'SYSTEM_PROMPT = "You are a concise task execution assistant."',
                'SYSTEM_PROMPT = "You are a concise task execution assistant. Provide structured steps first."',
                1,
            )

        marker_line = '    print("evolve marker: openai harness")'
        if marker_line not in updated and "def run_task(" in updated:
            updated = updated.replace(
                "def run_task(task: str) -> str:\n",
                f"def run_task(task: str) -> str:\n{marker_line}\n",
                1,
            )

        if updated == original:
            updated = f"{updated.rstrip()}\n\n# evolve marker: no-op fallback applied\n"

        file_path.write_text(updated, encoding="utf-8")
        return EvolverResult(
            changed_files=[target_file],
            summary="trivial openai harness evolve applied",
            metadata={"target_scope": "file"},
        )
