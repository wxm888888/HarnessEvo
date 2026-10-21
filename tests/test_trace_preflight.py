from __future__ import annotations

import json

from harness_evolution.core.types import CollectionArtifact
from harness_evolution.traces.preflight import classify_collection_artifact


def test_reward_zero_verifier_failure_is_valid_evolution_evidence(tmp_path):
    trial = tmp_path / "trial"
    agent = trial / "agent"
    agent.mkdir(parents=True)
    result = trial / "result.json"
    result.write_text(
        json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}}),
        encoding="utf-8",
    )
    trajectory = agent / "trajectory.json"
    trajectory.write_text("{}\n", encoding="utf-8")
    (agent / "openai-messages.jsonl").write_text("{}\n", encoding="utf-8")
    (agent / "openai-tools.json").write_text("[]\n", encoding="utf-8")
    (agent / "artifact-status.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    (agent / "verifier-summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "reward": "0.0",
                # A nonzero test-runner exit is the ordinary negative task
                # outcome, not a verifier infrastructure failure.
                "failure_class": "verifier_failure",
            }
        ),
        encoding="utf-8",
    )

    decision = classify_collection_artifact(
        CollectionArtifact(
            trajectory_path=trajectory,
            result_path=result,
            openai_messages_path=agent / "openai-messages.jsonl",
        )
    )

    assert decision.eligible is True
    assert decision.reason_code == "eligible"
