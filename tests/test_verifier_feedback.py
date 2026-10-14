from __future__ import annotations

import json
from pathlib import Path



from harness_evolution.traces.verifier_feedback import (  # noqa: E402
    build_verifier_feedback,
    materialize_verifier_feedback,
)


def test_build_verifier_feedback_parses_compact_pytest_output() -> None:
    stdout = (
        "================ test session starts ================pytest-9.0"
        "collected 4 items\n../../tmp/test_final_state.py .F\n"
        "________________ test_proxy_and_load_balancing _________________"
        "/tmp/test_final_state.py:35: in test_proxy_and_load_balancing "
        "assert len(counts) == 2"
        "E   AssertionError: Expected responses from exactly 2 backends, got 1 distinct response"
        "E   assert 1 == 2"
        "================ short test summary info ================"
        "FAILED ../../tmp/test_final_state.py::test_proxy_and_load_balancing - AssertionError"
        "================ 1 failed, 1 passed in 0.06s ================\n"
        "__POLAR_VERIFIER_REWARD__=0"
    )

    feedback = build_verifier_feedback(stdout=stdout, stderr="", status="ok")

    assert feedback["framework"] == "pytest"
    assert feedback["reward"] == "0"
    assert feedback["pytest"]["collected"] == 4
    assert feedback["pytest"]["counts"] == {"failed": 1, "passed": 1}
    assert feedback["pytest"]["failed_checks"] == [
        {
            "name": "test_proxy_and_load_balancing",
            "node_id": "../../tmp/test_final_state.py::test_proxy_and_load_balancing",
            "assertion": "Expected responses from exactly 2 backends, got 1 distinct response",
        }
    ]


def test_materialize_verifier_feedback_from_historical_trial(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    agent = trial / "agent"
    agent.mkdir(parents=True)
    trajectory = agent / "trajectory.json"
    result = trial / "result.json"
    trajectory.write_text("[]\n", encoding="utf-8")
    result.write_text("{}\n", encoding="utf-8")
    (trial / "verifier.stdout.log").write_text(
        "pytest collected 1 item\n1 passed in 0.01s\n__POLAR_VERIFIER_REWARD__=1",
        encoding="utf-8",
    )
    target = tmp_path / "evidence" / "verifier-feedback.json"

    assert materialize_verifier_feedback(
        trajectory_path=trajectory,
        result_path=result,
        target_path=target,
    ) == target
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["available"] is True
    assert payload["reward"] == "1"
    assert payload["pytest"]["counts"]["passed"] == 1
