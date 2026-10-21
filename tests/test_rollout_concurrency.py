from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from harness_evolution.proposer.batch import _collect_rollout_results


def test_runs_four_candidates_with_two_workers(monkeypatch):
    monkeypatch.setenv("HARNESSRL_EVOLVER_CONCURRENCY", "2")
    lock = threading.Lock()
    release = threading.Event()
    two_started = threading.Event()
    active = 0
    peak = 0
    started = []

    def candidate(index):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            started.append(index)
            if len(started) == 2:
                two_started.set()
        if not release.wait(timeout=5):
            raise RuntimeError("candidate release timed out")
        with lock:
            active -= 1
        return f"candidate-{index}"

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(_collect_rollout_results, candidate, 4)
        try:
            assert two_started.wait(timeout=5)
            with lock:
                assert len(started) == 2
                assert peak == 2
        finally:
            release.set()
        results = pending.result(timeout=5)
    assert results == {i: f"candidate-{i}" for i in range(4)}
    assert peak == 2


def test_default_retains_all_candidates(monkeypatch):
    monkeypatch.delenv("HARNESSRL_EVOLVER_CONCURRENCY", raising=False)
    assert _collect_rollout_results(lambda index: index, 4) == dict(enumerate(range(4)))


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_rejects_invalid_concurrency_before_start(monkeypatch, value):
    monkeypatch.setenv("HARNESSRL_EVOLVER_CONCURRENCY", value)
    started = []
    with pytest.raises(ValueError):
        _collect_rollout_results(started.append, 4)
    assert started == []
