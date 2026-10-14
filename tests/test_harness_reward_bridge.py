import json
import subprocess
import tempfile
import unittest
from pathlib import Path


from harness_evolution.integrations.slime.reward_bridge import build_harness_change_manifest, write_harness_change


class HarnessRewardBridgeTest(unittest.TestCase):
    def test_builds_manifest_from_accepted_evolver_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "harness"
            repo.mkdir()
            subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
            (repo / "runner.py").write_text("print('base')\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "-c", "user.email=test@example.com", "-c", "user.name=test", "commit", "-qm", "base"],
                check=True,
            )
            parent = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            (repo / "runner.py").write_text("print('candidate')\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "-c", "user.email=test@example.com", "-c", "user.name=test", "commit", "-qm", "candidate"],
                check=True,
            )
            candidate = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            job = root / "job" / "trial" / "agent"
            job.mkdir(parents=True)
            trajectory = job / "trajectory.json"
            trajectory.write_text(json.dumps({"task_id": "t", "events": []}), encoding="utf-8")
            metadata = {
                "run_id": "evolver-1",
                "input_base_commit": parent,
                "accepted_candidates": [{"workspace": str(repo), "commit": candidate}],
            }
            verdict = {"accepted": True, "winner_commit": candidate, "report_path": "gate.json"}
            manifest = build_harness_change_manifest(
                metadata,
                verdict,
                active_parent={"commit": parent},
                job_dir=job.parent.parent,
                config={
                    "co_evolution": {
                        "new_capabilities": ["verification_delta"],
                        "new_events": ["verification_delta", "rollback"],
                    }
                },
            )
            self.assertEqual(manifest["schema"], "harness_change.v1")
            self.assertEqual(manifest["candidate_version"], candidate)
            self.assertEqual(manifest["changed_files"], ["runner.py"])
            self.assertEqual(manifest["source"]["trajectory_paths"], [str(trajectory.resolve())])
            destination = write_harness_change(root / "harness_change.json", manifest)
            self.assertEqual(json.loads(destination.read_text())["new_events"], ["verification_delta", "rollback"])


if __name__ == "__main__":
    unittest.main()
