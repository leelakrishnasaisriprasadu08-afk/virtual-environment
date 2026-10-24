import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "examples" / "intent-classification.spec.json"
DATASET_PATH = PROJECT_ROOT / "examples" / "dialogue-intents.jsonl"


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "install the optional training extra")
class TrainingCliTests(unittest.TestCase):
    def test_cli_reviews_trains_verifies_promotes_and_predicts(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / "workspace"

            def command(*arguments):
                return subprocess.run(
                    [sys.executable, "-m", "virtual_environment.cli", *map(str, arguments)],
                    capture_output=True,
                    text=True,
                    cwd=PROJECT_ROOT,
                    check=True,
                ).stdout

            review = json.loads(command(
                "dataset-review", "--spec", SPEC_PATH, "--dataset", DATASET_PATH, "--workspace", workspace
            ))
            self.assertEqual(review["status"], "awaiting_human_approval")

            manifest = json.loads(command(
                "dataset-approve",
                "--review-id", review["review_id"],
                "--spec", SPEC_PATH,
                "--dataset", DATASET_PATH,
                "--reviewer", "cli-test-reviewer",
                "--note", "Synthetic data for CLI integration test.",
                "--workspace", workspace,
            ))
            self.assertEqual(manifest["approved_by"], "cli-test-reviewer")

            output_lines = command(
                "train", "--dataset-version", manifest["dataset_version"], "--workspace", workspace
            ).splitlines()
            events_and_run = [json.loads(line) for line in output_lines]
            run = events_and_run[-1]
            self.assertEqual(run["status"], "awaiting_approval")
            self.assertTrue(run["metrics"]["passed_thresholds"])
            self.assertEqual(events_and_run[0]["state"], "queued")
            self.assertEqual(events_and_run[-2]["state"], "awaiting_approval")

            status = json.loads(command("run-status", "--run-id", run["run_id"], "--workspace", workspace))
            self.assertEqual(status["status"], "awaiting_approval")
            events = json.loads(command("events", "--run-id", run["run_id"], "--workspace", workspace))
            self.assertEqual(len(events), 4)

            card = json.loads(command(
                "promote",
                "--run-id", run["run_id"],
                "--reviewer", "cli-release-reviewer",
                "--note", "Verified the synthetic demo metrics.",
                "--workspace", workspace,
            ))
            self.assertEqual(card["status"], "promoted")

            registered = json.loads(command("models", "--workspace", workspace))
            self.assertEqual(registered[0]["model_id"], card["model_id"])
            prediction = json.loads(command(
                "predict",
                "--model-id", card["model_id"],
                "--text", "Could you please help me?",
                "--workspace", workspace,
            ))
            self.assertEqual(prediction["label"], "request")


if __name__ == "__main__":
    unittest.main()
