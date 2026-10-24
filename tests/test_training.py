import json
import tempfile
import unittest
from pathlib import Path

from virtual_environment.training import (
    approve_dataset,
    list_models,
    predict,
    promote_run,
    review_dataset,
    run_events,
    train_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "examples" / "intent-classification.spec.json"
DATASET_PATH = PROJECT_ROOT / "examples" / "dialogue-intents.jsonl"


class TrainingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name) / "training"

    def tearDown(self):
        self.temp_dir.cleanup()

    def approve_example_dataset(self):
        report = review_dataset(SPEC_PATH, DATASET_PATH, self.workspace)
        manifest = approve_dataset(
            report["review_id"],
            SPEC_PATH,
            DATASET_PATH,
            self.workspace,
            "test-reviewer",
            "Synthetic example data for isolated workflow tests.",
        )
        return report, manifest

    def train_example_dataset(self):
        _, manifest = self.approve_example_dataset()
        return manifest, train_dataset(manifest["dataset_version"], self.workspace)

    def test_review_counts_examples_and_requires_person_approval(self):
        report = review_dataset(SPEC_PATH, DATASET_PATH, self.workspace)
        self.assertEqual(report["status"], "awaiting_human_approval")
        self.assertEqual(report["example_count"], 75)
        self.assertEqual(set(report["label_counts"].values()), {15, 20})
        self.assertEqual(report["checks"]["license_and_intended_use"], "human review required")

    def test_review_rejects_contact_information(self):
        dataset = Path(self.temp_dir.name) / "personal.jsonl"
        dataset.write_text('{"text":"Contact person@example.com for details","label":"greeting"}\n')
        with self.assertRaisesRegex(ValueError, "personal contact information"):
            review_dataset(SPEC_PATH, dataset, self.workspace)

    def test_dataset_cannot_change_after_review(self):
        report = review_dataset(SPEC_PATH, DATASET_PATH, self.workspace)
        changed_dataset = Path(self.temp_dir.name) / "changed.jsonl"
        changed_dataset.write_bytes(DATASET_PATH.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "changed after review"):
            approve_dataset(
                report["review_id"],
                SPEC_PATH,
                changed_dataset,
                self.workspace,
                "test-reviewer",
                "The changed content is rejected.",
            )

    def test_training_records_heldout_metrics_and_stream_events(self):
        _, run = self.train_example_dataset()
        metrics = run["metrics"]
        self.assertEqual(run["status"], "awaiting_approval")
        self.assertGreater(metrics["test_examples"], 0)
        self.assertGreater(metrics["validation_examples"], 0)
        self.assertGreater(metrics["accuracy"], metrics["majority_baseline_accuracy"])
        self.assertTrue(metrics["per_label_thresholds_passed"])
        self.assertEqual(metrics["label_order"], ["greeting", "question", "request", "thanks"])
        states = [event["state"] for event in run_events(run["run_id"], self.workspace)]
        self.assertEqual(states, ["queued", "training", "evaluating", "awaiting_approval"])
        with self.assertRaisesRegex(ValueError, "reviewer name"):
            promote_run(run["run_id"], self.workspace, "", "")

    def test_passing_model_requires_explicit_promotion_and_predicts(self):
        _, run = self.train_example_dataset()
        card = promote_run(
            run["run_id"],
            self.workspace,
            "release-reviewer",
            "Reviewed per-class results for the synthetic demo.",
        )
        self.assertEqual(card["status"], "promoted")
        self.assertEqual(card["release_approval"]["approved_by"], "release-reviewer")
        self.assertEqual(list_models(self.workspace)[0]["model_id"], card["model_id"])
        result = predict(card["model_id"], "Could you please help me?", self.workspace)
        self.assertEqual(result["label"], "request")
        self.assertGreater(float(result["confidence"]), 0)

    def test_changed_model_artifact_cannot_be_loaded(self):
        _, run = self.train_example_dataset()
        card = promote_run(run["run_id"], self.workspace, "reviewer", "Test only.")
        artifact = self.workspace / "runs" / run["run_id"] / card["model_artifact"]
        artifact.write_bytes(artifact.read_bytes() + b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            predict(card["model_id"], "Hello there.", self.workspace)

    def test_reviewer_and_approval_note_are_required(self):
        report = review_dataset(SPEC_PATH, DATASET_PATH, self.workspace)
        with self.assertRaisesRegex(ValueError, "reviewer name"):
            approve_dataset(
                report["review_id"], SPEC_PATH, DATASET_PATH, self.workspace, "", ""
            )


if __name__ == "__main__":
    unittest.main()
