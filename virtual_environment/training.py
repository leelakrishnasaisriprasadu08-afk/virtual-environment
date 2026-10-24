"""Local dataset review, text-classifier training, verification, and model registry."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MAX_DATASET_BYTES = 25 * 1024 * 1024
MAX_EXAMPLES = 100_000
MAX_TEXT_CHARS = 4_000
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{8,}\d)(?!\w)")
SLUG_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")
REVIEW_ID_PATTERN = re.compile(r"^[a-f0-9]{16}$")
RUN_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_id(value: str, pattern: re.Pattern[str], description: str) -> str:
    if not pattern.fullmatch(value):
        raise ValueError(f"Invalid {description}.")
    return value


def _write_bytes(path: Path, data: bytes) -> None:
    _secure_directory(path.parent)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(data)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    if os.name == "posix":
        temporary_path.chmod(0o600)
    os.replace(temporary_path, path)
    if os.name == "posix":
        path.chmod(0o600)


def _secure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        path.chmod(0o700)


def _write_json(path: Path, value: Any) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    _write_bytes(path, encoded)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read valid JSON from {path}.") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _load_spec(spec_path: str | Path) -> tuple[dict[str, Any], bytes]:
    path = Path(spec_path).expanduser()
    try:
        raw = path.read_bytes()
        spec = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Could not read a valid model specification JSON file.") from error
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("Model specification must be a JSON object with schema_version 1.")

    task = spec.get("task")
    dataset = spec.get("dataset")
    training = spec.get("training")
    if not isinstance(task, dict) or not isinstance(dataset, dict) or not isinstance(training, dict):
        raise ValueError("Specification must include task, dataset, and training objects.")
    if task.get("type") != "text_classification":
        raise ValueError("This first training workflow supports task.type='text_classification'.")
    for section, field in ((task, "id"), (dataset, "id"), (dataset, "source"), (dataset, "license"), (dataset, "intended_use")):
        value = section.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Specification field '{field}' must be a non-empty string.")
    for identifier in (task["id"], dataset["id"]):
        if not SLUG_PATTERN.fullmatch(identifier):
            raise ValueError("Task and dataset IDs may contain letters, numbers, '.', '_' and '-'.")

    labels = task.get("labels")
    if not isinstance(labels, list) or not 2 <= len(labels) <= 50:
        raise ValueError("task.labels must contain between 2 and 50 labels.")
    if any(not isinstance(label, str) or not label.strip() for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("task.labels must contain unique, non-empty strings.")
    task["labels"] = [label.strip() for label in labels]

    min_examples = training.get("min_examples_per_label", 4)
    test_fraction = training.get("test_fraction", 0.25)
    validation_fraction = training.get("validation_fraction", 0.2)
    seed = training.get("random_seed", 17)
    min_accuracy = training.get("min_accuracy", 0.6)
    min_macro_f1 = training.get("min_macro_f1", 0.6)
    min_per_label_f1 = training.get("min_per_label_f1", 0.3)
    if not isinstance(min_examples, int) or min_examples < 2:
        raise ValueError("training.min_examples_per_label must be an integer >= 2.")
    if not isinstance(test_fraction, (int, float)) or not 0.1 <= test_fraction <= 0.5:
        raise ValueError("training.test_fraction must be between 0.1 and 0.5.")
    if not isinstance(validation_fraction, (int, float)) or not 0.1 <= validation_fraction <= 0.4:
        raise ValueError("training.validation_fraction must be between 0.1 and 0.4.")
    if test_fraction + validation_fraction >= 0.8:
        raise ValueError("Training data must retain at least 20% of examples.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("training.random_seed must be a non-negative integer.")
    for name, threshold in (
        ("min_accuracy", min_accuracy),
        ("min_macro_f1", min_macro_f1),
        ("min_per_label_f1", min_per_label_f1),
    ):
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError(f"training.{name} must be between 0 and 1.")
    training["min_examples_per_label"] = min_examples
    training["test_fraction"] = float(test_fraction)
    training["validation_fraction"] = float(validation_fraction)
    training["random_seed"] = seed
    training["min_accuracy"] = float(min_accuracy)
    training["min_macro_f1"] = float(min_macro_f1)
    training["min_per_label_f1"] = float(min_per_label_f1)
    return spec, raw


def _parse_dataset(raw: bytes, labels: list[str], min_examples: int) -> tuple[list[dict[str, str]], dict[str, int]]:
    if len(raw) > MAX_DATASET_BYTES:
        raise ValueError(f"Dataset exceeds the {MAX_DATASET_BYTES // (1024 * 1024)} MB review limit.")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Dataset must be UTF-8 encoded JSON Lines.") from error

    allowed_labels = set(labels)
    rows: list[dict[str, str]] = []
    counts = {label: 0 for label in labels}
    seen_text: set[str] = set()
    for line_number, line in enumerate(decoded.splitlines(), 1):
        if not line.strip():
            continue
        if len(rows) >= MAX_EXAMPLES:
            raise ValueError(f"Dataset exceeds the {MAX_EXAMPLES} example review limit.")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Dataset line {line_number} is not valid JSON.") from error
        if not isinstance(row, dict) or set(row) != {"text", "label"}:
            raise ValueError(f"Dataset line {line_number} must contain exactly 'text' and 'label'.")
        text, label = row["text"], row["label"]
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"Dataset line {line_number} has empty text or exceeds {MAX_TEXT_CHARS} characters.")
        if not isinstance(label, str) or label not in allowed_labels:
            raise ValueError(f"Dataset line {line_number} uses a label not declared in the model specification.")
        if EMAIL_PATTERN.search(text) or PHONE_PATTERN.search(text):
            raise ValueError(f"Dataset line {line_number} may contain personal contact information; review it manually.")
        normalized = " ".join(text.casefold().split())
        if normalized in seen_text:
            raise ValueError(f"Dataset contains duplicate or near-identical text at line {line_number}.")
        seen_text.add(normalized)
        rows.append({"text": text.strip(), "label": label})
        counts[label] += 1

    if len(rows) < len(labels) * min_examples:
        raise ValueError("Dataset is too small to meet the per-label example minimum.")
    missing = [label for label, count in counts.items() if count < min_examples]
    if missing:
        raise ValueError("Insufficient examples for labels: " + ", ".join(missing))
    return rows, counts


def review_dataset(spec_path: str | Path, dataset_path: str | Path, workspace: str | Path) -> dict[str, Any]:
    """Validate dataset shape and policy metadata; save a report awaiting human approval."""
    spec, spec_raw = _load_spec(spec_path)
    dataset_file = Path(dataset_path).expanduser()
    try:
        if dataset_file.stat().st_size > MAX_DATASET_BYTES:
            raise ValueError(f"Dataset exceeds the {MAX_DATASET_BYTES // (1024 * 1024)} MB review limit.")
        dataset_raw = dataset_file.read_bytes()
    except OSError as error:
        raise ValueError("Could not read the specified dataset file.") from error
    rows, counts = _parse_dataset(
        dataset_raw,
        spec["task"]["labels"],
        spec["training"]["min_examples_per_label"],
    )
    review_id = _sha256(spec_raw + b"\0" + dataset_raw)[:16]
    report = {
        "review_id": review_id,
        "status": "awaiting_human_approval",
        "reviewed_at": _utc_now(),
        "spec_sha256": _sha256(spec_raw),
        "dataset_sha256": _sha256(dataset_raw),
        "task": spec["task"],
        "dataset": spec["dataset"],
        "example_count": len(rows),
        "label_counts": counts,
        "checks": {
            "utf8_jsonl_schema": "passed",
            "declared_labels_and_minimums": "passed",
            "duplicate_texts": "passed",
            "email_or_phone_pattern_scan": "passed",
            "license_and_intended_use": "human review required",
        },
        "note": "Automated checks do not establish license validity, consent, representativeness, or absence of all personal data.",
    }
    report_path = Path(workspace).expanduser() / "reviews" / f"{review_id}.json"
    _write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def approve_dataset(
    review_id: str,
    spec_path: str | Path,
    dataset_path: str | Path,
    workspace: str | Path,
    reviewer: str,
    approval_note: str,
) -> dict[str, Any]:
    """Create a content-addressed dataset version after explicit human approval."""
    _validate_id(review_id, REVIEW_ID_PATTERN, "dataset review ID")
    root = Path(workspace).expanduser()
    review_path = root / "reviews" / f"{review_id}.json"
    report = _read_json(review_path)
    if report.get("review_id") != review_id or report.get("status") != "awaiting_human_approval":
        raise ValueError("Review is missing, already approved, or has an invalid state.")
    if not reviewer.strip() or not approval_note.strip():
        raise ValueError("A reviewer name and approval note are required.")
    spec, spec_raw = _load_spec(spec_path)
    try:
        dataset_raw = Path(dataset_path).expanduser().read_bytes()
    except OSError as error:
        raise ValueError("Could not read the dataset for approval.") from error
    if _sha256(spec_raw) != report.get("spec_sha256") or _sha256(dataset_raw) != report.get("dataset_sha256"):
        raise ValueError("Specification or dataset changed after review; create a new review.")

    version = review_id
    version_dir = root / "datasets" / version
    _secure_directory(version_dir)
    dataset_target = version_dir / "dataset.jsonl"
    spec_target = version_dir / "spec.json"
    if dataset_target.exists() or spec_target.exists():
        raise ValueError("Dataset version already exists; use its existing approval record.")
    _write_bytes(spec_target, spec_raw)
    _write_json(
        version_dir / "manifest.json",
        {
            "dataset_version": version,
            "review_id": review_id,
            "dataset_sha256": report["dataset_sha256"],
            "spec_sha256": report["spec_sha256"],
            "task": report["task"],
            "dataset": report["dataset"],
            "example_count": report["example_count"],
            "label_counts": report["label_counts"],
            "approved_by": reviewer.strip(),
            "approval_note": approval_note.strip()[:1000],
            "approved_at": _utc_now(),
        },
    )
    _write_bytes(dataset_target, dataset_raw)
    report["status"] = "approved"
    report["approved_by"] = reviewer.strip()
    report["approved_at"] = _utc_now()
    _write_json(review_path, report)
    return _read_json(version_dir / "manifest.json")


def _initialize_store(workspace: Path) -> sqlite3.Connection:
    _secure_directory(workspace)
    connection = sqlite3.connect(workspace / "platform.sqlite3")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS training_runs (
            run_id TEXT PRIMARY KEY,
            dataset_version TEXT NOT NULL,
            task_id TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metrics_json TEXT NOT NULL DEFAULT '{}',
            model_artifact TEXT,
            artifact_sha256 TEXT,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,
            approved_by TEXT NOT NULL,
            approval_note TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES training_runs(run_id)
        );
        """
    )
    connection.commit()
    if os.name == "posix":
        (workspace / "platform.sqlite3").chmod(0o600)
    return connection


def _emit_event(run_dir: Path, run_id: str, state: str, stream: bool = False, **details: Any) -> None:
    event = {"run_id": run_id, "timestamp": _utc_now(), "state": state, **details}
    event_path = run_dir / "events.jsonl"
    _secure_directory(run_dir)
    with event_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n")
        log_file.flush()
        os.fsync(log_file.fileno())
    if os.name == "posix":
        event_path.chmod(0o600)
    if stream:
        print(json.dumps(event, sort_keys=True, ensure_ascii=True), flush=True)


def train_dataset(dataset_version: str, workspace: str | Path, stream_events: bool = False) -> dict[str, Any]:
    """Train a local TF-IDF/logistic-regression candidate and gate it on held-out metrics."""
    try:
        from joblib import dump
        from sklearn.dummy import DummyClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
    except ImportError as error:
        raise RuntimeError("Install training dependencies with: python -m pip install -e '.[training]'") from error

    _validate_id(dataset_version, REVIEW_ID_PATTERN, "dataset version")
    root = Path(workspace).expanduser()
    version_dir = root / "datasets" / dataset_version
    manifest = _read_json(version_dir / "manifest.json")
    if manifest.get("dataset_version") != dataset_version or not manifest.get("approved_by"):
        raise ValueError("Training requires an approved, immutable dataset version.")
    spec_path = version_dir / "spec.json"
    spec_raw = spec_path.read_bytes()
    if _sha256(spec_raw) != manifest.get("spec_sha256"):
        raise ValueError("Approved model specification changed; refusing to train.")
    spec, _ = _load_spec(spec_path)
    dataset_path = version_dir / "dataset.jsonl"
    dataset_raw = dataset_path.read_bytes()
    if _sha256(dataset_raw) != manifest.get("dataset_sha256"):
        raise ValueError("Approved dataset content changed; refusing to train.")
    rows, counts = _parse_dataset(
        dataset_raw,
        spec["task"]["labels"],
        spec["training"]["min_examples_per_label"],
    )

    run_id = uuid.uuid4().hex[:12]
    run_dir = root / "runs" / run_id
    _secure_directory(run_dir)
    connection = _initialize_store(root)
    started_at = _utc_now()
    with connection:
        connection.execute(
            "INSERT INTO training_runs(run_id, dataset_version, task_id, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, dataset_version, spec["task"]["id"], "queued", started_at),
        )
    _emit_event(run_dir, run_id, "queued", stream=stream_events, dataset_version=dataset_version, examples=len(rows))

    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    expected_labels = spec["task"]["labels"]
    try:
        _emit_event(
            run_dir,
            run_id,
            "training",
            stream=stream_events,
            stage="split",
            random_seed=spec["training"]["random_seed"],
        )
        train_validation_text, test_text, train_validation_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=spec["training"]["test_fraction"],
            random_state=spec["training"]["random_seed"],
            stratify=labels,
        )
        validation_share = spec["training"]["validation_fraction"] / (1 - spec["training"]["test_fraction"])
        train_text, validation_text, train_labels, validation_labels = train_test_split(
            train_validation_text,
            train_validation_labels,
            test_size=validation_share,
            random_state=spec["training"]["random_seed"],
            stratify=train_validation_labels,
        )
        candidate = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", max_features=20_000)),
                ("classifier", LogisticRegression(max_iter=1_000, random_state=spec["training"]["random_seed"])),
            ]
        )
        baseline = DummyClassifier(strategy="most_frequent")
        candidate.fit(train_text, train_labels)
        baseline.fit(train_text, train_labels)
        predicted = candidate.predict(test_text)
        validation_predicted = candidate.predict(validation_text)
        baseline_predicted = baseline.predict(test_text)
        accuracy = float(accuracy_score(test_labels, predicted))
        macro_f1 = float(f1_score(test_labels, predicted, labels=expected_labels, average="macro", zero_division=0))
        validation_accuracy = float(accuracy_score(validation_labels, validation_predicted))
        validation_macro_f1 = float(
            f1_score(validation_labels, validation_predicted, labels=expected_labels, average="macro", zero_division=0)
        )
        baseline_accuracy = float(accuracy_score(test_labels, baseline_predicted))
        report = classification_report(
            test_labels,
            predicted,
            labels=expected_labels,
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(test_labels, predicted, labels=expected_labels).tolist()
        per_label_f1 = {label: float(report[label]["f1-score"]) for label in expected_labels}
        per_label_passed = all(value >= spec["training"]["min_per_label_f1"] for value in per_label_f1.values())
        validation_passed = (
            validation_accuracy >= spec["training"]["min_accuracy"]
            and validation_macro_f1 >= spec["training"]["min_macro_f1"]
        )
        beats_baseline = accuracy > baseline_accuracy
        metrics = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "validation_accuracy": validation_accuracy,
            "validation_macro_f1": validation_macro_f1,
            "majority_baseline_accuracy": baseline_accuracy,
            "accuracy_threshold": spec["training"]["min_accuracy"],
            "macro_f1_threshold": spec["training"]["min_macro_f1"],
            "per_label_f1_threshold": spec["training"]["min_per_label_f1"],
            "per_label_f1": per_label_f1,
            "per_label_thresholds_passed": per_label_passed,
            "validation_thresholds_passed": validation_passed,
            "beats_majority_baseline": beats_baseline,
            "passed_thresholds": (
                accuracy >= spec["training"]["min_accuracy"]
                and macro_f1 >= spec["training"]["min_macro_f1"]
                and per_label_passed
                and validation_passed
                and beats_baseline
            ),
            "test_examples": len(test_text),
            "validation_examples": len(validation_text),
            "train_examples": len(train_text),
            "label_order": expected_labels,
            "confusion_matrix": matrix,
            "per_label": {label: report[label] for label in expected_labels},
            "dataset_label_counts": counts,
        }
        _emit_event(run_dir, run_id, "evaluating", stream=stream_events, metrics=metrics)

        artifact_path = run_dir / "model.joblib"
        with tempfile.NamedTemporaryFile(dir=run_dir, suffix=".joblib", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        dump(candidate, temporary_path)
        if os.name == "posix":
            temporary_path.chmod(0o600)
        os.replace(temporary_path, artifact_path)
        artifact_hash = _sha256(artifact_path.read_bytes())
        final_status = "awaiting_approval" if metrics["passed_thresholds"] else "rejected"
        run_record = {
            "run_id": run_id,
            "status": final_status,
            "created_at": started_at,
            "completed_at": _utc_now(),
            "task": spec["task"],
            "dataset_version": dataset_version,
            "dataset_manifest": manifest,
            "recipe": {
                "algorithm": "TfidfVectorizer(1-2 grams) + LogisticRegression",
                "random_seed": spec["training"]["random_seed"],
                "test_fraction": spec["training"]["test_fraction"],
                "python_version": __import__("sys").version.split()[0],
                "scikit_learn_version": __import__("sklearn").__version__,
            },
            "metrics": metrics,
            "model_artifact": "model.joblib",
            "model_artifact_sha256": artifact_hash,
            "limitations": [
                "This is a small text classifier, not a generative language model.",
                "Metrics apply only to this dataset version and held-out split.",
                "Joblib model files can execute code when loaded; load only artifacts from this trusted local workspace.",
            ],
        }
        _write_json(run_dir / "model_card.json", run_record)
        _write_json(run_dir / "metrics.json", metrics)
        with connection:
            connection.execute(
                "UPDATE training_runs SET status = ?, metrics_json = ?, model_artifact = ?, artifact_sha256 = ? WHERE run_id = ?",
                (final_status, json.dumps(metrics, sort_keys=True), str(artifact_path.relative_to(root)), artifact_hash, run_id),
            )
        _emit_event(run_dir, run_id, final_status, stream=stream_events, accuracy=accuracy, macro_f1=macro_f1)
        connection.close()
        return run_record
    except Exception as error:
        with connection:
            connection.execute(
                "UPDATE training_runs SET status = 'failed', error = ? WHERE run_id = ?",
                (type(error).__name__, run_id),
            )
        _emit_event(run_dir, run_id, "failed", stream=stream_events, error_type=type(error).__name__)
        connection.close()
        raise


def promote_run(run_id: str, workspace: str | Path, reviewer: str, approval_note: str) -> dict[str, Any]:
    """Promote a passing candidate only after explicit human approval."""
    _validate_id(run_id, RUN_ID_PATTERN, "training run ID")
    if not reviewer.strip() or not approval_note.strip():
        raise ValueError("A reviewer name and approval note are required.")
    root = Path(workspace).expanduser()
    connection = _initialize_store(root)
    row = connection.execute("SELECT * FROM training_runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None or row["status"] != "awaiting_approval":
        connection.close()
        raise ValueError("Only a passing candidate awaiting approval can be promoted.")
    run_dir = root / "runs" / run_id
    card_path = run_dir / "model_card.json"
    card = _read_json(card_path)
    artifact_path = run_dir / card["model_artifact"]
    if not artifact_path.is_file() or _sha256(artifact_path.read_bytes()) != row["artifact_sha256"]:
        connection.close()
        raise ValueError("Model artifact is missing or does not match its recorded checksum.")
    model_id = f"{card['task']['id']}-{run_id}"
    now = _utc_now()
    with connection:
        connection.execute(
            "INSERT INTO model_registry(model_id, run_id, status, approved_by, approval_note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (model_id, run_id, "active", reviewer.strip(), approval_note.strip()[:1000], now),
        )
        connection.execute("UPDATE training_runs SET status = 'promoted' WHERE run_id = ?", (run_id,))
    card["status"] = "promoted"
    card["model_id"] = model_id
    card["release_approval"] = {"approved_by": reviewer.strip(), "approval_note": approval_note.strip()[:1000], "approved_at": now}
    _write_json(card_path, card)
    _emit_event(run_dir, run_id, "promoted", model_id=model_id, approved_by=reviewer.strip())
    connection.close()
    return card


def get_run(run_id: str, workspace: str | Path) -> dict[str, Any] | None:
    _validate_id(run_id, RUN_ID_PATTERN, "training run ID")
    root = Path(workspace).expanduser()
    connection = _initialize_store(root)
    row = connection.execute("SELECT * FROM training_runs WHERE run_id = ?", (run_id,)).fetchone()
    connection.close()
    if row is None:
        return None
    result = dict(row)
    result["metrics"] = json.loads(result.pop("metrics_json"))
    return result


def list_runs(workspace: str | Path) -> list[dict[str, Any]]:
    root = Path(workspace).expanduser()
    connection = _initialize_store(root)
    rows = list(connection.execute("SELECT * FROM training_runs ORDER BY created_at DESC"))
    connection.close()
    return [{**dict(row), "metrics": json.loads(row["metrics_json"])} for row in rows]


def list_models(workspace: str | Path) -> list[dict[str, Any]]:
    root = Path(workspace).expanduser()
    connection = _initialize_store(root)
    rows = list(
        connection.execute(
            """SELECT model_id, run_id, status, approved_by, approval_note, created_at
               FROM model_registry ORDER BY created_at DESC"""
        )
    )
    connection.close()
    return [dict(row) for row in rows]


def predict(model_id: str, text: str, workspace: str | Path) -> dict[str, str]:
    """Run inference using a promoted local artifact after verifying its recorded checksum."""
    if not text.strip() or len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"Prediction text must contain 1-{MAX_TEXT_CHARS} characters.")
    root = Path(workspace).expanduser()
    connection = _initialize_store(root)
    row = connection.execute(
        """SELECT registry.run_id, runs.artifact_sha256
           FROM model_registry AS registry
           JOIN training_runs AS runs ON runs.run_id = registry.run_id
           WHERE registry.model_id = ? AND registry.status = 'active'""",
        (model_id,),
    ).fetchone()
    connection.close()
    if row is None:
        raise ValueError("No active model with that ID was found.")
    run_dir = root / "runs" / row["run_id"]
    card = _read_json(run_dir / "model_card.json")
    artifact_path = run_dir / card["model_artifact"]
    if not artifact_path.is_file() or _sha256(artifact_path.read_bytes()) != row["artifact_sha256"]:
        raise ValueError("Model artifact is missing or failed its checksum verification.")
    try:
        from joblib import load
    except ImportError as error:
        raise RuntimeError("Install training dependencies with: python -m pip install -e '.[training]'") from error
    model = load(artifact_path)
    label = str(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]
    confidence = float(max(probabilities))
    return {"model_id": model_id, "label": label, "confidence": f"{confidence:.4f}"}


def run_events(run_id: str, workspace: str | Path) -> list[dict[str, Any]]:
    _validate_id(run_id, RUN_ID_PATTERN, "training run ID")
    event_path = Path(workspace).expanduser() / "runs" / run_id / "events.jsonl"
    if not event_path.is_file():
        raise ValueError("No run event log was found for that ID.")
    try:
        return [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Run event log is unreadable.") from error
