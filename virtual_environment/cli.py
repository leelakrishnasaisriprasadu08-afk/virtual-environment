"""Command-line interface for the text-only learning environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .environment import LearningEnvironment
from .memory import StudentMemory, validate_profile_id
from .models import build_model
from .scenes import SCENES
from .training import (
    approve_dataset,
    get_run,
    list_models,
    list_runs,
    predict,
    promote_run,
    review_dataset,
    run_events,
    train_dataset,
)


def default_database_path() -> Path:
    configured = os.environ.get("VIRTUAL_ENV_DATA")
    if configured:
        return Path(configured).expanduser()
    data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return data_home / "virtual-environment" / "student.sqlite3"


def default_training_workspace() -> Path:
    state_home = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return state_home / "virtual-environment" / "training"


def _print_progress(memory: StudentMemory, profile_id: str) -> None:
    rows = memory.progress(profile_id)
    if not rows:
        print("No practice recorded yet.")
    else:
        print("\nProgress")
        for row in rows:
            status = "passed" if row["passed"] else "in progress"
            print(f"- {row['scene_id']}: {status}, {row['attempts']} attempt(s), average {row['average_score']:.2f}")
    preferences = memory.preferences(profile_id)
    if preferences:
        print("\nStored preferences (local to this profile)")
        for key, value in preferences:
            print(f"- {key}: {value}")


def _practice(environment: LearningEnvironment, memory: StudentMemory, profile_id: str, scene_id: str, share_memory: bool) -> None:
    result = environment.practice(profile_id, scene_id, share_memory)
    print(f"\nScene: {result.scene.topic}")
    print(f"Student: {result.response}")
    print(f"Score: {result.score:.0%} | {'PASS' if result.passed else 'RETRY'}")
    print(result.feedback)
    if result.passed and result.scene.preference_candidate:
        key, value = result.scene.preference_candidate
        confirmation = input(f"Save the observed preference '{value}' to profile '{profile_id}'? [y/N] ").strip().casefold()
        if confirmation == "y":
            memory.set_preference(profile_id, key, value)
            print("Preference saved locally.")


def _menu(environment: LearningEnvironment, memory: StudentMemory, profile_id: str, share_memory: bool) -> None:
    while True:
        print("\n1. Practice next scene")
        print("2. Choose a scene")
        print("3. Add or update a profile preference")
        print("4. View progress and preferences")
        print("5. Delete this profile's data")
        print("6. Quit")
        choice = input("Select: ").strip()

        if choice == "1":
            rows = {row["scene_id"]: row["passed"] for row in memory.progress(profile_id)}
            scene = next((item for item in SCENES if not rows.get(item.scene_id, False)), SCENES[0])
            _practice(environment, memory, profile_id, scene.scene_id, share_memory)
        elif choice == "2":
            for index, scene in enumerate(SCENES, 1):
                print(f"{index}. {scene.topic} ({scene.scene_id})")
            selected = input("Scene number: ").strip()
            if selected.isdigit() and 1 <= int(selected) <= len(SCENES):
                scene = SCENES[int(selected) - 1]
                _practice(environment, memory, profile_id, scene.scene_id, share_memory)
            else:
                print("Choose one of the listed scene numbers.")
        elif choice == "3":
            key = input("Preference key: ").strip()
            value = input("Preference value: ").strip()
            try:
                memory.set_preference(profile_id, key, value)
                print("Preference saved locally.")
            except ValueError as error:
                print(error)
        elif choice == "4":
            _print_progress(memory, profile_id)
        elif choice == "5":
            print("This permanently removes this profile's local preferences and practice history.")
            if input(f"Type DELETE to confirm for profile '{profile_id}': ").strip() == "DELETE":
                attempts, preferences = memory.delete_profile(profile_id)
                print(f"Deleted {attempts} practice record(s) and {preferences} preference(s).")
            else:
                print("Deletion cancelled.")
        elif choice == "6":
            return
        else:
            print("Choose a number from 1 to 6.")


def _add_workspace_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", type=Path, default=default_training_workspace())


def _training_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        if args.command == "dataset-review":
            result = review_dataset(args.spec, args.dataset, args.workspace)
        elif args.command == "dataset-approve":
            result = approve_dataset(
                args.review_id,
                args.spec,
                args.dataset,
                args.workspace,
                args.reviewer,
                args.note,
            )
        elif args.command == "train":
            result = train_dataset(args.dataset_version, args.workspace, stream_events=True)
        elif args.command == "promote":
            result = promote_run(args.run_id, args.workspace, args.reviewer, args.note)
        elif args.command == "run-status":
            result = get_run(args.run_id, args.workspace)
            if result is None:
                parser.error(f"No run found with ID '{args.run_id}'.")
        elif args.command == "runs":
            result = list_runs(args.workspace)
        elif args.command == "events":
            result = run_events(args.run_id, args.workspace)
        elif args.command == "models":
            result = list_models(args.workspace)
        elif args.command == "predict":
            result = predict(args.model_id, args.text, args.workspace)
        else:
            parser.error("Unknown command.")
            return
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
        return
    if args.command == "train":
        print(json.dumps(result, sort_keys=True, ensure_ascii=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Practice language and concepts in a bounded simulated environment.")
    parser.add_argument("--model", choices=("demo", "api"), default=os.environ.get("VIRTUAL_ENV_MODEL", "demo"))
    parser.add_argument("--profile", default="student", help="Local profile ID (letters, numbers, '_' and '-')")
    parser.add_argument("--data", type=Path, default=default_database_path(), help="Local SQLite database path")
    parser.add_argument(
        "--share-memory-with-model",
        action="store_true",
        help="Explicitly allow stored lessons and preferences to be sent to the configured model endpoint",
    )
    commands = parser.add_subparsers(dest="command")

    review_parser = commands.add_parser("dataset-review", help="Validate a JSONL dataset and create a review report")
    review_parser.add_argument("--spec", type=Path, required=True)
    review_parser.add_argument("--dataset", type=Path, required=True)
    _add_workspace_argument(review_parser)

    approval_parser = commands.add_parser("dataset-approve", help="Approve reviewed files and freeze a dataset version")
    approval_parser.add_argument("--review-id", required=True)
    approval_parser.add_argument("--spec", type=Path, required=True)
    approval_parser.add_argument("--dataset", type=Path, required=True)
    approval_parser.add_argument("--reviewer", required=True)
    approval_parser.add_argument("--note", required=True)
    _add_workspace_argument(approval_parser)

    train_parser = commands.add_parser("train", help="Train and evaluate a classifier on an approved dataset version")
    train_parser.add_argument("--dataset-version", required=True)
    _add_workspace_argument(train_parser)

    promote_parser = commands.add_parser("promote", help="Promote a passing candidate after human review")
    promote_parser.add_argument("--run-id", required=True)
    promote_parser.add_argument("--reviewer", required=True)
    promote_parser.add_argument("--note", required=True)
    _add_workspace_argument(promote_parser)

    status_parser = commands.add_parser("run-status", help="Show the status and metrics for one run")
    status_parser.add_argument("--run-id", required=True)
    _add_workspace_argument(status_parser)

    runs_parser = commands.add_parser("runs", help="List training runs")
    _add_workspace_argument(runs_parser)

    events_parser = commands.add_parser("events", help="Show structured events for a run")
    events_parser.add_argument("--run-id", required=True)
    _add_workspace_argument(events_parser)

    models_parser = commands.add_parser("models", help="List approved model versions")
    _add_workspace_argument(models_parser)

    predict_parser = commands.add_parser("predict", help="Run inference with an approved local classifier")
    predict_parser.add_argument("--model-id", required=True)
    predict_parser.add_argument("--text", required=True)
    _add_workspace_argument(predict_parser)

    args = parser.parse_args()

    if args.command:
        _training_command(parser, args)
        return

    try:
        profile_id = validate_profile_id(args.profile)
        student = build_model(args.model)
    except ValueError as error:
        parser.error(str(error))

    memory = StudentMemory(args.data)
    try:
        environment = LearningEnvironment(memory, student)
        print("Virtual AI Learning Environment")
        print(f"Model: {student.name} | Profile: {profile_id}")
        print(f"Local memory: {memory.path}")
        if args.model == "demo":
            print("The template baseline runs locally and makes no model API calls.")
        elif args.share_memory_with_model:
            print("Stored lessons and preferences may be sent to the configured model provider.")
        else:
            print("Only the current curated scene is sent to the configured model provider; stored memory stays local.")
        _menu(environment, memory, profile_id, args.share_memory_with_model)
    except (EOFError, KeyboardInterrupt):
        print("\nSession ended.")
    except RuntimeError as error:
        print(f"\n{error}")
    finally:
        memory.close()


if __name__ == "__main__":
    main()
