# Implementation Plan

This document is the working delivery plan for the Virtual AI Learning Environment. It describes the current code, the target product, how work is sequenced, and the evidence required to call each phase complete.

## 1. What "100%" Means

No finite training system can be verified for every situation or guarantee that an AI is complete, human-like, or always correct. Here, **100% means every required capability in the explicitly scoped product below has met its acceptance criteria**. A new task family, dataset source, deployment environment, or risk level requires its own requirements and evaluation plan; it is not silently covered by the percentage.

The target is a local-first platform that can:

- Capture a creator's intended task, supported labels or output format, user group, permitted data, prohibited behavior, privacy constraints, compute budget, and acceptance thresholds.
- Review, approve, version, and trace datasets before training uses them.
- Run reproducible training for the explicitly supported task families: small supervised text classification; generative supervised fine-tuning with a compatible open model and adapter method; and policies for compatible simulated environments.
- Evaluate a candidate against separate validation and untouched test data, compare it to baselines and approved versions, and block a release when a required gate fails.
- Record runs, metrics, dataset/model lineage, review notes, live events, releases, and rollback state.
- Let the AI student practice in bounded scenes and use separately controlled teacher or conversation-partner models where that is part of the task.
- Protect credentials, datasets, users, and training workers with controls appropriate to the data and deployment risk.

This is a platform for defined tasks, **not** a universal trainer for every model architecture or a guarantee that any model can solve any task.

## 2. Product Boundaries

### Version 1 target

- Local, single-user operation on one machine.
- Local JSONL datasets using `text` and `label` for classification tasks.
- Human review before a dataset version is used or a passing candidate is promoted.
- A reproducible CPU-based classifier workflow with a clear baseline, held-out metrics, and local model notes.
- CLI run visibility and an optional lightweight local dashboard after the command workflow is stable.
- The current conversation simulator remains a practice environment; classifier training and conversational-model inference are separate capabilities.

### Later, separately gated extensions

- Generative-model supervised fine-tuning (for example, LoRA on a compatible open model).
- Reinforcement-learning policies only for environments with explicit observations, actions, and meaningful reward signals.
- Approved dataset catalogs, external storage, concurrent jobs, multiple users, teacher/peer AI agents, and visual simulation.
- Cloud or multi-GPU training, production serving, and third-party data providers.

Do not enable an extension just because a library supports it. Add it only when a creator requirement, supported model, test plan, resource budget, and security review exist.

## 3. Current Implementation Snapshot

As of this plan, the repository already contains these building blocks:

| Area | Implemented now | Still missing for the target |
| --- | --- | --- |
| Practice environment | Four text scenes, template student, optional chat-completions adapter, rubric feedback, profile-scoped SQLite memory, profile deletion | Teacher/partner agents, richer curriculum, sharing controls and model feedback integrated into scenario training |
| Task specification | A JSON specification declares a text-classification task, labels, dataset description, and metric thresholds | Full validated specification for intended users, prohibited uses, privacy, risk, acceptance cases, and resource budget |
| Dataset intake | Local UTF-8 JSONL validation, declared-label counts, size limits, exact normalized duplicate check, email/phone pattern scan, review report, human approval, source checksums | General source registry, robust PII review, group/near-duplicate and contamination checks, transformation history, retention workflow, split management |
| Dataset storage | Approved copies under a local workspace with a manifest and restrictive local file permissions | Separate raw/quarantine/curated stores, encryption/key management, signatures, role-based access, remote object storage |
| Training | Optional scikit-learn TF-IDF plus logistic-regression classifier on CPU | Training-worker abstraction, retry/cancel, resource limits, checkpoints, generative-model training, RL training |
| Verification | Stratified train/validation/test split, majority baseline, accuracy, macro-F1, per-label F1, configured gates | Repeated-seed uncertainty, group split, calibration, robustness, regression suites, task-specific safety/privacy testing, independent test governance |
| Runs and registry | SQLite run metadata, JSON model card and metrics, JSONL events, explicit promotion, artifact checksum check before local inference | Full reproducibility metadata, active-version policy, model comparison, rollback/deactivation, artifact signing, access/audit history |
| Live status | Training events stream as JSONL in the CLI and are retained per run | Web dashboard, queued background jobs, cancellation/retry, charts, user access control, alerts |
| Security | Local permission tightening, no generated-code execution, API key from environment, explicit memory-sharing switch | Hardened worker isolation, egress policy, authentication/authorization, encrypted data, secret manager, robust PII and prompt-injection defenses |

The example dataset is synthetic and intentionally small. Its metrics demonstrate that the workflow runs; they are not evidence of useful real-world language understanding.

## 4. End-to-End Workflow

```text
Creator task specification
  -> dataset candidates from approved sources
  -> review report and human data approval
  -> immutable dataset + specification version
  -> base-model or baseline evaluation
  -> frozen training run
  -> validation, untouched test, regression, privacy, and safety checks
  -> candidate waiting for human release approval
  -> versioned registry entry and model notes
  -> staged deployment, monitoring, rollback, and retention
```

Each arrow is a persisted transition with an identifier, timestamp, owner where applicable, and a failure state. Failed transitions must leave inputs unchanged and must not silently promote, overwrite, or discard a version.

## 5. Phased Delivery Plan

Phase weights total 100 points. They are a **scope accounting tool**, not a claim that the AI itself is 100% capable. A phase earns its weight only when every mandatory acceptance item in that phase is demonstrated. Partial work is reported as partial and is not counted as complete.

### Phase 0: Product and task contract — 8 points

**Status:** Partial. The example classification spec is a starting point, not the complete creator contract.

**Work:** Define a versioned schema for task type, intended use, supported users/languages, inputs/outputs, allowed and disallowed behavior, source-data policy, privacy/risk class, compute budget, model baseline, thresholds, reviewers, and out-of-scope behavior. Validate specs before a dataset can be reviewed.

**Small case:** A creator requests English classroom-intent classification with four labels, forbids storing student names, and sets a minimum per-label F1. The tool rejects a spec missing its data-use statement or containing an unknown metric.

**Done when:** Schema and migration are versioned; invalid and incomplete examples fail with actionable errors; a saved task spec is included in every dataset manifest and training run.

### Phase 1: Dataset source and approval — 12 points

**Status:** Partial. Local JSONL review and explicit approval exist; catalog discovery and a full source record do not.

**Work:** Add source records for owner, origin, license, consent/legal basis, collection date, allowed use, attribution, retention, language, and known limitations. Keep source selection separate from AI ranking. AI may rank approved catalog metadata; only an authorized person can approve actual use. Keep imported data read-only in quarantine.

**Small case:** A candidate dataset has a license that disallows commercial use. The system may display/rank it for a non-commercial research task but must prevent approval for a commercial deployment.

**Done when:** Every approved dataset version has a reviewed provenance record, an explicit allowed-use match, an approver, and a reason; a failed review cannot be used by a training command.

### Phase 2: Data quality, privacy, and immutable versions — 10 points

**Status:** Partial. Schema, size, duplicate, and simple contact-pattern checks exist; these checks are not a comprehensive privacy scanner.

**Work:** Add reviewable transformation steps, language/label balance summaries, exact and near-duplicate reports, PII review, group identifiers for related examples, split-leakage prevention, deletion/retention actions, and a version manifest that covers source hashes, spec hash, transformations, splits, and approvals. Preserve raw inputs separately and never train directly from quarantine.

**Small case:** Two differently punctuated versions of the same sentence are reported as possible duplicates. A phone-like string is quarantined for human review; a test-set conversation group is prevented from appearing in training.

**Done when:** Approved versions are immutable and content-verified; every transformation is reproducible; data grouped by source/conversation cannot leak across splits; deletion policy is testable; known PII limitations are clearly reported.

### Phase 3: Reproducible training backends — 16 points

**Status:** Partial. A CPU text-classification backend exists; there is no worker queue, generative fine-tuning, or RL backend.

**Work:** Define a backend protocol with explicit input/output contracts, run config, seed, resource limits, status events, artifact output, and failure handling. Complete the classifier runner first. Then add separate, optional backends for generative SFT/LoRA and compatible Gymnasium RL environments only after their requirements and evaluations are defined. Do not make one generic training function pretend to support every model type.

**Small case:** The same approved dataset, code version, and seed produce the same split and equivalent metrics on repeat CPU runs. An unsupported model/task combination fails before allocating resources.

**Done when:** Each supported trainer has deterministic configuration capture, input checks, scoped read/write access, bounded resources, explicit failure/retry behavior, and a passing integration test. Model type, algorithm, and hardware compatibility are validated before start.

### Phase 4: Verification and release gates — 16 points

**Status:** Partial. The classifier has validation/test metrics, a simple baseline, and aggregate/per-label thresholds; test samples are small and not yet an independent governance boundary.

**Work:** Freeze evaluator versions and test sets. Add baseline and prior-release comparison, group-aware splits, repeated seeds or confidence intervals where sample counts permit, calibration/abstention, task-specific adversarial and privacy tests, regression cases, and per-class review. Keep validation data for tuning and the test set untouched until final evaluation. Failure at any mandatory gate blocks promotion.

**Small case:** A candidate scores well overall but has zero F1 for the `question` label. The report identifies that class failure and the candidate is rejected. A model that reveals a planted secret fails privacy verification even if its task score is high.

**Done when:** Every task spec names its evaluator and thresholds; results are reproducible from exact versions; all required gates pass before release; false passes and false failures have tests; reviewers can see failures and uncertainty rather than only one aggregate score.

### Phase 5: Model registry, notes, and rollback — 10 points

**Status:** Partial. Passing classifiers can be manually promoted and get a model card; registry lifecycle and rollback are not complete.

**Work:** Add model ID/version policy, immutable lineage, signed or otherwise integrity-protected release metadata, active/candidate/retired states, comparison view, deployment references, deactivation, rollback, and model-card review. Keep model artifacts separate from dataset and run metadata. Treat joblib/pickle artifacts as executable and load only trusted artifacts; use safer serialization for future model families where available.

**Small case:** A newly promoted model causes a regression. An authorized operator restores the previous model version and the registry records who rolled back, when, and why.

**Done when:** Each active model traces to one exact run, data version, spec, code revision, and evaluation; checksum/signature failures prevent loading; promotion and rollback are authorized and audited; deleting a dataset follows the defined retention/legal policy without corrupting lineage.

### Phase 6: Live run operations — 9 points

**Status:** Partial. The CLI emits and saves JSONL events for synchronous local runs; no web dashboard or job queue exists.

**Work:** Define event schema and state machine; add queue, cancel, retry, worker health, progress/ETA where estimable, resource use, sanitized errors, and artifact readiness. Build a local operator view only after the event contract is stable. Add notifications for failed or blocked runs.

**Small case:** A run on a 1,000-row demo set shows data validation, split, training, evaluation, and approval states; an invalid license stays blocked at review and never appears as training progress.

**Done when:** Events are ordered, persisted, queryable, and contain no secrets/raw examples; restart behavior is understood; cancellation and failure leave consistent state; the dashboard matches authoritative run records.

### Phase 7: Student environment and helper agents — 9 points

**Status:** Partial. A small scene simulator and optional language-model inference exist; no dedicated teacher/partner agents or training-feedback loop exists.

**Work:** Formalize scene/task interfaces, store observations/actions/feedback distinctly, and separate general lessons from user preferences. Add teacher, partner, and evaluator roles behind a backend gateway with minimal context and fixed permissions. Obtain explicit consent before storing preference memory or reusing conversations as training data. Keep environment memory separate from model-weight updates.

**Small case:** In an asking-for-help scene, a teacher explains a missing polite request, but does not receive another user's private preferences. A new unseen scene checks whether the student transfers the skill.

**Done when:** Role permissions are tested; no helper can expand its own access; feedback has provenance and can be reviewed; per-user memory can be inspected/deleted; held-out transfer is measured separately from memorized dialogue.

### Phase 8: Security, privacy, and operational controls — 7 points

**Status:** Partial. Local file permissions and environment-held API keys exist; there is no hardened sandbox or production security boundary.

**Work:** Threat-model dataset import, model files, API providers, helper-agent prompts, and users. Add least-privilege identities, secret management, storage encryption, network egress allowlists, rate/cost budgets, audit events, retention/deletion, backup/restore, and tested worker isolation. Do not treat prompts, keyword PII scans, hashes, or ordinary containers as complete security controls.

**Small case:** A dataset row contains an instruction to reveal secrets. It remains ordinary untrusted training text and cannot grant filesystem or API access. A remote endpoint outside the allowlist is blocked before a request is made.

**Done when:** Security boundaries are enforced outside the model; sensitive data flows are documented; access and deletion controls are tested; restore and incident procedures exist; an independent review closes critical findings for the deployment's risk level.

### Phase 9: Release, maintenance, and support — 3 points

**Status:** Partial. A README and unit/integration tests exist; no versioned release process or operational support policy exists.

**Work:** Add CI across supported Python versions, dependency update and vulnerability checks, database/schema migration tests, backup/restore verification, release notes, operator runbook, user-facing limitations, and a compatibility/deprecation policy.

**Small case:** A release candidate is blocked if the dataset approval, training, model promotion, or rollback integration test fails on the supported Python version.

**Done when:** A clean install reproduces the documented workflow; the test suite and security checks pass in CI; migrations and rollback are exercised; the release has an owner, support boundary, and recovery instructions.

## 6. Technology Plan

Choose the smallest technology that meets the phase's acceptance criteria. Avoid operating a distributed platform before local measurements show that one is needed.

| Concern | Version 1 choice | Add/replace when justified |
| --- | --- | --- |
| Task specification | Versioned JSON schema; Pydantic validation is a reasonable implementation option | Schema migrations and compatibility tests as spec versions grow |
| Small supervised trainer | Existing scikit-learn TF-IDF/logistic-regression backend | PyTorch/Transformers only for model families that require neural training |
| Dataset handling | Local JSONL and content hashes | Hugging Face Datasets/PyArrow for larger formats; DVC or equivalent when remote data versioning becomes necessary |
| Generative fine-tuning | Not implemented; separate backend milestone | Transformers + PEFT/LoRA; TRL only for a defined supported training objective and compatible model |
| Simulated RL | Existing scenes are not yet a Gymnasium environment | Gymnasium API plus Stable-Baselines3 for a compatible policy-learning task; PettingZoo only for genuinely multi-agent learning |
| Metadata and artifacts | SQLite, JSON manifests/model cards, local workspace | PostgreSQL and controlled object storage for concurrent users/artifacts |
| Registry and tracking | Current local registry is a prototype | Adopt MLflow if it replaces rather than duplicates the project registry/lineage source of truth |
| Run processing | Synchronous local CLI with JSONL events | A persistent queue/worker only when concurrency or duration requires it |
| Live view | CLI status/events | FastAPI + server-sent events and a small operator UI after event/schema stability |
| Isolation | No hardened OS sandbox | Rootless container/VM or gVisor, restricted mounts, no Docker socket, and default-deny network policy before untrusted jobs |
| Secret storage | Environment variables for local development only | A secret manager and provider gateway for shared/production deployment |

## 7. Completion Tracking

Use the phase weights in Section 5. The project owner updates each phase to `not started`, `in progress`, `blocked`, or `done`, with links to its tests, release artifacts, and review evidence. **Only `done` phases earn their points.** Do not estimate from file count, lines of code, or the number of tools installed.

At every release, publish:

- Required capabilities in scope and explicitly out of scope.
- Dataset/model/run versions used for the verification report.
- Per-task and per-label metrics, baselines, thresholds, and failed tests.
- Known privacy, safety, quality, and operational limitations.
- Human approvals, deployment version, rollback target, and data-retention status.

The project reaches 100% for this plan only when all weighted phases are done for a named release and all mandatory acceptance tests pass. Adding a new task family or deployment risk level creates new work and resets the relevant completion gates; it does not imply that prior results were wrong.

## 8. Recommended Work Order

1. Finish Phase 0's specification schema and agree on the supported V1 task set.
2. Close Phase 1 and Phase 2 before accepting any non-synthetic or personal dataset.
3. Harden the classifier workflow in Phase 3 and Phase 4; keep synthetic examples for CI and require separate reviewed data for real training.
4. Complete registry, notes, and rollback before serving a candidate outside the local developer machine.
5. Add live dashboard/queue only after event contracts and state transitions are stable.
6. Integrate teacher/partner agents and broader model trainers as separate, explicitly approved task families.
7. Complete security and release gates before multi-user or production use.

At each step, deliver a narrow vertical slice with one successful case, one rejection/failure case, automated tests, and updated documentation. Do not move a failed verification gate forward by lowering thresholds after seeing the result; revise the task specification first, document the reason, and rerun against an untouched test set.
