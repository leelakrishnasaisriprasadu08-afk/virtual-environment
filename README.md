# Virtual AI Learning Environment

An experimental, bounded text simulation where an AI student practices through a structured sequence of scenes. The student can observe example conversations, respond, receive feedback, and build a limited local history of practice.

The goal is to study whether interaction and feedback help an AI use what it already knows in new situations. This project does not claim to create a person or a complete human-like intelligence.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the phased product scope, completion criteria, example cases, and evidence-based 100-point progress model.

## The Idea

A pretrained language model begins with broad language ability. The environment gives it a persistent student role and a curriculum: scenes have topics, participants, goals, and rules. The student observes example conversations, responds in practice, and gets feedback from a teacher or evaluator. Simulated events can also teach concepts through scenarios and consequences.

This complements large-scale model training; it does not replace it. A model API or local pretrained model supplies language capability, while the environment controls practice, feedback, state, and evaluation. At first, experiences should change the scenario and the student's explicit memory, not silently update the model's weights. Any future fine-tuning should use reviewed, consented data in a separate offline process.

## Quick Start

Requires Python 3.11 or later. The demo uses only the Python standard library and makes no network requests.

```sh
python3 -m virtual_environment.cli
```

The default template student follows fixed responses so you can try the scenes, scoring, and local memory workflow without model credentials. It is a baseline for the environment, not a trained AI model.

## Train a Small Example Model

The optional training workflow uses a synthetic, locally bundled dataset to train a text-intent classifier. It is **not** LLM fine-tuning. It demonstrates review, approval, training, verification, run tracking, registry promotion, and inference. Install the optional dependency:

```sh
python3 -m pip install -e '.[training]'
```

Review and inspect the dataset report. Automated checks do not approve its license or intended use:

```sh
python3 -m virtual_environment.cli dataset-review \
	--spec examples/intent-classification.spec.json \
	--dataset examples/dialogue-intents.jsonl \
	--workspace ./training-workspace
```

Use the `review_id` from the report to approve and freeze the exact dataset and task specification. The approval note is required:

```sh
python3 -m virtual_environment.cli dataset-approve \
	--review-id REVIEW_ID \
	--spec examples/intent-classification.spec.json \
	--dataset examples/dialogue-intents.jsonl \
	--reviewer "Your Name" \
	--note "Reviewed the synthetic demo data and task scope." \
	--workspace ./training-workspace
```

Use the returned `dataset_version` to train. The command streams newline-delimited JSON state and metric events; a run passing the configured gates enters `awaiting_approval`, not automatic release:

```sh
python3 -m virtual_environment.cli train \
	--dataset-version DATASET_VERSION \
	--workspace ./training-workspace
```

Inspect the run, then explicitly promote it after human review. Replace `RUN_ID` with the ID in the training output:

```sh
python3 -m virtual_environment.cli run-status --run-id RUN_ID --workspace ./training-workspace
python3 -m virtual_environment.cli promote \
	--run-id RUN_ID \
	--reviewer "Your Name" \
	--note "Reviewed validation, test, and per-label metrics." \
	--workspace ./training-workspace
python3 -m virtual_environment.cli models --workspace ./training-workspace
python3 -m virtual_environment.cli predict \
	--model-id MODEL_ID \
	--text "Could you please help me?" \
	--workspace ./training-workspace
```

Workspace files default to `~/.local/state/virtual-environment/training/` when `--workspace` is omitted. The example data is synthetic and too small for meaningful claims about real-world language understanding.

Run the tests with:

```sh
python3 -m unittest discover -s tests -v
```

To use an OpenAI-compatible chat-completions endpoint, set its API key in the process environment and run with `--model api`:

```sh
export MODEL_API_KEY="your-key"
export MODEL_NAME="your-model-name"
export MODEL_BASE_URL="https://your-provider.example/v1"
python3 -m virtual_environment.cli --model api
```

The endpoint must use HTTPS unless it is on localhost. The configured provider receives the current curated scene. Stored preferences and successful feedback stay local unless `--share-memory-with-model` is supplied. Review your provider's data handling before using private content. Never put API keys in source files, prompts, or student memory.

Useful options:

- `--profile student-1` keeps separate local progress for a profile ID.
- `--data /path/to/student.sqlite3` selects the SQLite file; by default it is under `~/.local/share/virtual-environment/`.
- Choose **Delete this profile's data** in the menu and type `DELETE` to remove that profile's practice and preferences.

## Implemented MVP and Architecture

- **World and curriculum:** Topic-based scenes define the situation, available actions, participants, and expected outcomes. A simulated timeline can provide continuity without pretending the model has a real life.
- **AI student:** A pretrained model receives only the current scene and the minimum relevant memory. Its model provider is hidden behind a backend adapter so the project can change providers or use a local model.
- **Teacher and conversation partners:** The MVP includes example dialogue and a deterministic rubric evaluator. Separate teacher and conversation-partner model agents are a later step; an evaluator should never be treated as an unquestionable source of truth.
- **Memory:** Keep session experiences, general lessons, and user-specific preferences as separate records. User-specific memory should be isolated, inspectable, and removable; it should not automatically become shared training data.
- **Evaluation and audit:** The MVP records scene outcomes and feedback locally. Detailed model-version and data-access audit trails remain future work.

## Implemented Today

The current prototype includes a text-only practice environment with four scenes, an optional OpenAI-compatible inference adapter, rubric-style feedback, and profile-scoped SQLite memory. Its separate training workflow reviews local `text`/`label` JSONL against a creator-authored task specification, freezes reviewed inputs by checksum, trains a TF-IDF/logistic-regression classifier, measures validation and test metrics, emits JSONL run events, stores a model card, and requires a person to promote passing candidates.

This does **not** train or fine-tune a generative language model, search external dataset catalogs, run distributed/GPU jobs, or provide a live web dashboard. The classifier run and its held-out metrics verify only this narrow synthetic task; they are not model-quality or safety certification.

The app does not execute model-generated code or give the student tools or filesystem access. It is an application-level simulation, not a hardened OS sandbox. API mode makes requests to its configured endpoint. Do not use private or sensitive datasets with this prototype.

## Efficient Platform Workflow

Build the training platform as gated stages. Keep the first training workflow local and single-worker; add distributed queues and infrastructure only when measured job volume requires them.

1. **Creator defines a model specification.** Record intended tasks, languages, users, allowed and prohibited behavior, permitted data sources, privacy constraints, compute budget, and measurable acceptance thresholds. Validate the specification before dataset discovery.
2. **Discover and review datasets.** Search only approved catalogs. An AI helper may rank candidate dataset metadata against the creator's specification, but a person approves each source, license, consent basis, and intended use. Never let the helper autonomously download arbitrary web data into training.
3. **Ingest into quarantine.** Keep the original read-only, record its source, license, owner, collection date, checksum, and access terms. Scan schema, language, quality, duplicates, personal information, and unsafe content. Reject or quarantine data that fails policy.
4. **Create an immutable dataset version.** Store a manifest with transformation steps, checksums, and train/validation/test split identifiers. Prevent test examples and near-duplicates from leaking into training. A creator reviews the dataset summary and approves the exact version.
5. **Run a baseline before training.** Evaluate the selected base model against the approved test plan. Save the model ID, prompt/configuration, dataset version, code revision, seed, and baseline metrics.
6. **Train from a frozen run configuration.** A job reads only approved, versioned inputs and writes checkpoints and structured progress events. Start with a pretrained model and a parameter-efficient method such as LoRA when appropriate; full foundation-model training is a separate, substantially more expensive project.
7. **Evaluate the candidate before promotion.** Run held-out task, regression, safety, privacy, and data-leakage checks. Compare with the baseline and the currently approved model. A failed gate blocks promotion; a person reviews results and explicitly approves the candidate.
8. **Register, deploy, and monitor.** Store the approved model and its complete lineage in a registry. Deploy a versioned candidate, monitor quality and operational metrics, support rollback, and retain or delete artifacts according to policy.

## Dataset, Training, and Storage Flow

```text
Creator specification
	-> approved catalog search (AI may rank; person approves)
	-> read-only raw data in quarantine
	-> validation, redaction, deduplication, leakage checks
	-> approved immutable dataset version
	-> isolated training worker + frozen run configuration
	-> checkpoints, metrics, logs, and resource events
	-> independent evaluation gates
	-> human-approved model registry entry
	-> versioned deployment and monitoring
```

Keep raw datasets, curated dataset versions, run metadata, and model artifacts in separate storage locations with separate access policies. Training workers receive read-only access to the approved dataset version and write access only to their own run directory. The live view should show counts, hashes, statuses, and metrics, not raw training examples or secrets.

Each training run should be reproducible from a manifest containing: run ID; creator specification version; base model and revision; dataset IDs, versions, and checksums; preprocessing version; train/validation/test split; code revision; tokenizer; hyperparameters; random seed; hardware; start/end times; checkpoint references; evaluation results; and approval state. Do not mutate a completed run; create a new run when inputs or settings change.

## Model Verification and Release Gates

Verification means evidence against a defined intended-use specification, not proof that a model works in every possible situation. Define tests and thresholds before training. At minimum, each candidate should have:

- **Data checks:** provenance and license review, schema and quality checks, duplicate and split-leakage checks, and privacy review.
- **Task checks:** held-out examples and scenario tests covering each declared capability; report per-task results, not only one average score.
- **Regression checks:** compare with the base model and the last approved model on the same fixed test sets.
- **Safety and privacy checks:** test prohibited behavior and attempts to reveal training or user data; record failures and their severity.
- **Operational checks:** latency, throughput, memory, failure rate, and cost under the expected workload.
- **Human review:** inspect representative outputs and high-impact failures before approval.

Use thresholds appropriate to the task and risk. Keep test data separate from training and tuning data. Record the exact evaluator versions and test-set versions so that later results can be compared fairly. A passing score applies only to the tested scope; show untested or uncertain areas clearly.

## Live Run View and Model Notes

Represent a run with explicit states such as `queued`, `validating_data`, `training`, `evaluating`, `awaiting_approval`, `completed`, `failed`, and `cancelled`. Stream structured events with timestamps, run ID, step, training and validation metrics, checkpoint availability, elapsed time, hardware use, and sanitized errors. Keep secrets and raw samples out of events and logs.

For each model version, maintain a readable model card or run note containing intended use and out-of-scope uses; base model; dataset lineage and licenses; training recipe; evaluation scope and results; known limitations; safety/privacy findings; reviewers; approval status; and deployment/rollback history. Make it possible to compare two versions and trace each metric to its run and data versions.

## Technology Choices

| Need | Start with | Add when needed |
| --- | --- | --- |
| Current practice environment | Existing Python scene classes, CLI, and SQLite | Gymnasium when a formal single-agent RL `reset`/`step` contract is useful; it is an environment API, not a language trainer. |
| Model inference | Existing template baseline or OpenAI-compatible adapter | Transformers with PyTorch for local inference; keep provider APIs behind a backend adapter. |
| Dataset ingestion | Reviewed local JSONL with a task specification and checksum manifest | Hugging Face Datasets and PyArrow for larger structured datasets; DVC for dataset versioning and remote artifact references. |
| Dataset approval | Human-reviewed source and license checklist | A catalog UI that can use an AI helper to rank approved metadata, while leaving approval to a person. |
| Small text-classifier training | Optional scikit-learn TF-IDF + logistic regression pipeline | Use as a lightweight dataset/run workflow example, not as a conversational model. |
| Generative-model fine-tuning | Not implemented | Transformers + PEFT/LoRA; TRL when a supported supervised or preference-training method is selected. A generic chat API is not a training pipeline. |
| Run tracking and model registry | Local SQLite metadata, JSON model cards, and JSONL event logs | MLflow for shared run metadata, metrics, artifacts, and model versions when concurrent users need it. |
| Verification | Separate validation/test splits, majority baseline, per-class F1, thresholds, and explicit human promotion | `lm-evaluation-harness` for compatible standardized evaluations plus project-specific held-out, regression, safety, and privacy tests. |
| Live run status | Training events stream to CLI and are stored as JSONL | FastAPI with server-sent events for a web dashboard when there is a UI to serve. |
| Metadata and artifacts | Local SQLite and workspace files with checksums | PostgreSQL for concurrent job/registry metadata and controlled object storage for datasets/checkpoints when multi-user workloads require it. |
| Isolation | No OS-level sandbox in this prototype | Least-privilege workers, restricted mounts and egress, and hardened container or VM isolation before processing untrusted data or code. |
| Multiple simulation agents / 3D | Not implemented | PettingZoo for multi-agent RL or Unity ML-Agents for a real 3D simulation, only if the research question requires them. |

Avoid adding Kubernetes, distributed training, a message broker, or a large dashboard stack to the first local training workflow. First measure job duration, concurrency, artifact size, and monitoring needs; scale the part that becomes a bottleneck.

## Privacy and Sandbox Boundaries

- Keep source datasets outside the student's runtime. A scenario service should provide only curated, minimized examples needed for the current lesson.
- Treat conversations and imported documents as untrusted input. They can contain instructions intended to manipulate the model, so do not let text grant new permissions or bypass policy.
- The demo model makes no network requests. API mode sends scene context only to the configured endpoint; a production deployment should route calls through a gateway with provider allowlists, request-size limits, quotas, and audit logs.
- Store API keys only in backend secret storage. Never put keys in prompts, student memory, source control, or files readable by the agent.
- Redact personal information where possible, obtain consent before using conversations, and define deletion and retention policies. Check a model provider's data-retention terms before sending private content.
- Use least privilege for every helper agent. A dialogue partner should not receive private preference records unless they are needed and authorized for that scene.
- Treat prompt instructions as guidance, not a security boundary. Enforce access controls in the surrounding software and test for unintended disclosure.

## Measuring Progress

Evaluate with held-out scenes and track:

- Success on the scene's stated goal and the quality of language use.
- Transfer to new situations that share a concept but not the example wording.
- Accuracy and appropriateness when using explicitly stored preferences.
- Privacy checks, including whether hidden or unrelated dataset content appears in outputs.
- The current prototype records scene outcomes, feedback, and profile-specific preferences in SQLite; it does not collect training-run telemetry or model verification results.
- Reliability, latency, and cost across repeated runs.

Compare environment practice against a baseline model given the same task without the practice history. This helps show whether the environment adds value beyond the model's pretrained capability.

## Limitations

Simulated practice can improve measured, task-specific behavior, but it does not ensure general intelligence, human-like understanding, or safe behavior in the real world. Rewards and evaluator rubrics can be incomplete or biased. Language ability learned from a few simulated conversations will be limited, and model outputs can still be incorrect. Keep people in control of data use, evaluation, and any decision to turn reviewed examples into a later training dataset.

## References

- [Gymnasium: Creating a Custom Environment](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/en/index)
- [Apache Arrow and PyArrow](https://arrow.apache.org/docs/python/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index)
- [Hugging Face TRL](https://huggingface.co/docs/trl/index)
- [MLflow Tracking and Model Registry](https://mlflow.org/docs/latest/)
- [EleutherAI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PettingZoo: Multi-Agent Reinforcement Learning API](https://pettingzoo.farama.org/)
- [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [gVisor Documentation](https://gvisor.dev/docs/)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)