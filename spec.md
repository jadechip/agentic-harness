---

# Agentic Harness System Specification

A harness-driven platform for “distributed cognition” coding agents, with: **DSL-defined workflows**, **skill execution**, **artifact + evaluation contracts**, **AlphaCode-style sampling**, **feedback loops**, **trace logging**, and **RL optimization of harnesses**. Must support **OpenRouter** as the default LLM provider.

## 0) Non-Negotiable Outcomes

The finished system must satisfy all of the following:

1. **Runs real coding workflows** against a repo: it must actually **edit files** and **verify changes** (tests/build/smoke).
2. Harnesses are **declarative** (DSL-driven). RL must be able to mutate harness parameters **without code changes**.
3. Skills are **capability templates** (not a huge library of tiny hardcoded programs). Determinism comes primarily from **artifacts + evaluations**, not from scripted implementations.
4. Every task produces:

   * an artifact (schema validated)
   * evaluation metrics + score
   * traces sufficient for RL (token usage, latency, model, candidate selection, tool calls, etc.)
5. System supports **offline deterministic test mode** (no external API calls) while also supporting **real LLM runs** via OpenRouter.
6. No “silent mock success”: using a mock provider must be **explicit**.

---

## 1) Conceptual Model

### 1.1 Primitives

The system is defined by these primitives:

* **Harness**: a workflow definition (tasks + flow + feedback + settings).
* **Task**: binds a skill, context policy, artifact output schema, evaluation contract, sampling + retry parameters.
* **Skill**: a reusable cognitive capability template executed by an agent/model; it may do deterministic context discovery, but must rely on LLM for reasoning-heavy synthesis.
* **Tool**: sandboxed side-effect interface (shell, read/write files, git, python).
* **Artifact**: immutable, versioned structured output (JSON content validated by schema).
* **Evaluation Contract**: objective scoring rules; supports deterministic metrics + optional LLM judging.
* **Flow**: DAG of tasks (execution order).
* **Feedback**: explicit loops on events (e.g., `verification.failure -> debug_failure`).
* **Trace**: full telemetry per candidate/run for debugging + RL optimization.

### 1.2 Key Principle

**Generation → Evaluation → Improvement**
(Explicitly inspired by AlphaCode’s generate-many-and-filter approach.)

---

## 2) Architecture

### 2.1 Layers

**Layer A — Harness DSL**

* Declarative harness definition.
* RL mutates harness parameters here (model choice, sampling count, retry limits, budgets, evaluation mode).

**Layer B — Runtime (Python)**

* DSL parser/validator → task graph
* scheduler executes flow DAG
* executor runs skills and supports sampling
* evaluation engine scores artifacts
* improvement loop retries using evaluation feedback
* stores persist artifacts, runs, traces
* RL optimizer searches harness parameter space

### 2.2 Modules (recommended structure)

```
agent_harness/

core/
  artifacts.py
  skills.py
  tasks.py
  evaluations.py
  harness.py
  traces.py

dsl/
  parser.py
  validator.py

runtime/
  scheduler.py
  executor.py
  context_builder.py
  improvement_loop.py

providers/
  base_provider.py
  openrouter_provider.py
  provider_factory.py

tools/
  base_tool.py
  shell_tool.py
  file_tool.py
  git_tool.py
  python_tool.py

evaluation/
  engine.py
  deterministic_checks.py
  llm_judge.py   (can be heuristic in tests; real judge optional)

store/
  artifact_store.py
  trace_store.py
  run_store.py

skills/
  analyze_repository.py
  generate_plan.py
  implement_change.py
  verify_behavior.py
  debug_failure.py

evaluations/
  repo_map_quality.py
  plan_quality.py
  patch_quality.py
  test_pass_rate.py

rl/
  optimizer.py
  harness_space.py
  reward.py

cli/
  main.py

examples/
  coding_agent.harness.yaml

tests/
```

---

## 3) Core Data Models (MUST be stable early)

### 3.1 Artifact

* Immutable once written.
* Must include unique id, type, schema_version, content, produced_by_task, created_at.
* Must be schema-validated at creation time.
* Must be versioned: reattempts create new artifact IDs (never overwrite).

Minimum fields:

* `id: str`
* `type: str`
* `schema_version: str`
* `content: dict`
* `produced_by_task: str`
* `created_at: datetime`
* `parent_artifact_id: Optional[str]` (for improvement lineage, recommended)
* `candidate_index: Optional[int]` (for sampling traceability)

### 3.2 Skill

* Not “a deterministic program per task.” It’s a capability template:

  * goal + prompt templates + tool permissions + structured output requirement
* Skill definition includes:

  * `name`
  * `description`
  * `input_artifacts`
  * `output_artifact`
  * `allowed_tools`

### 3.3 Task

Task binds skill + evaluation + sampling + retry + context policy:

* `name`
* `skill`
* `context` (artifact names to include)
* `produces`
* `evaluation`
* `samples`
* `retry_limit`

### 3.4 Evaluation Contract

* `name`
* `criteria` (list of metric keys)
* `weights` (criterion→float)
* `pass_threshold` (0–1)

### 3.5 Trace

Traces must support RL + debugging; must store:

**Run identity**

* run_id
* harness name/version
* user request
* repo path (or hash)
* timestamp

**Task/candidate identity**

* task_name
* skill_name
* attempt_number
* candidate_index
* selected (bool)

**Model/provider**

* provider_name (openrouter)
* model
* temperature
* max_tokens
* reasoning settings (if used)
* prompt(s): system + user (or assembled)

**Execution**

* tool_calls (ordered list with inputs/outputs, exit codes)
* artifacts generated (ids, types)
* schema validation results
* exceptions if any

**Scoring**

* per-criterion metrics
* final score
* pass/fail
* evaluator mode

**Cost**

* token_usage (prompt + completion if available)
* latency seconds
* estimated cost (optional if provider returns it)

---

## 4) Harness DSL Requirements

### 4.1 Format

* Start with YAML harness specs (simple + RL-friendly).
* Parser produces internal HarnessConfig and TaskGraph.

### 4.2 Mandatory DSL elements

Top-level:

* `harness: <name>`
* `tasks: { ... }` non-empty
* `flow: [ "a -> b", ... ]` optional but recommended; if empty must infer order or error
* `feedback:` optional rules
* `settings:` optional but used heavily (provider/model/eval mode/budgets)

Per-task:

* `skill`
* `produces`
* `evaluate` (or `evaluation`)
* `samples`
* `retry_limit`
* `context` optional list

### 4.3 Validation (must be strict)

Validator must enforce:

**Graph validity**

* Every task referenced in `flow` exists.
* DAG: flow edges cannot create cycles.
* No unreachable tasks unless explicitly allowed.
* Execution order must be deterministic (topological sort).

**Feedback validity**

* Feedback source task exists.
* Feedback target task exists.
* Feedback edges must not be mixed into the main DAG; handled separately.

**Schema validity**

* Task `produces` must reference a known artifact schema.
* Task evaluation must reference a known evaluation contract.

**Settings validity**

* Provider/model configured unless mock mode explicitly enabled.

---

## 5) Runtime Semantics

### 5.1 Execution (main loop)

Given a harness, user_request, repo_path:

1. Start Run record.
2. Compute `execution_order` from flow DAG.
3. For each task in order:

   * Build context deterministically from artifact store.
   * Run **improvement loop** (attempts up to retry_limit):

     * Generate **N candidates** (AlphaCode sampling)
     * Evaluate each candidate artifact
     * Choose best candidate (max score)
     * If best meets threshold: accept + persist artifact + proceed
     * Else: create feedback message from evaluator; retry attempt
   * If still fail and feedback rule exists: jump to feedback target task; increment feedback cycle count
   * Stop if max feedback cycles exceeded or runtime timeout exceeded
4. Finish Run record with status, outcomes, metadata.

### 5.2 AlphaCode-style sampling (required)

For each attempt:

* Run `samples` candidates (parallel or sequential).
* Evaluate all.
* Select best.
* Optionally compute **pass@k** metrics for reporting (highly recommended).

This sampling must be:

* generic (applies to any task)
* traceable (candidate_index + selected flag stored)

### 5.3 Candidate selection and evaluation linkage (required)

The pipeline must be explicit:

* `executor.run_task(...) -> [candidates]`
* `evaluation_engine.score(candidate.artifact) -> score + metrics + feedback`
* pick best candidate by score
* store chosen candidate and mark selected in traces

---

## 6) Context Engineering System

### 6.1 Context Builder responsibilities

Given `task.context` list:

* fetch those artifacts (latest accepted versions)
* fetch repo metadata (path, optionally directory listing)
* fetch task-specific feedback messages from previous attempts
* enforce token budget:

  * summarize or truncate long content
  * include stable references/IDs for traceability
* assemble:

  * system prompt
  * user prompt
  * structured output instruction (schema)

Context must be deterministic for a given set of artifacts/settings (except LLM generation).

---

## 7) Tool Sandbox Requirements

Tools must:

* be sandboxed to the repo directory
* capture all input/output for tracing
* be permissioned per skill/task allowed_tools
* enforce safety rules (deny dangerous commands)

Required tools:

* shell (with cwd pinned to repo root)
* file_read
* file_write / file_edit
* git (optional but useful)
* python (optional)
* patch applier (can be via file_edit or shell `git apply`)

Safety constraints:

* deny destructive ops outside repo
* deny access to arbitrary filesystem
* log everything

---

## 8) Evaluation System

### 8.1 Modes

Evaluation engine must support these modes:

1. **deterministic**: only deterministic checks (default for CI tests)
2. **hybrid**: deterministic + optional LLM judge for qualitative criteria (default for real runs)
3. **llm**: only LLM judge (allowed but not recommended)

Config via harness settings:

* `settings.evaluation_mode: deterministic|hybrid|llm`

### 8.2 Deterministic checks (examples)

For repo mapping:

* top-level directory coverage
* file count sanity
* presence of entrypoints/config
* dependency graph non-empty if applicable

For implementation:

* at least one file changed (diff non-empty)
* patch applies cleanly
* no forbidden files touched

For verification:

* test command exit code
* parse failures into structured report

### 8.3 LLM judge (quality checks)

LLM judge is used only for qualitative metrics:

* architecture summary quality
* roadmap/planning quality
* issue reporting quality (QA report clarity)
* spec compliance (if deterministic is insufficient)

**Important:** In offline mode, a heuristic judge may stand in, but it must be explicit and never treated as “real quality”.

### 8.4 Score computation

Weighted sum:

* metrics per criterion normalized 0–1
* final score = Σ weight_i * metric_i / Σ weights
* pass if score ≥ pass_threshold
* evaluator must return **feedback strings** explaining the lowest metrics, e.g.:

  * “Improve criterion X (current=0.57): missing …”

---

## 9) Improvement Loop Requirements

### 9.1 Retries

For each attempt:

* include in prompt:

  * previous artifact (if any)
  * evaluation feedback
  * what failed (explicitly)
  * constraints (time, tool limits)
* ensure retries are not identical by:

  * forcing the agent to address specific missing criteria
  * optionally raising reasoning effort on later attempts

### 9.2 Feedback cycles (workflow-level loops)

Feedback rules are explicit edges triggered by events, e.g.:

* `verification.failure -> debug_failure`
* `debug_failure.success -> implementation`

Config:

* `settings.max_feedback_cycles`
* `settings.max_runtime_seconds`

Scheduler must:

* keep flow DAG separate
* jump to feedback target tasks when conditions met
* stop on cycle limit

---

## 10) Skill Library Requirements (minimum set)

The system ships with a small, broad set of skills:

1. `analyze_repository`
2. `generate_plan`
3. `implement_change`
4. `verify_behavior`
5. `debug_failure`

### 10.1 Skills must be LLM-driven in real mode

Skills may do deterministic context discovery (e.g., filesystem walk), but must use the provider for:

* synthesis
* planning
* patch generation
* debugging reasoning
* structured report writing

### 10.2 implement_change MUST edit files (critical)

`implement_change` must:

* generate patch instructions via LLM (structured)
* apply edits via tool sandbox (file_edit / patch)
* optionally run quick checks (lint/build)
* output artifact including:

  * files changed
  * diff summary
  * commands executed
  * patch plan + (optional) patch content

If implement_change only emits a plan, verification will never pass. This is the core “make it real” step.

### 10.3 verify_behavior MUST run something real

Verification must:

* choose appropriate test command(s) based on repo:

  * prefer `pytest -q`
  * else `npm test` / `pnpm test` / `go test ./...` etc.
  * else fallback smoke checks (import/build)
* capture stdout/stderr + exit codes
* create QAReport artifact with:

  * commands run
  * failures
  * summary
* evaluation contract must heavily weight exit code pass/fail

### 10.4 debug_failure must generate a repair plan

On failure:

* read QAReport
* propose minimal fix plan (structured)
* hand off to implement_change

---

## 11) LLM Provider Integration (OpenRouter)

### 11.1 Provider Layer

Add provider abstraction:

* base_provider defines interface + normalized response
* OpenRouter provider implements it
* provider factory chooses provider from settings

### 11.2 OpenRouter usage rules

* Must consult OpenRouter docs for:

  * endpoint, headers, request/response fields
  * model naming format
  * usage fields (tokens) if available
* Must support:

  * model id
  * temperature
  * max_tokens
  * system + user prompts
  * tool calling if later extended (optional for MVP)

### 11.3 Configuration + enforcement (no silent mock)

CLI + settings must enforce:

* Default provider: OpenRouter
* Require `OPENROUTER_API_KEY` and `settings.model` unless explicit mock mode
* Mock mode must require explicit flag:

  * `--mock` or `settings.allow_mock: true` plus `--mock`
* If no API key and no mock flag: **hard error** (explain how to configure)

### 11.4 Provider must populate traces

Provider response must supply:

* model
* token usage
* latency
* raw text
* tool calls (optional)

---

## 12) RL Optimization Requirements

RL operates on **harness parameters**, not code.

### 12.1 Search space

At minimum, RL can mutate:

* model choice (OpenRouter model id)
* temperature
* max_tokens
* per-task samples
* per-task retry_limit
* evaluation_mode (deterministic/hybrid)
* context budget settings
* feedback cycle limits

Later expansions (allowed, not required for MVP):

* flow structure mutations
* adding/removing tasks
* model per role (planner/executor/verifier)

### 12.2 Reward function (must be trace-derived)

Reward must reflect:

* quality: final artifact score(s), especially verification score
* correctness: verification pass = major bonus
* cost: token usage penalty
* speed: latency penalty
* stability: fewer retries / fewer feedback cycles bonus

Example:
`reward = wq*score - wt*tokens - wl*latency + wv*verification_pass - wr*retries`

### 12.3 Algorithm

Start with evolutionary search (as implemented), then optionally bandits/PPO.
RL runner must:

* mutate harness settings
* run harness
* compute reward from trace/run outcome
* persist results for analysis

---

## 13) Storage Requirements

Stores must be append-only:

* artifact_store: immutable artifacts + metadata + lineage
* trace_store: candidate-level traces
* run_store: run metadata and outcomes

SQLite is fine for MVP, but schema must support:

* multiple attempts per task
* multiple candidates per attempt
* selection marker
* per-criterion metrics
* provider/model metadata

---

## 14) CLI Requirements

### 14.1 Commands

Minimum commands:

* `run` — execute harness against repo
* `optimize` — run RL search over harness configs
* `export-traces` — dump JSON for training/analysis (optional but recommended)

### 14.2 Flags (must)

* `--harness <path>`
* `--repo <path>`
* `--request <string>`
* `--mock` (explicit opt-in)
* optional:

  * `--model <id>` overrides YAML
  * `--evaluation-mode` overrides YAML
  * `--max-runtime-seconds` overrides YAML

### 14.3 Output

`run` should output:

* run_id
* success/fail
* feedback cycles
* per task outcome list:

  * attempts
  * selected artifact id
  * score + pass/fail
  * key feedback messages
* where artifacts/traces were stored (paths or ids)

---

## 15) Testing Requirements

You currently have a small test suite; it must grow.

### 15.1 Required unit tests

* DSL parser: parse tasks/flow/feedback/settings
* DSL validator: unknown tasks in flow, cycle detection, missing evaluation/schema
* Executor sampling: N candidates created; deterministic candidate_index order; exception handling
* Candidate selection: best score selected, selected flag true
* Artifact immutability: new artifact each attempt/candidate
* Evaluation engine: deterministic scoring returns expected metrics + feedback strings
* Improvement loop: retries until threshold or retry_limit; feedback injected into prompt/context
* Trace completeness: includes model/token/latency/candidate/selected and tool calls

### 15.2 Integration tests

* Mock provider end-to-end: predictable outputs; verifies patch applies and verification runs
* Fixture repo with failing tests: verify detects failure
* Fixture repo with passing tests: verify passes

### 15.3 No-network CI

CI must run entirely in deterministic mode:

* mock provider OR deterministic judge mode
* no OpenRouter calls
* all tests pass without API keys

---

## 16) Known Current-State Issues to Fix (based on our discussion)

These are explicit “must fix” items:

1. **No silent mock fallback**: real runs must error if provider not configured and `--mock` not set.
2. **implement_change must edit files**: stop emitting only “plans”.
3. **verify_behavior must run real commands**: tests/build/smoke; produce QAReport.
4. **Evaluation must prioritize correctness**: test exit codes and real signals over heuristic “richness”.
5. **Traces must be RL-ready**: candidate_index, selected flag, model/provider, token usage, latency, metric breakdowns.
6. **Graph validation**: flow must be a valid DAG; feedback edges handled separately.

---

## 17) Definition of “Working as Expected”

A run is considered “working” when:

* With OpenRouter configured:

  * `repo_analysis` produces a comprehensive CodebaseMap grounded in real repo scanning + LLM synthesis
  * `planning` produces a PhasePlan grounded in requirements + codebase map
  * `implementation` changes files (verified by diff) using tool sandbox
  * `verification` runs tests/build and reports results
  * loop fixes failures until pass or limits reached
  * traces record everything

* With `--mock`:

  * same orchestration runs, but model calls are stubbed
  * used only for tests/dev

* RL `optimize`:

  * mutates harness settings (model/samples/retries/etc.)
  * uses traces + reward function to rank configs
  * produces a best harness config artifact/log

---

## 18) Practical “First Real Milestone” (strongly recommended)

Pick one small target repo fixture and make the agent succeed end-to-end:

* Example: add a trivial OAuth stub route or add a simple CLI flag
* Ensure tests exist
* Verify the harness can:

  * edit file
  * run tests
  * pass
    This becomes your “hello world” benchmark for harness optimization.

---
