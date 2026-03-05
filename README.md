# Agent Harness

Harness-driven coding agent system implementing:

- YAML Harness DSL parsing to task graph
- Skill-based task execution
- OpenRouter provider integration (explicit, no silent mock fallback)
- Structured artifacts and evaluation contracts
- Improvement loops with retries and candidate selection
- Full trace capture (provider/model/candidate/selection/criterion scores)
- RL parameter optimization (evolutionary search)

## OpenRouter Run

Set API key and run with explicit model:

```bash
echo 'OPENROUTER_API_KEY=xxxx' > .env
python -m agent_harness.cli.main run \
  --harness examples/coding_agent.harness.yaml \
  --repo . \
  --provider openrouter \
  --model openai/gpt-4o \
  --evaluation-mode hybrid \
  --request "Add OAuth login to repository"
```

Notes:
- `--provider` defaults to `openrouter`.
- `--model` is required unless `--mock` is set.
- The CLI auto-loads `.env` from `--repo` and harness directory roots.
- Without `OPENROUTER_API_KEY` in non-mock mode, the CLI exits with a clear error.

## Mock Run (Explicit)

```bash
python -m agent_harness.cli.main run \
  --harness examples/coding_agent.harness.yaml \
  --repo . \
  --mock \
  --request "Add OAuth login to repository"
```

Run contract:
- Non-mock runs fail if no repository changes were applied.
- Non-mock runs fail if verification was not executed.

## Outputs

- Run/artifact/trace data is stored in SQLite (default `.agent_harness.db`).
- `ImplementationPatch` artifacts include:
  - `files_changed`
  - `diff_summary`
  - `commands_executed`
  - `git_diff_snippet`
- `QAReport` artifacts include verification command results and failure summaries.
- Example feedback loop in the default harness: `verification.failure -> debug_failure -> implementation`.

Export traces:

```bash
python -m agent_harness.cli.main export-traces \
  --run-id <RUN_ID> \
  --output /tmp/traces.json
```

Optimize harness parameters from a baseline trace run:

```bash
python -m agent_harness.cli.main optimize \
  --run-id <RUN_ID>
```

Run-based optimization (mutate settings and execute harness each trial):

```bash
python -m agent_harness.cli.main optimize \
  --harness examples/coding_agent.harness.yaml \
  --repo . \
  --request "Add OAuth login to repository" \
  --mock \
  --generations 3 \
  --population 6
```

## Project Layout

```text
agent_harness/
  core/
  dsl/
  runtime/
  evaluation/
  tools/
  store/
  providers/
  skills/
  evaluations/
  rl/
  cli/
examples/
  coding_agent.harness.yaml
tests/
```
