# Agent Harness

Harness-driven coding agent system implementing:

- YAML Harness DSL parsing to task graph
- Skill-based task execution
- Structured artifacts and evaluation contracts
- Improvement loops with retries
- Full trace capture
- RL parameter optimization (evolutionary search)

## Quick Start

```bash
python -m agent_harness.cli.main run \
  --harness examples/coding_agent.harness.yaml \
  --repo . \
  --request "Add OAuth login to repository"
```

Export traces:

```bash
python -m agent_harness.cli.main export-traces \
  --run-id <RUN_ID> \
  --output /tmp/traces.json
```

Optimize harness parameters from a baseline run:

```bash
python -m agent_harness.cli.main optimize \
  --run-id <RUN_ID>
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
  skills/
  evaluations/
  rl/
  cli/
examples/
  coding_agent.harness.yaml
tests/
```
