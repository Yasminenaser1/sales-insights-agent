# Sales Insights Agent

An autonomous agent that analyzes sales data on its own. Give it a CSV and it
decides what's worth investigating, writes and runs its own analysis code,
checks its own work, and produces a business report — with every number
traceable to the data.

Built to run fully local and free: the model is [Ollama](https://ollama.com)
(`llama3.1:8b`), no API keys, no cloud costs.

## What it does

- **Investigates a dataset autonomously.** You don't ask questions — it plans
  the analyses itself (trends, segments, comparisons, concentration), then runs
  them.
- **Writes and runs its own Python.** The agent generates pandas code and
  executes it in a sandbox.
- **Fixes its own mistakes.** If its code errors, it reads the traceback and
  rewrites the code, up to a few tries.
- **Compares two datasets.** "This year vs last year — what changed?" It runs
  the same analyses on both and reports the differences.
- **Checks itself for honesty.** Every number in the final report is verified
  against the real analysis output; unsupported figures are caught and the
  report is rewritten to remove them.
- **Guarantees correct wording.** Direction words ("rose" / "fell") are
  corrected in code against the actual numbers, since a small model gets these
  wrong.

## How it's built

The agent is built in layers, each one relying on the one below it:

| Layer | File | What it does |
|-------|------|--------------|
| Sandbox | `agent/sandbox.py` | Runs agent-written code in a separate process with a timeout, so nothing can hang or harm the machine |
| Code writer | `agent/analyst.py` | Turns a plain-English question into pandas code, runs it, explains the result |
| Planner | `agent/planner.py` | The agent chooses which analyses to run from a menu of analysis types |
| Validator | `agent/validate.py` | Checks each planned analysis against the real columns; repairs or drops ones that don't fit |
| Self-correction | `agent/self_correct.py` | Retries failed code after the agent reads its own error |
| Orchestrator | `agent/orchestrator.py` | Runs the full autonomous loop end to end |
| Report | `agent/report.py` | Writes a business summary and verifies its own numbers |
| Verifier | `agent/verify.py` | Flags any report number that doesn't trace back to the data |
| Comparison | `agent/compare.py` | Runs one plan on two datasets and reports what changed |

## Design principles

- **The model never invents numbers.** Every figure comes from code that
  actually ran against the data. The report is verified against those results.
- **Enforce in code what the model can't do reliably.** The LLM writes the
  prose; deterministic code guarantees the facts (number traceability, direction
  words).
- **The agent proposes, code disposes.** Plans and reports are sanity-checked by
  plain Python, so an imperfect model still produces valid, honest output.

## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install pandas ollama
# ollama must be running with llama3.1:8b pulled

# Investigate one dataset:
python3 -m agent.report

# Compare two datasets:
python3 -m agent.compare data/sales.csv data/sales_2025.csv
```

## Limitations

- The analysis is **descriptive, not causal** — it shows *what* changed, not
  *why*.
- It runs on a small local model, so it's built defensively: the value is in the
  guardrails (sandbox, validation, verification, self-correction) as much as the
  analysis itself.
- The sample datasets are synthetic, generated with known patterns to
  demonstrate the agent.
