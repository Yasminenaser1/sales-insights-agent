# Sales Insights Agent

An autonomous agent that analyzes sales data on its own. Give it a CSV and it
decides what's worth investigating, writes and runs its own analysis code,
follows the interesting leads deeper, and produces a business report — with
every number guaranteed correct.

Runs fully local and free: the model is [Ollama](https://ollama.com)
(`llama3.1:8b`), no API keys, no cloud costs. The project is really a case study
in one problem: **how to make a small, unreliable model produce reliable output.**

## What it does

- **Investigates on its own.** You don't ask questions — it plans the analyses
  itself (trends, segments, comparisons, concentration) and runs them.
- **Writes and runs its own Python**, executed in a sandbox.
- **Follows its own leads.** After a result, it decides what's most interesting
  and digs deeper — e.g. "West leads sales → break West down by category →
  Office Supplies drives it" — producing an investigation trail, not just a flat
  summary.
- **Fixes its own code.** On an error it reads the traceback and rewrites the
  code.
- **Compares two datasets.** "This year vs last year — what changed?"
- **Can't get the numbers wrong.** The final report is written with placeholders;
  code fills in every figure from verified facts, so the model never types a
  number and can't garble one.

## The reliability story

A small local model is fast and free but makes mistakes: it garbles numbers,
misreads which value is largest, and gets "rose" vs "fell" backwards. Rather than
hope it behaves, the system is built so it *can't* go wrong on the things that
matter. Each layer solves one real failure:

| Failure of the raw model | How it's handled |
|--------------------------|------------------|
| Writes code that crashes | Runs in a sandbox with a timeout; self-corrects from the traceback |
| Plans analyses that don't fit the data | A validator repairs or drops them before they run |
| Invents numbers in the report | Every figure is verified against real output; unsupported ones trigger a rewrite |
| Garbles digits when writing prose | The report uses placeholders; **code** fills in every number |
| Says "increased" when a value fell | Direction words are corrected in code against the actual delta |
| Misreads which group is the standout | Code computes the true standout; the agent's investigation is corrected to it |

The principle throughout: **the model writes the language, code guarantees the
facts.**

## How it's built

| File | Role |
|------|------|
| `agent/sandbox.py` | Runs agent-written code safely (separate process + timeout) |
| `agent/analyst.py` | Plain-English question → pandas code → explanation |
| `agent/planner.py` | The agent chooses which analyses to run |
| `agent/validate.py` | Checks/repairs the plan against the real columns |
| `agent/self_correct.py` | Retries failed code after the agent reads its error |
| `agent/orchestrator.py` | Runs the full autonomous loop |
| `agent/dig.py` | Reactive digging — follows leads, verifies the standout |
| `agent/verify.py` | Flags report numbers not backed by the data |
| `agent/direction_check.py` | Corrects wrong "rose/fell" wording in code |
| `agent/facts.py` | Extracts exact, labeled facts from the data |
| `agent/templated_report.py` | Model writes placeholders; code fills the numbers |
| `agent/full_report.py` | The main deliverable: findings + a followed lead |
| `agent/compare.py` | Compares two datasets |

## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install pandas ollama
# ollama running with llama3.1:8b pulled

python3 -m agent.full_report                              # investigate one dataset
python3 -m agent.compare data/sales.csv data/sales_2025.csv   # compare two
python3 -m agent.templated_report                        # the placeholder-safe report
```

## Limitations

- Descriptive, not causal — it shows *what*, not *why*.
- Built around a small local model, so the value is as much in the guardrails as
  in the analysis.
- Sample datasets are synthetic, generated with known patterns to demonstrate
  the agent.
