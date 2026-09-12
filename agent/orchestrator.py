"""Layer 3 payoff: the full autonomous loop.

Hand it a dataset. It plans analyses (planner), sanity-checks them (validator),
then for each one asks the model to write the analysis code, runs it in the
sandbox, and collects the result. No question from the user - it decides what
to investigate and does it.
"""
import os

import ollama

from agent.planner import make_plan
from agent.validate import validate_plan
from agent.sandbox import run_code
from agent.analyst import strip_fences, data_context

MODEL = os.getenv("ANALYST_MODEL", "llama3.1:8b")

# Turn a validated plan step into a concrete instruction for the code-writer.
STEP_INSTRUCTIONS = {
    "trend": "Show how {metric} changes over time using {col}. Resample by month and print the monthly totals.",
    "segment": "Show total {metric} broken down by {col}, sorted high to low.",
    "comparison": "Compare {metric} across the groups of {col}: print the total and the mean for each group.",
    "concentration": "Show whether {metric} is concentrated: print each {col} group's share of total {metric} as a percentage, sorted high to low.",
}

CODE_SYSTEM = """You are a data analyst writing a short pandas snippet.

{context}

Rules:
- The DataFrame is loaded as `df`. Do not read any file.
- Write only the code for the task, and print() the result clearly with a short label.
- pandas as pd and numpy as np are available. No plotting, no file writing.
- Reply with ONLY code, no markdown fences, no explanation."""


def write_code_for(step: dict, context: str) -> str:
    task = STEP_INSTRUCTIONS[step["type"]].format(metric=step["metric"], col=step["col"])
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": CODE_SYSTEM.format(context=context)},
            {"role": "user", "content": task},
        ],
        options={"temperature": 0},
    )
    return strip_fences(resp.message.content)


def investigate(path: str) -> list:
    """Run the full autonomous loop. Returns a list of completed analyses."""
    context = data_context(path)
    raw_plan = make_plan(path)
    plan, rejected = validate_plan(raw_plan, path)

    print(f"Planned {len(raw_plan)} analyses; {len(plan)} valid after checks.\n")

    findings = []
    for i, step in enumerate(plan, 1):
        label = f"{step['type']}: {step['metric']} by {step['col']}"
        print(f"[{i}/{len(plan)}] {label} ...")
        code = write_code_for(step, context)
        result = run_code(code, path)
        findings.append({
            "step": step,
            "label": label,
            "code": code,
            "output": result["output"],
            "error": result["error"],
            "ok": result["ok"],
        })
        print("   " + ("done" if result["ok"] else "FAILED: " + result["error"][:80]))
    return findings


if __name__ == "__main__":
    findings = investigate("data/sales.csv")
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for f in findings:
        print(f"\n### {f['label']}")
        if f["ok"]:
            print(f["output"])
        else:
            print("(analysis failed:", f["error"][:100], ")")
