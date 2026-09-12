"""Reactive digging: the agent looks at a result, decides if anything is worth
investigating, and proposes ONE structured follow-up analysis. Guided, so the
follow-up is always valid; the agent supplies the judgment (what to drill into).
"""
import json
import os
import re

import ollama

from agent.validate import column_kinds

MODEL = os.getenv("ANALYST_MODEL", "llama3.1:8b")

DIG_SYSTEM = """You are a data analyst deciding whether a result is worth digging into.

You are given: the analysis that was just run, and its output.

Available columns to drill down on (categories): {cats}
Available metrics (numeric): {metrics}

Look at the output. Decide if one group stands out - unusually high, unusually low, or otherwise notable. If so, propose ONE follow-up: filter to that standout group, then break the same metric down by a DIFFERENT category column.

Reply with ONLY a JSON object, no other text:
{{"dig": true/false,
  "reason": "<what stood out, one phrase>",
  "filter_col": "<category column of the standout group>",
  "filter_value": "<the standout group's value>",
  "breakdown_col": "<a DIFFERENT category column to break it down by>",
  "metric": "<numeric metric to analyze>"}}

If nothing clearly stands out, reply {{"dig": false, "reason": "<why not>"}}.
Use only real column names. breakdown_col must differ from filter_col."""


def propose_dig(finding: dict, path: str) -> dict:
    kinds = column_kinds(path)
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": DIG_SYSTEM.format(cats=kinds["cat"], metrics=kinds["numeric"])},
            {"role": "user", "content": f"Analysis: {finding['label']}\n\nOutput:\n{finding['output']}"},
        ],
        options={"temperature": 0},
    )
    text = resp.message.content.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"dig": False, "reason": "could not parse a follow-up"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"dig": False, "reason": "invalid follow-up"}


def validate_dig(dig: dict, path: str) -> tuple[bool, str]:
    """Make sure a proposed dig is runnable, so we never execute nonsense."""
    if not dig.get("dig"):
        return False, dig.get("reason", "no dig proposed")
    kinds = column_kinds(path)
    if dig.get("filter_col") not in kinds["cat"]:
        return False, f"filter_col {dig.get('filter_col')!r} is not a category"
    if dig.get("breakdown_col") not in kinds["cat"]:
        return False, f"breakdown_col {dig.get('breakdown_col')!r} is not a category"
    if dig.get("breakdown_col") == dig.get("filter_col"):
        return False, "breakdown_col must differ from filter_col"
    if dig.get("metric") not in kinds["numeric"]:
        return False, f"metric {dig.get('metric')!r} is not numeric"
    return True, "ok"


def true_standout(finding_output: str, filter_col: str, path: str):
    """Compute the actual standout group (max of the metric) for the given
    category column, so we can check the agent's claim against reality."""
    import pandas as pd
    df = pd.read_csv(path, parse_dates=["order_date"])
    # We don't re-derive the metric here; the caller passes it in verify step.
    return None  # placeholder, real logic in verify_and_correct


def verify_and_correct(dig: dict, path: str) -> dict:
    """Check the agent's claimed standout against the real data; if it picked
    the wrong group, correct filter_value to the true top group. Keeps the
    agent's chosen breakdown_col and metric."""
    import pandas as pd
    if not dig.get("dig"):
        return dig
    df = pd.read_csv(path, parse_dates=["order_date"])
    fcol, metric = dig.get("filter_col"), dig.get("metric")
    if fcol not in df.columns or metric not in df.columns:
        return dig  # validation will catch this separately

    totals = df.groupby(fcol)[metric].sum()
    true_top = str(totals.idxmax())
    claimed = str(dig.get("filter_value"))

    if claimed != true_top:
        dig["_corrected"] = f"agent said {claimed!r} was the standout, but {true_top!r} is actually highest by {metric}"
        dig["filter_value"] = true_top
        dig["reason"] = f"{true_top} is highest by {metric}"
    return dig


# --- The digging loop -------------------------------------------------------
from agent.orchestrator import write_code_for
from agent.self_correct import run_with_fixes
from agent.analyst import data_context, strip_fences

MAX_DEPTH = 2

DIG_CODE_SYSTEM = """You are a data analyst writing a short pandas snippet.

{context}

Task: filter df to rows where {filter_col} == {filter_value!r}, then show total {metric}
broken down by {breakdown_col}, sorted high to low. print() the result with a short label.

Rules: df is already loaded. pandas as pd, numpy as np available. No file reading, no plotting.
Reply with ONLY code, no fences, no explanation."""


def _run_dig(dig: dict, path: str, context: str) -> dict:
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": DIG_CODE_SYSTEM.format(
                context=context, filter_col=dig["filter_col"], filter_value=dig["filter_value"],
                metric=dig["metric"], breakdown_col=dig["breakdown_col"])},
            {"role": "user", "content": "Write the analysis."},
        ],
        options={"temperature": 0},
    )
    code = strip_fences(resp.message.content)
    result = run_with_fixes(code, path, context)
    label = f"{dig['metric']} for {dig['filter_col']}={dig['filter_value']} broken down by {dig['breakdown_col']}"
    return {"label": label, "output": result["output"], "ok": result["ok"], "code": result["code"]}


def investigate_deeper(finding: dict, path: str) -> list:
    """Starting from one finding, follow leads: propose a dig, verify/correct it,
    run it, then consider digging into THAT result. Returns the investigation trail."""
    context = data_context(path)
    trail = []
    current = finding

    for depth in range(MAX_DEPTH):
        dig = propose_dig(current, path)
        dig = verify_and_correct(dig, path)
        ok, reason = validate_dig(dig, path)
        if not ok:
            trail.append({"depth": depth + 1, "dug": False, "reason": reason})
            break

        step = _run_dig(dig, path, context)
        trail.append({
            "depth": depth + 1,
            "dug": True,
            "reason": dig.get("reason"),
            "corrected": dig.get("_corrected"),
            "label": step["label"],
            "output": step["output"],
            "ok": step["ok"],
        })
        if not step["ok"]:
            break
        current = step  # dig into the new result next round

    return trail


if __name__ == "__main__":
    finding = {
        "label": "segment: sales by region",
        "output": "region\nCentral    113105.90\nEast       128721.80\nSouth       71117.16\nWest       165109.06",
    }
    trail = investigate_deeper(finding, "data/sales.csv")
    print("INVESTIGATION TRAIL\n")
    print("Start: segment: sales by region (West leads)\n")
    for step in trail:
        if step["dug"]:
            note = f"  [corrected: {step['corrected']}]" if step.get("corrected") else ""
            print(f"Depth {step['depth']}: dug because '{step['reason']}'{note}")
            print(f"  -> {step['label']}")
            print("  " + (step["output"].replace("\n", "\n  ") if step["ok"] else "(failed)"))
            print()
        else:
            print(f"Depth {step['depth']}: stopped - {step['reason']}\n")
