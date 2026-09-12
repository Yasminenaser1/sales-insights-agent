"""Layer 3: the agent plans which analyses to run, on its own.

Given the dataset's shape and a menu of analysis types, the agent chooses which
ones fit this data and fills in the specifics (metric, time column, segment
column). It returns a structured plan we can then execute one step at a time.
"""
import json
import os
import re

import ollama

MODEL = os.getenv("ANALYST_MODEL", "llama3.1:8b")

ANALYSIS_MENU = """Available analysis types:
- trend: how a numeric metric changes over time. Needs: metric (numeric col), time_col (a date col).
- segment: how a numeric metric breaks down across a category. Needs: metric (numeric col), segment_col (a category col).
- concentration: whether a metric is concentrated in a few items (top-N share). Needs: metric (numeric col), segment_col (a category col).
- comparison: compare a metric between the groups of a category. Needs: metric (numeric col), segment_col (a category col).
"""

PLANNER_SYSTEM = """You are a senior sales analyst deciding what to investigate in a new dataset.

{context}

{menu}

Choose 3-5 analyses that would surface the most useful business insights from THIS dataset. Pick analysis types that fit the columns available, and fill in the specific columns for each.

Reply with ONLY a JSON array, no other text. Each item:
{{"type": "<one of the menu types>", "metric": "<numeric column>", "col": "<time or category column>", "why": "<one short phrase>"}}

Use real column names from the dataset. Choose a mix of analysis types, not the same one repeated."""


def data_context(path: str) -> str:
    import pandas as pd
    df = pd.read_csv(path, parse_dates=["order_date"])
    numeric = [c for c in df.columns if str(df[c].dtype).startswith(("int", "float"))]
    dates = [c for c in df.columns if "datetime" in str(df[c].dtype)]
    cats = [c for c in df.columns if df[c].dtype == object]
    return (
        f"Dataset: {len(df)} rows.\n"
        f"Numeric columns: {numeric}\n"
        f"Date columns: {dates}\n"
        f"Category columns: {cats}"
    )


def make_plan(path: str) -> list:
    context = data_context(path)
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM.format(context=context, menu=ANALYSIS_MENU)},
            {"role": "user", "content": "Plan the analyses."},
        ],
        options={"temperature": 0},
    )
    text = resp.message.content.strip()
    # Pull out the JSON array even if the model adds stray text.
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"planner did not return JSON:\n{text}")
    return json.loads(match.group(0))


if __name__ == "__main__":
    plan = make_plan("data/sales.csv")
    print(f"The agent planned {len(plan)} analyses:\n")
    for i, step in enumerate(plan, 1):
        print(f"{i}. [{step.get('type')}] metric={step.get('metric')}, col={step.get('col')}")
        print(f"   why: {step.get('why')}")
