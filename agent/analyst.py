"""A sales-analyst agent (Layer 2): turns a plain-English question into
Python analysis, runs it safely, and explains the result.

The agent writes code that prints its answer; the sandbox runs it; the agent
then explains the printed result in plain English. The rule that matters:
the explanation must be based on what the code actually returned, never made up.
"""
import os
import re

import ollama

from agent.sandbox import run_code

MODEL = os.getenv("ANALYST_MODEL", "llama3.1:8b")
DATA_PATH = "data/sales.csv"

# Shown to the model so it knows the shape of the data without us hardcoding it.
def data_context(path: str) -> str:
    import pandas as pd
    df = pd.read_csv(path, parse_dates=["order_date"])
    lines = [f"The dataset is a pandas DataFrame `df` with {len(df)} rows and these columns:"]
    for col in df.columns:
        lines.append(f"  - {col} ({df[col].dtype})")
    lines.append(f"Date range: {df['order_date'].min().date()} to {df['order_date'].max().date()}")
    return "\n".join(lines)


CODE_SYSTEM = """You are a data analyst. You write short Python (pandas) snippets to answer questions about a sales dataset.

{context}

Rules:
- The DataFrame is already loaded as `df`. Do not read any file.
- Write only the code needed to answer the question, and print() the result clearly.
- Use pandas/numpy only. No plotting, no file writing, no imports beyond pandas as pd and numpy as np (already available).
- Keep it short. Reply with ONLY the code, no explanation, no markdown fences."""

EXPLAIN_SYSTEM = """You are a data analyst explaining a result to a business audience.
You are given a question and the exact output of the analysis code that answered it.
Explain what it shows in 1-3 plain sentences. Use ONLY the numbers in the output - never invent or round beyond what's shown. If the output is an error, say the analysis failed."""


def strip_fences(text: str) -> str:
    """Models often wrap code in ```python ... ``` despite being told not to."""
    text = re.sub(r"^```[a-zA-Z]*\n", "", text.strip())
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def ask(question: str) -> dict:
    context = data_context(DATA_PATH)

    # Step 1: the agent writes analysis code.
    code_resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": CODE_SYSTEM.format(context=context)},
            {"role": "user", "content": question},
        ],
        options={"temperature": 0},
    )
    code = strip_fences(code_resp.message.content)

    # Step 2: run it safely.
    result = run_code(code, DATA_PATH)

    # Step 3: the agent explains the actual output.
    explain_resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXPLAIN_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nCode output:\n{result['output'] or result['error']}"},
        ],
        options={"temperature": 0},
    )
    answer = explain_resp.message.content.strip()

    return {"question": question, "code": code, "output": result["output"], "error": result["error"], "answer": answer}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What are total sales by region?"
    r = ask(q)
    print("QUESTION:", r["question"])
    print("\n--- code the agent wrote ---")
    print(r["code"])
    print("\n--- output ---")
    print(r["output"] or ("ERROR: " + r["error"]))
    print("\n--- answer ---")
    print(r["answer"])
