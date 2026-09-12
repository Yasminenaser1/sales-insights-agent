"""Layer 4: run agent-written code, and if it errors, let the agent read the
traceback and fix its own code - the way a human analyst would. Retries a few
times, then gives up gracefully.
"""
import os

import ollama

from agent.sandbox import run_code
from agent.analyst import strip_fences

MODEL = os.getenv("ANALYST_MODEL", "llama3.1:8b")
MAX_ATTEMPTS = 3

FIX_SYSTEM = """You are a data analyst fixing a pandas snippet that failed.

{context}

You are given the code you wrote and the error it produced. Return corrected code that fixes the error.
- The DataFrame is loaded as `df`. Do not read any file.
- Common fix: `resample` on a time frequency needs a datetime index - use `df.resample('ME', on='order_date')` or set the index first.
- pandas as pd and numpy as np are available. print() the result.
- Reply with ONLY the corrected code, no fences, no explanation."""


def run_with_fixes(code: str, data_path: str, context: str) -> dict:
    """Run code; on error, ask the agent to fix it and retry. Returns the final
    result plus how many attempts it took and the code that finally ran."""
    attempts = []
    current = code

    for n in range(1, MAX_ATTEMPTS + 1):
        result = run_code(current, data_path)
        attempts.append({"attempt": n, "code": current, "ok": result["ok"], "error": result["error"]})
        if result["ok"]:
            return {"ok": True, "output": result["output"], "code": current, "attempts": attempts}

        # Ask the agent to fix its own code using the error.
        if n < MAX_ATTEMPTS:
            fix = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": FIX_SYSTEM.format(context=context)},
                    {"role": "user", "content": f"Code:\n{current}\n\nError:\n{result['error'][-800:]}"},
                ],
                options={"temperature": 0},
            )
            current = strip_fences(fix.message.content)

    return {"ok": False, "output": "", "error": attempts[-1]["error"], "code": current, "attempts": attempts}
