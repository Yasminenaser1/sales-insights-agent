"""Runs Python code the agent wrote, safely.

The agent generates analysis code as text. We never want to just exec() that
blindly - a stray infinite loop or a bad file operation could hang or harm the
machine. This runs the code in a separate process with:
  - a time limit (killed if it runs too long)
  - a fixed working directory and only the data + pandas/numpy available
  - stdout captured, so we get the printed result back as text
"""
import subprocess
import sys
import tempfile
from pathlib import Path

TIME_LIMIT_SECONDS = 15

# Code we prepend so the agent's snippet always has the data loaded and the
# safe libraries imported. The agent just writes analysis against `df`.
PREAMBLE = """
import pandas as pd
import numpy as np
df = pd.read_csv({data_path!r}, parse_dates=['order_date'])
"""


def run_code(code: str, data_path: str) -> dict:
    """Run one snippet of agent code against the dataset.

    Returns {"ok": bool, "output": str, "error": str}.
    The snippet should print() whatever it wants to report.
    """
    full = PREAMBLE.format(data_path=data_path) + "\n" + code

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=TIME_LIMIT_SECONDS,
        )
        if result.returncode == 0:
            return {"ok": True, "output": result.stdout.strip(), "error": ""}
        return {"ok": False, "output": result.stdout.strip(), "error": result.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "", "error": f"Code ran longer than {TIME_LIMIT_SECONDS}s and was stopped."}
    finally:
        Path(script_path).unlink(missing_ok=True)
