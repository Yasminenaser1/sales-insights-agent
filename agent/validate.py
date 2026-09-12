"""Layer 3 guardrail: sanity-check the agent's plan against what each
analysis type actually requires, using the real column types.

The agent proposes; this code disposes. It keeps valid steps, repairs
near-misses where it can, and drops steps that can't be made sense of.
Nothing here calls the model - it is deterministic and testable.
"""
import pandas as pd

# What kind of column each analysis type needs for its "col" slot.
#   trend       -> col must be a DATE column
#   segment / comparison / concentration -> col must be a CATEGORY column
COL_KIND = {
    "trend": "date",
    "segment": "cat",
    "comparison": "cat",
    "concentration": "cat",
}


def column_kinds(path: str) -> dict:
    df = pd.read_csv(path, parse_dates=["order_date"])
    numeric = [c for c in df.columns if str(df[c].dtype).startswith(("int", "float"))]
    dates = [c for c in df.columns if "datetime" in str(df[c].dtype)]
    used = set(numeric) | set(dates)
    # In pandas 3.0 text columns can be 'str'/'string' dtype, not just 'object'.
    # Treat any non-numeric, non-date column as a usable category.
    cats = [c for c in df.columns if c not in used]
    return {"numeric": numeric, "date": dates, "cat": cats}


def validate_plan(plan: list, path: str) -> tuple[list, list]:
    """Return (kept, rejected). Each kept step has a valid type, metric, and col.

    Repairs one common mistake: if col is the wrong kind but a column of the
    right kind exists, swap in the first available one.
    """
    kinds = column_kinds(path)
    kept, rejected = [], []

    for step in plan:
        atype = step.get("type")
        metric = step.get("metric")
        col = step.get("col")

        if atype not in COL_KIND:
            rejected.append((step, f"unknown analysis type {atype!r}"))
            continue
        if metric not in kinds["numeric"]:
            rejected.append((step, f"metric {metric!r} is not a numeric column"))
            continue

        need = COL_KIND[atype]  # "date" or "cat"
        if col in kinds[need]:
            kept.append(step)  # already valid
            continue

        # Repair: col is the wrong kind - swap in the first column of the right kind.
        if kinds[need]:
            step["col"] = kinds[need][0]
            step["_repaired"] = f"col was {col!r}, swapped to {kinds[need][0]!r}"
            kept.append(step)
        else:
            rejected.append((step, f"no {need} column available for {atype}"))

    # Drop duplicate (type, metric, col) steps the agent may have repeated.
    seen, deduped = set(), []
    for step in kept:
        key = (step["type"], step["metric"], step["col"])
        if key not in seen:
            seen.add(key)
            deduped.append(step)

    return deduped, rejected
