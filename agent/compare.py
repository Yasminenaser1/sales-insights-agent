"""Compare two datasets and report what changed.

The trick that makes it apples-to-apples: plan the analyses ONCE, then run the
same plan on both datasets. Different plans per dataset wouldn't be comparable.
Then we compute the deltas and the agent writes a 'what changed' report, with
the same verification + self-repair as the single-dataset report.
"""
import os

import ollama

from agent.planner import make_plan
from agent.validate import validate_plan
from agent.orchestrator import write_code_for
from agent.self_correct import run_with_fixes
from agent.analyst import data_context
from agent.verify import verify_report
from agent.direction_check import check_directions

MODEL = os.getenv("ANALYST_MODEL", "llama3.1:8b")
MAX_REPAIRS = 2


def run_plan_on(path: str, plan: list, context: str) -> list:
    """Run a fixed plan against one dataset and collect the outputs."""
    findings = []
    for step in plan:
        label = f"{step['type']}: {step['metric']} by {step['col']}"
        code = write_code_for(step, context)
        result = run_with_fixes(code, path, context)
        findings.append({"label": label, "output": result["output"], "ok": result["ok"]})
    return findings


COMPARE_SYSTEM = """You are a senior sales analyst comparing two periods of sales data (A = earlier, B = later).

You are given the SAME analyses run on both datasets. Write a short "What Changed" report:

1. One sentence overview of the biggest shift.
2. 3-4 bullet points, each naming a specific change with the before -> after numbers from the results (e.g. "West sales fell from X to Y").
3. A short "Caveats" line: this is descriptive, it shows what changed, not why, and both are single datasets.

Strict rules:
- Use ONLY numbers that appear verbatim in the results below. Never invent or estimate.
- Focus on real differences between A and B. If something barely changed, say it was stable.
- Under 250 words, plain business language."""


def _write_comparison(a_text: str, b_text: str, extra: str = "") -> str:
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": COMPARE_SYSTEM},
            {"role": "user", "content": f"=== DATASET A (earlier) ===\n{a_text}\n\n=== DATASET B (later) ===\n{b_text}\n\nWrite the What Changed report.{extra}"},
        ],
        options={"temperature": 0},
    )
    return resp.message.content.strip()


def format_findings(findings: list) -> str:
    return "\n\n".join(f"## {f['label']}\n{f['output']}" for f in findings if f["ok"])


def compare(path_a: str, path_b: str) -> dict:
    # Plan once (on A), validate, then run the SAME plan on both - keeps it comparable.
    context_a = data_context(path_a)
    raw_plan = make_plan(path_a)
    plan, _ = validate_plan(raw_plan, path_a)
    print(f"Planned {len(plan)} analyses; running the same set on both datasets.\n")

    print("Analyzing dataset A...")
    findings_a = run_plan_on(path_a, plan, context_a)
    print("Analyzing dataset B...")
    findings_b = run_plan_on(path_b, plan, data_context(path_b))

    a_text, b_text = format_findings(findings_a), format_findings(findings_b)

    report = _write_comparison(a_text, b_text)
    # Verify against BOTH datasets' numbers (a real figure could come from either).
    all_findings = findings_a + findings_b
    check = verify_report(report, all_findings)
    repairs = 0
    while check["unsupported"] and repairs < MAX_REPAIRS:
        repairs += 1
        bad = ", ".join(str(n) for n in check["unsupported"])
        extra = (f"\n\nIMPORTANT: these numbers are NOT in the results: {bad}. "
                 f"Rewrite using ONLY numbers that appear verbatim above.")
        report = _write_comparison(a_text, b_text, extra)
        check = verify_report(report, all_findings)

    # Second guardrail: fix any direction words that contradict the numbers.
    dir_fixes = 0
    contradictions = check_directions(report)
    while contradictions and dir_fixes < MAX_REPAIRS:
        dir_fixes += 1
        problems = "; ".join(
            f"you wrote '{c['word']}' but {c['from']:.0f} to {c['to']:.0f} is a {c['actual']}"
            for c in contradictions
        )
        extra = (f"\n\nIMPORTANT: some direction words are wrong: {problems}. "
                 f"Rewrite so every 'increased/decreased/rose/fell' matches the actual numbers. "
                 f"Keep all numbers exactly as they appear in the results.")
        report = _write_comparison(a_text, b_text, extra)
        contradictions = check_directions(report)

    check = verify_report(report, all_findings)  # re-verify numbers after the rewrite
    return {"report": report, "check": check, "repairs": repairs, "dir_fixes": dir_fixes}


if __name__ == "__main__":
    import sys
    a = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"
    b = sys.argv[2] if len(sys.argv) > 2 else "data/sales_2025.csv"
    result = compare(a, b)
    print("\n" + "=" * 60)
    print("WHAT CHANGED (A -> B)")
    print("=" * 60 + "\n")
    print(result["report"])
    check = result["check"]
    print("\n" + "-" * 60)
    if result["repairs"]:
        print(f"(self-repaired numbers {result['repairs']} time(s))")
    if result.get("dir_fixes"):
        print(f"(fixed direction wording {result['dir_fixes']} time(s))")
    print(f"VERIFICATION: {len(check['supported'])}/{check['n_report_numbers']} numbers trace back (trust: {check['trust']}).")
    if check["unsupported"]:
        print(f"WARNING - still unsupported: {check['unsupported']}")
