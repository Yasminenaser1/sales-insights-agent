"""The main deliverable: an investigation, not just a summary.

Runs the autonomous plan, picks the most dig-worthy finding, follows that lead
deeper (the reactive digging loop), then writes a report that presents the
headline findings AND the investigation trail - all number-verified and
self-repaired, same as before.
"""
import os

import ollama

from agent.orchestrator import investigate
from agent.dig import investigate_deeper
from agent.verify import verify_report
from agent.direction_check import fix_directions

MODEL = os.getenv("ANALYST_MODEL", "llama3.1:8b")
MAX_REPAIRS = 2


def pick_finding_to_dig(findings: list) -> dict | None:
    """Code picks the lead to follow: the first successful segment/comparison
    finding (those have groups worth drilling into). Deterministic, not the model."""
    for f in findings:
        if f["ok"] and f["step"]["type"] in ("segment", "comparison", "concentration"):
            return f
    return None


def format_findings(findings: list) -> str:
    return "\n\n".join(f"## {f['label']}\n{f['output']}" for f in findings if f["ok"])


def format_trail(trail: list) -> str:
    lines = []
    for s in trail:
        if s["dug"] and s["ok"]:
            lines.append(f"Dug in ({s['reason']}):\n## {s['label']}\n{s['output']}")
    return "\n\n".join(lines)


REPORT_SYSTEM = """You are a senior sales analyst writing an executive summary that shows its investigation.

You are given: (1) headline analyses, and (2) a deeper investigation that followed the most interesting lead.

Write a concise report:
1. One-sentence overview.
2. 3-4 headline insights, each citing a specific number.
3. A short "Going deeper" paragraph that describes what the investigation into the standout found, citing numbers from the investigation section.
4. A "Caveats" line: descriptive not causal, single dataset.

Strict rules: use ONLY numbers that appear verbatim in the results below. Never invent or estimate. Under 300 words, plain business language."""


def _write(results_text: str, trail_text: str, extra: str = "") -> str:
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM},
            {"role": "user", "content": f"HEADLINE ANALYSES:\n\n{results_text}\n\nDEEPER INVESTIGATION:\n\n{trail_text}\n\nWrite the report.{extra}"},
        ],
        options={"temperature": 0},
    )
    return resp.message.content.strip()


def full_report(path: str) -> dict:
    findings = investigate(path)
    lead = pick_finding_to_dig(findings)

    trail = []
    if lead:
        print(f"\nFollowing the most interesting lead: {lead['label']}")
        trail = investigate_deeper(lead, path)

    results_text = format_findings(findings)
    trail_text = format_trail(trail) or "(no deeper investigation was warranted)"
    all_findings = findings + [{"ok": s.get("ok"), "output": s.get("output", "")} for s in trail if s.get("dug")]

    report = _write(results_text, trail_text)
    check = verify_report(report, all_findings)
    repairs = 0
    while check["unsupported"] and repairs < MAX_REPAIRS:
        repairs += 1
        bad = ", ".join(str(n) for n in check["unsupported"])
        report = _write(results_text, trail_text,
                        f"\n\nIMPORTANT: these numbers are NOT in the results: {bad}. Rewrite using only numbers that appear verbatim above.")
        check = verify_report(report, all_findings)

    report, dir_fixes = fix_directions(report)
    return {"report": report, "check": check, "repairs": repairs, "dir_fixes": dir_fixes, "trail": trail}


if __name__ == "__main__":
    r = full_report("data/sales.csv")
    print("\n" + "=" * 60)
    print("SALES INVESTIGATION REPORT")
    print("=" * 60 + "\n")
    print(r["report"])
    c = r["check"]
    print("\n" + "-" * 60)
    if r["repairs"]:
        print(f"(self-repaired numbers {r['repairs']}x)")
    if r["dir_fixes"]:
        print(f"(corrected direction wording {r['dir_fixes']}x)")
    print(f"VERIFICATION: {len(c['supported'])}/{c['n_report_numbers']} numbers trace back (trust: {c['trust']}).")
    if c["unsupported"]:
        print(f"WARNING - unsupported: {c['unsupported']}")
