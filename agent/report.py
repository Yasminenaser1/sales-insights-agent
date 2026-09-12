"""Layer 5: turn the raw findings into a business insight report.

The agent reads every completed analysis and writes a short executive summary:
the most important things in the data, in plain business language, using ONLY
the numbers the analyses actually produced. It also states what it can't
conclude - the honesty guardrail.
"""
import os

import ollama

from agent.orchestrator import investigate

MODEL = os.getenv("ANALYST_MODEL", "llama3.1:8b")

REPORT_SYSTEM = """You are a senior sales analyst writing a short executive summary for business leaders.

You are given the results of several analyses that were run on a sales dataset. Write a concise report:

1. Start with a 1-2 sentence overview.
2. Give the 3-4 most important insights as bullet points. Each must cite a specific number from the results.
3. End with a short "Caveats" line noting what this analysis does NOT tell you (e.g. it doesn't explain WHY, it's descriptive not causal, the data is one dataset).

Strict rules:
- Use ONLY numbers that appear in the results below. Never invent, estimate, or round beyond what's shown.
- If a result looks flat or unremarkable, say so honestly rather than inventing a trend.
- Keep it under 250 words. Plain business language, no jargon."""


def format_findings(findings: list) -> str:
    parts = []
    for f in findings:
        if f["ok"]:
            parts.append(f"## {f['label']}\n{f['output']}")
    return "\n\n".join(parts)


def make_report(path: str) -> dict:
    findings = investigate(path)
    results_text = format_findings(findings)

    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM},
            {"role": "user", "content": f"Analysis results:\n\n{results_text}\n\nWrite the executive summary."},
        ],
        options={"temperature": 0},
    )
    return {"report": resp.message.content.strip(), "findings": findings}


if __name__ == "__main__":
    result = make_report("data/sales.csv")
    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60 + "\n")
    print(result["report"])
