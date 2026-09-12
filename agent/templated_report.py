"""Level-2 report: the model writes narrative with {placeholders}; code fills in
every number from the verified facts. The model never types a figure, so it
cannot garble one. Any placeholder the model invents is caught and the report
is regenerated.
"""
import os
import re

import ollama

from agent.facts import build_facts

MODEL = os.getenv("ANALYST_MODEL", "llama3.1:8b")
MAX_TRIES = 3


def facts_menu(facts: dict) -> str:
    return "\n".join(f"  {{{name}}} = {f['display']}" for name, f in facts.items())


REPORT_SYSTEM = """You are a senior sales analyst writing a short executive summary.

You must NOT write any numbers yourself. Instead, use these placeholders, which will be filled in with the exact values:

{menu}

Write a 4-6 sentence summary of the sales performance. Every figure MUST be a placeholder from the list above, written exactly like {{top_region}}. Do not invent placeholders and do not type any raw numbers or region/category names - always use a placeholder.

End with one short caveat sentence (this may be plain text): the analysis is descriptive, not causal.

Reply with ONLY the summary."""


def find_placeholders(text: str) -> list:
    return re.findall(r"\{(\w+)\}", text)


def fill(text: str, facts: dict) -> str:
    def repl(m):
        name = m.group(1)
        return facts[name]["display"] if name in facts else m.group(0)
    return re.sub(r"\{(\w+)\}", repl, text)


def templated_report(path: str) -> dict:
    facts = build_facts(path)
    valid_names = set(facts)

    draft = ""
    for attempt in range(1, MAX_TRIES + 1):
        extra = ""
        if attempt > 1:
            extra = f"\n\nYour previous draft used placeholders that don't exist: {bad}. Use ONLY these: {sorted(valid_names)}."
        resp = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": REPORT_SYSTEM.format(menu=facts_menu(facts))},
                {"role": "user", "content": "Write the summary." + extra},
            ],
            options={"temperature": 0},
        )
        draft = resp.message.content.strip()
        used = find_placeholders(draft)
        bad = [p for p in used if p not in valid_names]
        # Also flag raw numbers the model typed instead of using placeholders.
        raw_numbers = re.findall(r"(?<!\{)\b\d[\d,]*\.?\d+\b(?!\})", draft)
        if not bad and not raw_numbers:
            break

    filled = fill(draft, facts)
    return {
        "template": draft,
        "report": filled,
        "bad_placeholders": bad,
        "raw_numbers": raw_numbers,
        "attempts": attempt,
    }


if __name__ == "__main__":
    r = templated_report("data/sales.csv")
    print("=" * 60)
    print("TEMPLATED REPORT (numbers filled by code, not the model)")
    print("=" * 60 + "\n")
    print(r["report"])
    print("\n" + "-" * 60)
    print(f"attempts: {r['attempts']}")
    if r["bad_placeholders"]:
        print(f"WARNING - invented placeholders: {r['bad_placeholders']}")
    if r["raw_numbers"]:
        print(f"NOTE - model typed raw numbers instead of placeholders: {r['raw_numbers']}")
    print("\n--- the template the model wrote (before filling) ---")
    print(r["template"])
