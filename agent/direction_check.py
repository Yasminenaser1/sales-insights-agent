"""Catch direction words that contradict the numbers.

The agent sometimes says a value "increased from 165,109 to 123,610" when that
is actually a decrease. This scans for '<direction> ... from A to B' patterns
and flags any where the direction word disagrees with the actual delta.
"""
import re

UP_WORDS = ["increased", "rose", "grew", "jumped", "climbed", "up", "higher", "gained"]
DOWN_WORDS = ["decreased", "fell", "dropped", "declined", "down", "lower", "shrank", "shrunk", "lost"]


def _num(s: str) -> float:
    return float(s.replace(",", "").replace("$", "").replace("%", ""))


def check_directions(report: str) -> list:
    """Return a list of contradictions: dicts with phrase, word, from, to, actual."""
    contradictions = []
    pattern = re.compile(
        r"(increased|rose|grew|jumped|climbed|gained|decreased|fell|dropped|declined|shrank|shrunk|lost)"
        r"[^.]*?from\s+\$?([\d,]+\.?\d*)\s*%?\s+to\s+\$?([\d,]+\.?\d*)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(report):
        word = m.group(1).lower()
        a, b = _num(m.group(2)), _num(m.group(3))
        said_up = word in UP_WORDS
        actual_up = b > a
        if said_up != actual_up:
            actual = "increase" if actual_up else "decrease"
            contradictions.append({
                "phrase": m.group(0).strip(),
                "word": word,
                "from": a,
                "to": b,
                "actual": actual,
            })
    return contradictions
