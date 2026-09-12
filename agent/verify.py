"""Layer 6: verify the report's numbers against the real analysis output.

The agent can misread a table and state a number that never actually appeared
(e.g. calling a total a "mean"). This checks every number in the report against
the numbers the analyses actually produced. It doesn't need to know the right
answer - it just enforces: every figure you state must trace back to real output.
"""
import re


def extract_numbers(text: str) -> list:
    """Pull numeric values from text: 1,234.56 / 35.7% / $238 etc. -> floats."""
    nums = []
    for raw in re.findall(r"-?\d[\d,]*\.?\d*", text):
        cleaned = raw.replace(",", "")
        try:
            nums.append(round(float(cleaned), 2))
        except ValueError:
            pass
    return nums


def verify_report(report: str, findings: list, tol: float = 0.02) -> dict:
    """Check each number in the report against numbers in the findings.

    A report number is 'supported' if it matches (within a small relative
    tolerance) some number that appears in the analysis outputs. Returns the
    supported and unsupported numbers so we can flag the report if needed.
    """
    # All numbers the analyses actually produced.
    real = set()
    for f in findings:
        if f.get("ok"):
            for n in extract_numbers(f["output"]):
                real.add(n)

    report_nums = extract_numbers(report)

    supported, unsupported = [], []
    for n in report_nums:
        # A match is exact, or within tol relative distance (handles rounding).
        hit = any(
            n == r or (r != 0 and abs(n - r) / abs(r) <= tol)
            for r in real
        )
        (supported if hit else unsupported).append(n)

    return {
        "supported": supported,
        "unsupported": unsupported,
        "n_report_numbers": len(report_nums),
        "n_unsupported": len(unsupported),
        "trust": "high" if not unsupported else ("medium" if len(unsupported) <= 2 else "low"),
    }
