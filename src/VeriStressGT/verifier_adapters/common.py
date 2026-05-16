from __future__ import annotations

import re
from typing import Optional

STATUSES = ("UNSAT", "SAT", "UNKNOWN", "TIMEOUT", "ERROR")

# Authoritative "Result: <token>" line.
# ABCROWN can print "Result: timeout" even when conda exits nonzero.
_RESULT_LINE_RE = re.compile(
    r"(?im)^\s*Result\s*:\s*"
    r"(unsat|sat|unknown|timeout|timed out|verified|safe|unsafe|falsified|violated|counterexample)"
    r"\s*$"
)

# Hard timeout indicators.
# Avoid matching phrases like "Remaining timeout".
_HARD_TIMEOUT_RE = re.compile(
    r"(?im)(?:^|\n)\s*"
    r"(?:time limit reached|timed out|timeout reached|terminated.*timeout|killed.*timeout)"
    r"\b"
)


def normalize_status_from_text(text: str) -> Optional[str]:
    t = (text or "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # 1) Authoritative: last "Result:" line wins.
    matches = list(_RESULT_LINE_RE.finditer(t))
    if matches:
        token = matches[-1].group(1).strip().lower()

        if token in {"unsat", "verified", "safe"}:
            return "UNSAT"

        if token in {"sat", "unsafe", "falsified", "violated", "counterexample"}:
            return "SAT"

        if token in {"timeout", "timed out"}:
            return "TIMEOUT"

        if token == "unknown":
            return "UNKNOWN"

    # 2) Hard timeout only.
    if _HARD_TIMEOUT_RE.search(t):
        return "TIMEOUT"

    # 3) Conservative fallbacks.
    tlow = t.lower()

    if re.search(r"\bunsat\b", tlow):
        return "UNSAT"

    # Avoid SAT matching inside UNSAT. This is already mostly handled by word
    # boundaries, but keep SAT after UNSAT.
    if re.search(r"\bsat\b", tlow):
        return "SAT"

    if re.search(r"\bunknown\b", tlow):
        return "UNKNOWN"

    return None


def finalize_status(parsed: Optional[str], rc: int, timed_out: bool) -> str:
    # External wall-clock kill should still override everything.
    if timed_out:
        return "TIMEOUT"

    # But parsed stdout/stderr result beats nonzero conda rc.
    if parsed in STATUSES:
        return parsed

    if rc == 0:
        return "UNKNOWN"

    return "ERROR"