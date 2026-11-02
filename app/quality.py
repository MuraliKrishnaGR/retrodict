"""Quality gate for everything Retrodict shows people. It never changes the result; it lists problems
so they are seen and fixed, not shipped. Run after every photo (scripts/run.py prints it)."""
import re

from app.schemas import TRACE_NAMES

MAX_SUMMARY_WORDS = 60
ABSENCE = re.compile(r"\b(no|absence of|without|lack of|not (?:seen|visible|present))\b[^.]*\b("
                     + "|".join(sorted(t.replace("_", " ") for t in TRACE_NAMES)) + r")\b", re.I)


def check_result(res: dict) -> list:
    issues = []
    ids = {o["id"] for o in res.get("objects", [])}
    texts = [("summary", res.get("summary", ""))] + [
        (f"event {e['id']}", e.get("description", "")) for e in res.get("events", [])]

    for where, text in texts:
        if re.search(r"\bo\d+\b", text):
            issues.append(f"{where}: contains an object id")
        if "_" in text:
            issues.append(f"{where}: contains an internal name with an underscore")
    summary = res.get("summary", "")
    if not summary.strip():
        issues.append("summary: empty")
    if len(summary.split()) > MAX_SUMMARY_WORDS:
        issues.append(f"summary: {len(summary.split())} words (max {MAX_SUMMARY_WORDS})")
    if len(ABSENCE.findall(summary)) >= 2:
        issues.append("summary: lists traces that are absent")

    for e in res.get("events", []):
        if not e.get("evidence"):
            issues.append(f"event {e['id']}: no evidence")
        for ev in e.get("evidence", []):
            if ev["object_id"] not in ids:
                issues.append(f"event {e['id']}: evidence points at unknown object {ev['object_id']}")
        lo, hi = e["minutes_before_min"], e["minutes_before_max"]
        if lo < 0 or hi < lo:
            issues.append(f"event {e['id']}: bad time range {lo} to {hi}")
        if e.get("status") not in ("verified", "partly_verified", "unverified", "contested"):
            issues.append(f"event {e['id']}: not checked by the verifier")
    for oid, s in res.get("states", {}).items():
        if s.get("confidence", 0) == 0 and s.get("observations") == ["unclear"]:
            issues.append(f"object {oid}: the model did not describe it")
    return issues
