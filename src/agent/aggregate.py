import json
from collections import Counter
from pathlib import Path


def load_events(path: str) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def summarise(events: list[dict]) -> dict:
    starts = [e for e in events if e.get("event") == "violation_start"]
    ends = [e for e in events if e.get("event") == "violation_end"]

    by_ppe = Counter(ppe for e in starts for ppe in e.get("missing_ppe", []))
    durations = [e.get("duration_s", 0.0) for e in ends]

    return {
        "camera_id": next((e.get("camera_id") for e in events if e.get("camera_id")), "unknown"),
        "total_violations": len(starts),
        "unique_workers": len({e.get("person_id") for e in starts}),
        "by_missing_ppe": dict(by_ppe.most_common()),
        "total_violation_seconds": round(sum(durations), 1),
        "longest_violation_seconds": round(max(durations, default=0.0), 1),
    }