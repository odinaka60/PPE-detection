import json
import time
from pathlib import Path

class AuditLogger:
    def __init__(self, output_dir: str):
        self.path = Path(output_dir) / "violations.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict):
        record.setdefault("logged_at", time.time())
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")