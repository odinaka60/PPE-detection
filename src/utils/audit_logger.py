import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

class AuditLogger:
    def __init__(self, output_dir: str):
        self.path = Path(output_dir) / "violations.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, camera_id, missing_ppe, confidences):
        record = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "camera_id": camera_id,
            "violation_type": "missing_ppe",
            "missing_items": missing_ppe,
            "detection_confidence": confidences,
            "schema_version": "1.0"
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")