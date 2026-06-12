from pathlib import Path
from datetime import date

def save_report(report):
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"compliance_report_{date.today()}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to {out_path}")

