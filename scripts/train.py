from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def train(
    data: str,
    model: str = "yolov8s.pt",
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    patience: int = 15,
    project: str = "runs/train",
    name: str = "ppe",
    save_period: int = 10,
    resume: bool = False,
    export: bool = False,
    half: bool = False,
) -> Path:
 

    last = Path(project) / name / "weights" / "last.pt"

    if resume and last.exists():
        print(f"Resuming from {last}")
        yolo = YOLO(str(last))
        results = yolo.train(resume=True)
    else:
        yolo = YOLO(model)
        results = yolo.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            patience=patience,
            project=project,
            name=name,
            save_period=save_period,
        )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Training complete. Best weights: {best}")

    if export:
        print("Exporting to ONNX ...")
        YOLO(str(best)).export(format="onnx", imgsz=imgsz, half=half)
        print(f"Exported: {best.with_suffix('.onnx')}")

    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a YOLOv8 PPE-detection model.")
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--model", default="yolov8s.pt", help="Base weights to start from")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="GPU id, e.g. '0', or 'cpu'")
    p.add_argument("--patience", type=int, default=15, help="Early-stopping patience")
    p.add_argument("--project", default="runs/train", help="Output root directory")
    p.add_argument("--name", default="ppe", help="Run name (subfolder of --project)")
    p.add_argument("--save-period", type=int, default=10, help="Checkpoint every N epochs")
    p.add_argument("--resume", action="store_true", help="Resume from last.pt if present")
    p.add_argument("--export", action="store_true", help="Export best.pt to ONNX when done")
    p.add_argument("--half", action="store_true", help="FP16 ONNX export (GPU deploy only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        save_period=args.save_period,
        resume=args.resume,
        export=args.export,
        half=args.half,
    )


if __name__ == "__main__":
    main()
