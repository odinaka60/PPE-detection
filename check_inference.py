from ultralytics import YOLO
from src.core.detection import load_model
from src.utils.config_loader import load_config


cfg = load_config("configs/config.yaml")
model    = load_model(cfg["model"]["path"])

#model = YOLO("models/best.onnx")

results = model("tests/images/construction_worker.png")
detections = []

for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class": class_name,
                "confidence": round(confidence, 3),
                "box": (int(x1), int(y1), int(x2), int(y2)),
            })

print(detections)