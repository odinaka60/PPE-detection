from ultralytics import YOLO

def load_model(path: str):
    return YOLO(path)

def run_detection(frame, model, confidence_threshold: float):
    results = model(frame, conf=confidence_threshold, verbose=False)
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
    return detections