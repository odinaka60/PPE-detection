from ultralytics import YOLO


def load_model(path: str):
    return YOLO(path)


def run_tracking(frame, model, conf: float, person_class: str = "Person"):
    """One tracked inference pass over a frame.

    Returns two lists:
      persons   -> [{"id": track_id, "box": (x1,y1,x2,y2)}, ...]
      ppe_items -> [{"class": name, "confidence": c, "box": (x1,y1,x2,y2)}, ...]
    """
    results = model.track(frame, persist=True, conf=conf, verbose=False)

    persons, ppe_items = [], []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            class_name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            xyxy = (int(x1), int(y1), int(x2), int(y2))

            if class_name == person_class:
                if box.id is not None:
                    persons.append({"id": int(box.id[0]), "box": xyxy})
            else:
                ppe_items.append({
                    "class": class_name,
                    "confidence": round(float(box.conf[0]), 3),
                    "box": xyxy,
                })

    return persons, ppe_items


def run_detection(frame, model, confidence_threshold: float):
    """Frame-level detection without tracking for compatibility / debugging."""
    results = model(frame, conf=confidence_threshold, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            class_name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class": class_name,
                "confidence": round(float(box.conf[0]), 3),
                "box": (int(x1), int(y1), int(x2), int(y2)),
            })
    return detections