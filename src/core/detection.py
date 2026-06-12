from ultralytics import YOLO

def load_models(ppe_path, person_path):
    return YOLO(ppe_path, task="detect"), YOLO(person_path)

def run_tracking(frame, ppe_model, person_model, conf, person_conf=0.3, min_person_h=0.25):
    fh = frame.shape[0]
    persons = []
    for r in person_model.track(frame, persist=True, classes=[0],
                                conf=person_conf, verbose=False):
        if r.boxes is None:
            continue
        for box in r.boxes:
            if box.id is None:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if (y2 - y1) < min_person_h * fh:      # skip distant/background people
                continue
            persons.append({"id": int(box.id[0]), "box": (x1, y1, x2, y2)})

    # PPE from your custom model (its Person class is ignored)
    ppe_items = []
    for r in ppe_model(frame, conf=conf, verbose=False):
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = ppe_model.names[int(box.cls[0])]
            if cls == "Person" or cls.startswith("NO-"): 
                continue
            ppe_items.append({"class": cls,
                              "confidence": round(float(box.conf[0]), 3),
                              "box": tuple(map(int, box.xyxy[0].tolist()))})
    return persons, ppe_items