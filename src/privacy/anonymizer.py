import cv2
from pathlib import Path

_MODELS_DIR = Path(__file__).parent.parent.parent / "models"

def load_face_net():
    net = cv2.dnn.readNetFromCaffe(
        str(_MODELS_DIR / "deploy.prototxt"),
        str(_MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"),
    )
    return net

def anonymize_faces(frame, face_net):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
           (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x1 >= x2 or y1 >= y2:
                continue
            roi = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 0)