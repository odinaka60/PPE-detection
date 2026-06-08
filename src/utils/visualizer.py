import cv2

CLASS_COLORS = {
    "Helmet":      (0, 255, 0),
    "Safety Vest": (0, 200, 255),
    "Gloves":      (255, 180, 0),
    "Mask":        (180, 0, 255),
    "Glasses":     (255, 255, 0),
}
_DEFAULT_COLOR = (200, 200, 200)

def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f"{d['class']} {d['confidence']:.0%}"
        color = CLASS_COLORS.get(d["class"], _DEFAULT_COLOR)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            color, thickness=-1
        )
        cv2.putText(
            frame, label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), thickness=2, lineType=cv2.LINE_AA
        )
