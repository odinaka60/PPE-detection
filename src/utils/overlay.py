import cv2

GREEN = (0, 200, 0)
RED   = (0, 0, 255)
BLUE  = (255, 160, 0)


def draw_overlay(frame, persons, ppe_items, worn, required):
    """Draw PPE boxes (thin, blue) and person boxes coloured by compliance:
    green = wearing everything required, red = missing something."""
    for item in ppe_items:
        x1, y1, x2, y2 = item["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), BLUE, 1)
        cv2.putText(frame, item["class"], (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, BLUE, 1)

    for p in persons:
        x1, y1, x2, y2 = p["box"]
        missing = required - worn.get(p["id"], set())
        color = RED if missing else GREEN
        label = f"ID {p['id']}"
        label += "  missing: " + ", ".join(sorted(missing)) if missing else "  OK"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame
