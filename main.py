import cv2

from src.utils.config_loader import load_config
from src.utils.audit_logger import AuditLogger
from src.utils.overlay import draw_overlay
from src.core.detection import load_model, run_tracking
from src.core.association import associate_ppe_to_persons
from src.core.violation_tracker import ViolationTracker


def main():
    cfg = load_config("configs/config.yaml")

    model  = load_model(cfg["model"]["path"])
    logger = AuditLogger(cfg["audit"]["output_dir"])
    cap    = cv2.VideoCapture(cfg["camera"]["source"])

    REQUIRED     = set(cfg["ppe"]["required_classes"])
    PERSON_CLASS = cfg["ppe"].get("person_class", "Person")
    MIN_CONTAIN  = cfg["ppe"].get("min_containment", 0.5)
    CONF         = cfg["model"]["confidence_threshold"]
    INFER_EVERY  = cfg.get("runtime", {}).get("infer_every", 5)

    vcfg = cfg.get("violation", {})
    violations = ViolationTracker(
        REQUIRED,
        confirm_after=vcfg.get("confirm_after", 3),
        clear_after=vcfg.get("clear_after", 2),
        forget_after=vcfg.get("forget_after", 15),
    )

    
    persons, ppe_items, worn = [], [], {}
    frame_count = 0

    print("System running — press Q to quit")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % INFER_EVERY == 0:
                persons, ppe_items = run_tracking(
                    frame, model, CONF, person_class=PERSON_CLASS
                )
                worn = associate_ppe_to_persons(persons, ppe_items, MIN_CONTAIN)

                for ev in violations.update(worn):
                    ev["camera_id"] = cfg["audit"]["camera_id"]
                    logger.write(ev)  

            frame_count += 1
            draw_overlay(frame, persons, ppe_items, worn, REQUIRED)
            cv2.imshow("PPE Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()