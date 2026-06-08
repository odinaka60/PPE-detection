import cv2
from src.utils.config_loader import load_config
from src.utils.audit_logger import AuditLogger
from src.core.detection import load_model, run_detection

def main():
    cfg = load_config("configs/config.yaml")

    model    = load_model(cfg["model"]["path"])
    logger   = AuditLogger(cfg["audit"]["output_dir"])
    cap      = cv2.VideoCapture(cfg["camera"]["source"])

    REQUIRED = set(cfg["ppe"]["required_classes"])
    frame_count = 0
    INFER_EVERY = 5
    detections = []

    print("System running — press Q to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if frame_count % INFER_EVERY == 0:
                detections = run_detection(
                    frame, model,
                    cfg["model"]["confidence_threshold"]
                )

                detected_classes = {d["class"] for d in detections}
                missing = REQUIRED - detected_classes

                if missing:
                    confidences = {d["class"]: d["confidence"]
                                  for d in detections}
                    logger.write(
                        camera_id   = cfg["audit"]["camera_id"],
                        missing_ppe = list(missing),
                        confidences = confidences
                    )

            frame_count += 1
            cv2.imshow("PPE Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()