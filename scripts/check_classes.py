from ultralytics import YOLO
from src.core.detection import load_models
from src.utils.config_loader import load_config


cfg = load_config("configs/config.yaml")
ppe_model, person_model = load_models(cfg["model"]["path"], cfg["model"]["person_path"])

print("Your exact class names are:")
print(ppe_model.names)