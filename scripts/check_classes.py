from ultralytics import YOLO
from src.core.detection import load_model
from src.utils.config_loader import load_config


cfg = load_config("configs/config.yaml")
model    = load_model(cfg["model"]["path"])
#model = YOLO("models/best.onnx")

print("Your exact class names are:")
print(model.names)