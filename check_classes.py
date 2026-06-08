from ultralytics import YOLO

model = YOLO("models/best.onnx")

print("Your exact class names are:")
print(model.names)