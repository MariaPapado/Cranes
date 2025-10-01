from ultralytics import YOLO
# Create a new YOLO11n-OBB model from scratch
model = YOLO("yolo11n-obb.yaml")

# Train the model on the DOTAv1 dataset
results = model.train(data="DOTAv1.yaml", epochs=100, imgsz=1024, batch=2)
