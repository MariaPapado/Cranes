from ultralytics import YOLO

model = YOLO("./runs/obb/train_dota/weights/best.pt")              # pretrained model i trained on DOTA with train_on_dota.py
#print(model)
model.train(
    data="CRANES.yaml",
    imgsz=640,
    epochs=50,
    batch=8,
#    flipud=0.50,
#    fliplr=0.50,
#    augment=True,
#    freeze=10,                               # optional: freeze backbone
#    lr0=0.001, weight_decay=5e-4,
)
