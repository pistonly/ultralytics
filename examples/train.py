import sys
sys.path.insert(0, "../")

from ultralytics import YOLO

# model = YOLO("../ultralytics/cfg/models/v8/yolov8-leakyrelu.yaml").load("./weights/yolov8n.pt")
# model.train(data="../ultralytics/cfg/datasets/coco.yaml", epochs=300, fraction=1, project='leakyrelu_train')
model = YOLO("./weights/yolov8n.pt")

model.train(data="../ultralytics/cfg/datasets/multi_sod4birdsTrain-coco.yaml", epochs=300, fraction=1, project='sod4bird_finetune')
