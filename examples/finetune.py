import sys
sys.path.insert(0, "../")

from ultralytics import YOLO

model = YOLO("./weights/yolov8n.pt")

args = {'lr0': 0.002, 'lrf': 0.02, 'epochs': 300, 'fraction': 1, 'project': 'sod4bird_finetune',
        'data': "../ultralytics/cfg/datasets/multi_sod4birdsTrain-coco.yaml"}
model.train(**args)
