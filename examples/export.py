import sys
sys.path.insert(0, "../")

from ultralytics import YOLO

pt_model = "/mnt/root2204/home/liuyang/Documents/YOLO/ultralytics-compare/examples/leakyrelu_train/train/weights/best.pt"
model = YOLO(pt_model)
# model.export(format="torchscript", batch=1, imgsz=640)
model.export(format="onnx", batch=1, imgsz=640)
