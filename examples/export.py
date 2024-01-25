import sys
sys.path.insert(0, "../")

from ultralytics import YOLO
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--imgsz", type=int, default=[736, 1280], nargs="+",
                    help="4K: [2176, 3840]\n1080: [1088, 1920]\n720: [736, 1280]")
parser.add_argument("--w", default="./yolov8n.pt", type=str)

flags = parser.parse_args()

# pt_model = "/mnt/root2204/home/liuyang/Documents/YOLO/ultralytics-compare/examples/leakyrelu_train/train/weights/best.pt"
model = YOLO(flags.w)
# model.export(format="torchscript", batch=1, imgsz=640)
# model.export(format="onnx", batch=1, imgsz=640)
model.export(format="onnx", batch=1, imgsz=flags.imgsz)

w = Path(flags.w)
f = str(w.with_suffix('.onnx'))

f_new = str(w.parent / f"{w.stem}_{flags.imgsz[0]}x{flags.imgsz[1]}.onnx")
subprocess.run(['mv', f, f_new])
