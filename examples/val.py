import sys

sys.path.insert(0, "../")

from ultralytics import YOLO, PPYOLO, DAMOYOLO
from pathlib import Path
import pdb

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--w", default="", type=str)
parser.add_argument("--replace", action='store_true', help="override old results")

flags = parser.parse_args()


# w = "./weights/yolov5s_640x640_b-1.onnx"
# w = "./weights/yolov5su_640x640_b-1.onnx"
# w = "./weights/yolov6s.onnx"
# w = "./weights/yolov7-tiny_b1.onnx"
# w = "./weights/yolov7-tiny_640x640_1output_int8mixfp16_1.bmodel"
# w = "./weights/yolov8n.onnx"
# w = "./weights/yolov8n.pt"
# w = "./weights/yolov8n_640x640_b-1_fp16_8.5.1.7.engine"
# w = "./weights/damoyolo_tinynasL20_T_436.onnx"
w = "./weights/paddle-ppyoloe-plus-crn-t-auxhead-relu-300e-coco_640x640_b-0.onnx"
# w = "./weights/paddle-rtdetr-r18vd-6x-coco_640x640_b-0.onnx"
# w = "./weights/paddle-ppyoloe-plus-crn-t-auxhead-relu-300e-coco_640x640_b-0_int8_minmax_8.5.1.7.engine"

if not flags.w:
    flags.w = w

flags.w = Path(flags.w)
model_name = flags.w.name

if model_name.startswith("paddle"):
    model_cls = PPYOLO
elif model_name.startswith("damo"):
    model_cls = DAMOYOLO
else:
    model_cls = YOLO
model = model_cls(flags.w, task="detect")

# name = Path(w).stem.split("_")[0]
name = Path(flags.w).stem

exists_ok = True  # not increase save_dir. If want new results, set it to: False.

args = {'conf': 0.25, 'iou':0.65, 'name':name, "plots": True, 'project': "runs", 'device': "0",
        'exist_ok': exists_ok, 'replace': flags.replace}

metrics = model.val(data="../ultralytics/cfg/datasets/coco.yaml", **args)
if metrics is None:
    print("skiped")

