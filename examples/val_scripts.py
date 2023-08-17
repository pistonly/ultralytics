import sys

sys.path.insert(0, "../")

from ultralytics import YOLO, PPYOLO, DAMOYOLO
from pathlib import Path
import argparse
import os
import pdb


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="./weights_test_onnx", type=str)
parser.add_argument("--project", default="val-fold", type=str)
parser.add_argument("--iou", default=0.65, type=float)
parser.add_argument("--conf", default=0.25, type=float)
parser.add_argument("--data", default="../ultralytics/datasets/coco1.yaml", type=str, help="val data, default: coco val2017")
parser.add_argument("--replace", action='store_true', help="replace exist fold, else skip")

flags = parser.parse_args()

model_dir = Path(flags.model_dir)
assert model_dir.is_dir(), f"{flags.model_dir} is not directory!"

project = f"{flags.project}_conf-{flags.conf}_iou-{flags.iou}"

# do not increase save_dir
exists_ok = True

error_weights = []
for w in model_dir.iterdir():

    # filter
    if not (str(w).endswith("-0.onnx") or str(w).endswith("-1.onnx") or str(w).endswith("_8.5.1.7.engine")):
        continue

    model_name = Path(w).name
    if model_name.startswith("paddle"):
        model_cls = PPYOLO
    elif model_name.startswith("damo"):
        model_cls = DAMOYOLO
    else:
        model_cls = YOLO

    name = Path(w).stem
    args = {'iou': flags.iou, "conf": flags.conf, "name": name, "project": project,
            'exist_ok': exists_ok, 'replace': flags.replace, 'device': "0"}

    model = None
    try:
        model = model_cls(w, task="detect")
        metric = model.val(data="../ultralytics/cfg/datasets/coco.yaml", **args)
        if metric is None:
            print(f"skip weights: {str(w)}")
    except Exception as exc:
        print(exc.args)
        error_weights.append(str(w))
        # remove reside dir
        if hasattr(model, "save_dir") and Path(model.save_dir).is_dir():
            for _f in Path(model.save_dir).iterdir():
                os.remove(str(_f))
            os.removedirs(model.save_dir)
    del model
print(f"error weights:\n {error_weights}")
