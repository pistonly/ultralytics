import cv2
from ultralytics.data.build import build_om_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER, ops
from pathlib import Path
import numpy as np
import torch


__all__ = "HisiValidator",

current_dir = Path(__file__).absolute().parent


# def forward_on_2output_numpy(bbox, cls):
#     # x_cat shape: (1, 144, 8400)
#     # cls: (1, 80, 8400)
#     cls = torch.sigmoid(cls)
#     # y: (1, 84, 8400)
#     y = torch.cat((bbox, cls), 1)
#     return y



class HisiValidator(DetectionValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args:dict={}, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.add_callback("on_val_end", self.close_board)
        self.add_callback("on_val_start", self.start_board)
        self.ssh_cfg = args.get('ssh_cfg')
        self.om_exe = args.get("om_exe")
        self.model = None

    def preprocess(self, batch):
        for img_i in batch['img']:
            self.model.board_model.input_one_image(img_i)
        return batch

    def postprocess(self, preds):
        preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou, labels=self.lb, multi_label=True, agnostic=self.args.single_cls, max_det=self.args.max_det)
        return preds

    @staticmethod
    def close_board(obj):
        obj.model.board_model.stop_board()

    @staticmethod
    def start_board(obj):
        kwargs = {}
        if obj.ssh_cfg is not None:
            kwargs['ssh_cfg'] = obj.ssh_cfg
        if obj.om_exe is not None:
            kwargs['om_exe'] = obj.om_exe
        obj.model.board_model.start_board(**kwargs)

    # def build_dataset(self, img_path, mode='val', batch=None):
    #     '''
    #     '''
    #     return build_om_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)
