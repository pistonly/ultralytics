# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.data import build_pp_dataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.plotting import plot_images, output_to_target
import pdb

__all__ = 'PPYOLOValidator',  # tuple or list


class PPYOLOValidator(DetectionValidator):

    def build_dataset(self, img_path, mode='val', batch=None):
        """Build YOLO Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        return build_pp_dataset(self.args, img_path, batch, self.data, mode=mode, stride=gs)

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float())
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch['img'].shape[2:]
            nb = len(batch['img'])
            bboxes = batch['bboxes'] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch['cls'][batch['batch_idx'] == i], bboxes[batch['batch_idx'] == i]], dim=-1)
                for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling

        return batch

    def postprocess(self, preds):
        """
        Apply Non-maximum suppression to prediction outputs.
        output (x1, y1, x2, y2, conf, class)
        """
        if len(preds[0].shape) == 2:  # model with nms. for detr
            # TODO: check for batch > 1
            pred = preds[0]
            pred = torch.cat([pred[:, -4:], pred[:, 1:2] , pred[:, 0:1]], dim=1)
            preds = torch.unsqueeze(pred, 0)
            return preds
        elif len(preds[0].shape) == 1:
            pred = preds[1]
            pred = torch.cat([pred[:, -4:], pred[:, 1:2] , pred[:, 0:1]], dim=1)
            preds = torch.unsqueeze(pred, 0)
            return preds
        else:
            box_pred = preds[0].permute([0, 2, 1])  # to [batch, bbox:4, nd]
            preds = torch.cat([box_pred, preds[1]], axis=1)
            preds = ops.non_max_suppression(preds,
                                            self.args.conf,
                                            self.args.iou,
                                            labels=self.lb,
                                            multi_label=True,
                                            box_is_xyxy=True)
            return preds

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            shape = batch['ori_shape'][si]
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()  # paddle model scaled in model, native-space xyxy, conf, cls

            # Evaluate
            if nl:
                tbox = ops.xywh2xyxy(bbox)  # target boxes
                tbox[..., [0, 2]] *= shape[1]  # native-space pred
                tbox[..., [1, 3]] *= shape[0]  # native-space pred
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                # NOTE: To get correct metrics, the inputs of `_process_batch` should always be float32 type.
                correct_bboxes = self._process_batch(predn.float(), labelsn)
                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, pred[:, 4], predn[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            if self.args.save_txt:
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)

    def plot_predictions(self, batch, preds, ni):
        '''
        preds: list[(x1, y1, x2, y2, conf, cls)], unscaled bboxes by pp-model. Need to rescale to input image size.
        '''
        img = batch['img']  # scaled image for model
        scale_factor = batch['scale_factor']  # new / old
        preds_new = []
        for i, pred in enumerate(preds):
            bboxes = pred[:, :4]
            # rescale bboxes
            bboxes[:, [0, 2]] *= scale_factor[i][1]
            bboxes[:, [1, 3]] *= scale_factor[i][0]
            preds_new.append(pred)
        plot_images(img, *output_to_target(preds_new, max_det=self.args.max_det),
                    paths=batch['im_file'],
                    fname=self.save_dir / f'val_batch{ni}_pred.jpg',
                    names=self.names,
                    on_plot=self.on_plot)
