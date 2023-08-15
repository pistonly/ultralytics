# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.data import build_damo_dataset
from ultralytics.data.augment import Compose, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops
from ultralytics.utils.torch_utils import de_parallel
import pdb

__all__ = 'DAMOYOLOValidator',  # tuple or list


class DAMOYOLOValidator(DetectionValidator):

    def build_dataset(self, img_path, mode='val', batch=None):
        """Build YOLO Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        return build_damo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=gs)

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
        """Apply Non-maximum suppression to prediction outputs.
        return:
        (List[torch.Tensor]): tensor shape (num_boxes, 6 + num_masks) (x1, y1, x2, y2, conf, cls, mask1, mask2, ...)
        """

        # onnx preds=[[1, 8400, 80], [1, 8400, 4]]
        output0, output1 = preds
        if output0.shape[-1] == 80:
            det_cls, det_boxes = output0, output1
        else:
            det_cls, det_boxes = output1, output0
        preds = torch.concat([det_boxes, det_cls], dim=-1)
        preds = torch.permute(preds, [0, 2, 1])

        # # onnx end2end
        # if self.model.onnx and len(self.model.output_names) > 1:
        #     # ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
        #     num_dets, det_boxes, det_scores, det_classes = preds
        #     batch = num_dets.shape[0]
        #     preds = []
        #     for i in range(batch):
        #         preds.append(torch.concat((det_boxes[i], torch.reshape(det_scores, (-1, 1)), torch.reshape(det_classes, (-1, 1))), 1))
        #     return preds

        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        labels=self.lb,
                                        multi_label=True,
                                        agnostic=self.args.single_cls,
                                        max_det=self.args.max_det,
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
            predn = pred.clone()  # paddle model scaled in model
            ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4],
                            shape, ratio_pad=batch['ratio_pad'][si])  # native-space pred

            # Evaluate
            if nl:
                height, width = batch['img'].shape[2:]
                # height, width = batch['resized_shape'][si]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor((width, height, width, height), device=self.device)  # target boxes
                ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                                ratio_pad=batch['ratio_pad'][si])  # native-space labels
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                # NOTE: To get correct metrics, the inputs of `_process_batch` should always be float32 type.
                correct_bboxes = self._process_batch(predn.float(), labelsn)
                # TODO: maybe remove these `self.` arguments as they already are member variable
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
            self.stats.append((correct_bboxes, predn[:, 4], predn[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            if self.args.save_txt:
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)
