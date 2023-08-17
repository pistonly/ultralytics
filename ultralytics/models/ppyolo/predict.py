import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.data.pp_transforms import PP_Compose, PPFormat_predict
import yaml
import pdb


class DetectionPredictor(BasePredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        infer_config = self.args.infer_config
        with open(infer_config) as f:
            yml_conf = yaml.safe_load(f)
        self.transforms = PP_Compose(yml_conf['Preprocess'])
        self.transforms.append(PPFormat_predict(device=self.device))

        name_list = yml_conf['label_list']
        self.names = {i: name_i for i, name_i in enumerate(name_list)}

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of Results objects."""
        if len(preds[0].shape) == 2:  # model with nms. for detr
            # TODO: check for batch > 1
            pred = preds[0]
            pred = torch.cat([pred[:, -4:], pred[:, 1:2] , pred[:, 0:1]], dim=1)
            preds = torch.unsqueeze(pred, 0)
        elif len(preds[0].shape) == 1:
            pred = preds[1]
            pred = torch.cat([pred[:, -4:], pred[:, 1:2] , pred[:, 0:1]], dim=1)
            preds = torch.unsqueeze(pred, 0)
        else:
            box_pred = preds[0].permute([0, 2, 1])  # to [batch, bbox:4, nd]
            preds = torch.cat([box_pred, preds[1]], axis=1)
            preds = ops.non_max_suppression(preds,
                                            self.args.conf,
                                            self.args.iou,
                                            agnostic=self.args.agnostic_nms,
                                            max_det=self.args.max_det,
                                            classes=self.args.classes,
                                            multi_label=True,
                                            box_is_xyxy=True)


        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.names, boxes=pred))
        return results

    def pre_transform(self, im):
        """Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """
        batch = [self.transforms(None, {}, im_i) for im_i in im]
        # collate to batch
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img' or k == 'scale_factor':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)
            new_batch[k] = value
        return new_batch

    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        batch = self.pre_transform(im)

        if self.model.fp16:
            batch['img'] = batch['img'].half()

        return batch
