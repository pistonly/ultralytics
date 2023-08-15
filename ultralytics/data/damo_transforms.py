from PIL import Image
import cv2
import numpy as np
from typing import Any
import random
import torch
from torchvision.transforms import functional as F
from .augment import Format, Compose
import pdb


class DAMO_Resize(object):
    def __init__(self, max_range):
        if not isinstance(max_range, (list, tuple)):
            max_range = (max_range, )
        self.max_range = max_range

    def get_size_ratio(self, image_size):
        target_size = random.choice(self.max_range)
        w, h = image_size
        t_w, t_h = target_size, target_size
        r = min(t_w / w, t_h / h)
        o_w, o_h = int(w * r), int(h * r)
        return (o_w, o_h)

    # backuped old __call__ method from DAMO code
    def __call__bac(self, image, target=None):
        h, w = image.shape[:2]
        size = self.get_size_ratio((w, h))

        image = cv2.resize(image, size,
                           interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        image = image.transpose((2, 0, 1))
        image = np.ascontiguousarray(image, dtype=np.float32)
        if isinstance(target, list):
            target = [t.resize(size) for t in target]
        elif target is None:
            return image, target
        else:
            target = target.resize(size)
        return image, target

    def __call__(self, labels=dict(), image=None):
        image = labels.get("img") if image is None else image
        h0, w0 = image.shape[:2]
        w, h = self.get_size_ratio((w0, h0))

        image = cv2.resize(image, (w, h),
                         interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        image = image.transpose((2, 0, 1))
        image = np.ascontiguousarray(image, dtype=np.float32)
        labels['img'] = image
        labels['ratio_pad'] = (h / h0, w / w0)
        labels['ori_shape'] = (h0, w0)
        labels['resized_shape'] = (h, w)
        return labels


class DAMO_RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__bac(self, image, target):
        if random.random() < self.prob:
            image = image[:, :, ::-1]
            image = np.ascontiguousarray(image, dtype=np.float32)
            if target is not None:
                target = target.transpose(0)
        return image, target

    def __call__(self, labels=dict(), image=None):
        image = labels.get("img") if image is None else image
        if random.random() < self.prob:
            image = image[:, :, ::-1]
            image = np.ascontiguousarray(image, dtype=np.float32)
            self._update_labels(labels)
            labels['img'] = image
        return labels

    def _update_labels(self, labels):
        # TODO: flip bbox
        pass


class DAMO_ToTensor(object):
    def __call__bac(self, image, target):
        return torch.from_numpy(image), target

    def __call__(self, labels=dict(), image=None):
        image = labels.get("img") if image is None else image
        image = torch.from_numpy(image)
        labels['img'] = image
        return labels


class DAMO_Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__bac(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

    def __call__(self, labels=dict(), image=None):
        image = labels.get("img") if image is None else image
        image = F.normalize(image, mean=self.mean, std=self.std)
        labels['img'] = image
        return labels


class DAMO_Padding_UL(object):
    '''
    padding image to upleft of canvas.
    '''
    def __init__(self, image_max_range):
        self.padded_shape = image_max_range[0:2]

    def __call__bac(self, image, target=None) -> Any:
        c, h, w = image.shape
        assert c == 1 or c == 3 or c==4, "channel number is wrong!"
        assert h <= self.padded_shape[0], "image max range[0] is too small"
        assert w <= self.padded_shape[1], "image max range[1] is too small"
        padded_img = image.new_zeros((c, *self.padded_shape))
        padded_img[:c, :h, :w].copy_(image)
        return padded_img, target

    def __call__(self, labels=dict(), image=None) -> Any:
        image = labels.get("img") if image is None else image
        c, h, w = image.shape
        assert c == 1 or c == 3 or c==4, "channel number is wrong!"
        assert h <= self.padded_shape[0], "image max range[0] is too small"
        assert w <= self.padded_shape[1], "image max range[1] is too small"
        padded_img = image.new_zeros((c, *self.padded_shape))
        padded_img[:c, :h, :w].copy_(image)
        self._update_labels(labels)
        labels['img'] = padded_img

        h1, w1 = padded_img.shape[-2:]
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'], (0, 0, h1 - h, w1 - w))  # top left bottom right
        return labels


    def _update_labels(self, labels):
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[-2:][::-1])
        return labels


class DAMOFormat(Format):
    def _format_img(self, img):
        '''
        do nothing!
        '''
        return img

    def __call__(self, labels):
        """Return formatted image, classes, bounding boxes & keypoints to be used by 'collate_fn'."""
        img = labels['img']
        h, w = img.shape[-2:]
        cls = labels.pop('cls')
        instances = labels.pop('instances')
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)

        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
            else:
                masks = torch.zeros(1 if self.mask_overlap else nl, img.shape[0] // self.mask_ratio,
                                    img.shape[1] // self.mask_ratio)
            labels['masks'] = masks
        if self.normalize:
            instances.normalize(w, h)
        labels['cls'] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels['bboxes'] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        if self.return_keypoint:
            labels['keypoints'] = torch.from_numpy(instances.keypoints)
        # Then we can use collate_fn
        if self.batch_idx:
            labels['batch_idx'] = torch.zeros(nl)
        return labels


class DAMO_Compose(Compose):

    def __call__(self, img_path, labels):
        image = np.asarray(Image.open(img_path).convert('RGB'))
        labels['img'] = image

        for t in self.transforms:
            labels = t(labels)

        return labels

