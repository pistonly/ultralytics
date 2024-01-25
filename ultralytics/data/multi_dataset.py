from .dataset import YOLODataset
import random
import numpy as np


class MultiYOLODataset(YOLODataset):
    """
    :param path_dict: {'multi_ds': [{'path': path0, 'ratio': 0.3, 'label_map': array([1, 3, 0, 4]), 'cache_image': 1},
                   {'path': path1, 'ratio': 0.3, 'label_map': array([80]), 'cache_image': 1},
                   {'path': path2, 'ratio': 0.4, 'label_map': None, 'cache_image': 1},
                   {'base_path': path3, 'path': path4, 'ratio': 0.1, 'label_map': None, 'cache_image': 0},
                   {'path': path3, 'hyp': {'mixup': 0.5}}]}

    """
    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
        assert isinstance(kwargs['img_path'], dict), f"img_path should be dict, which is: {kwargs['img_path']}"
        _, img_path_list = kwargs['img_path'].popitem()

        self.datasets_multi = []
        self.datasets_ratios = []
        self.datasets_labelMaps = []
        self.datasets_lens = []
        for i, path_dict in enumerate(img_path_list):
            dataset_path = path_dict['path']
            # TODO: update hyp
            dataset_hyp = path_dict.get('hyp', {})
            kwargs_i = kwargs.copy()
            kwargs_i['img_path'] = dataset_path

            self.datasets_multi.append(YOLODataset(*args, data=data, use_segments=use_segments,
                                              use_keypoints=use_keypoints, **kwargs_i))
            self.datasets_ratios.append(path_dict['ratio'])
            self.datasets_labelMaps.append(path_dict.get('label_map'))
            self.datasets_lens.append(len(self.datasets_multi[i]))
        self.ratios_cum = np.cumsum(self.datasets_ratios)

    def __getitem__(self, index):
        rand_num = random.random()
        for i, r_edge in enumerate(self.ratios_cum):
            if rand_num <= r_edge:
                index = index % self.datasets_lens[i]
                item = self.datasets_multi[i].__getitem__(index)
                if self.datasets_labelMaps[i] is not None:
                    item['cls'].apply_(lambda x: self.datasets_labelMaps[i].get(x, x))
                return item

    def __len__(self):
        return self.datasets_lens[0]



