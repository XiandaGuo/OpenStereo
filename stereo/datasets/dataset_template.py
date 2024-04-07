# @Time    : 2023/11/9 16:50
# @Author  : zhangchenming
import os
import random
import torch.utils.data as torch_data
from .dataset_utils import stereo_trans


def build_transform_by_cfg(transform_config):
    transform_compose = []
    for cur_cfg in transform_config:
        cur_augmentor = getattr(stereo_trans, cur_cfg.NAME)(config=cur_cfg)
        transform_compose.append(cur_augmentor)
    return stereo_trans.Compose(transform_compose)


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__()
        self.data_info = data_info
        self.data_cfg = data_cfg
        self.mode = mode
        self.root = self.data_info.DATA_PATH

        self.split_file = self.data_info.DATA_SPLIT[self.mode.upper()]
        self.data_list = []
        if os.path.exists(self.split_file):
            with open(self.split_file, 'r') as fp:
                self.data_list.extend([x.strip().split(' ') for x in fp.readlines()])

        transform_config = self.data_cfg.DATA_TRANSFORM[self.mode.upper()]
        self.transform = build_transform_by_cfg(transform_config)

    def __len__(self):
        return len(self.data_list)


def get_size(base_size, w_range, h_range, random_type):
    if random_type == 'range':
        w = random.randint(int(w_range[0] * base_size[1]), int(w_range[1] * base_size[1]))
        h = random.randint(int(h_range[0] * base_size[0]), int(h_range[1] * base_size[0]))
    elif random_type == 'choice':
        w = random.choice(w_range) if isinstance(w_range, list) else w_range
        h = random.choice(h_range) if isinstance(h_range, list) else h_range
    else:
        raise NotImplementedError
    return int(h), int(w)


def custom_collate(data_list, concat_dataset, batch_uniform=False,
                   random_type=None, h_range=None, w_range=None):
    if batch_uniform:
        for each_dataset in concat_dataset.datasets:
            for cur_t in each_dataset.transform.transforms:
                if type(cur_t).__name__ == 'RandomCrop':
                    base_size = cur_t.base_size
                    cur_t.crop_size = get_size(base_size, w_range, h_range, random_type)
                    break

    return torch_data.default_collate(data_list)
