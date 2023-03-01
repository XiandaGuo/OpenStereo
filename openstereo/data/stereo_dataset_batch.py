import random

import numpy as np
import torch

from . import stereo_trans as ST
from .stereo_dataset import StereoDataset


class StereoBatchDataset(StereoDataset):
    def __init__(self, data_cfg, scope='train'):
        super().__init__(data_cfg, scope)
        # for batch uniform
        self.batch_uniform = data_cfg['batch_uniform'] if 'batch_uniform' in data_cfg else False
        self.random_type = data_cfg['random_type'] if 'random_type' in data_cfg else None
        self.w_range = data_cfg['w_range'] if 'w_range' in data_cfg else None
        self.h_range = data_cfg['h_range'] if 'h_range' in data_cfg else None
        self.random_crop_index = None

    def build_dataset(self):
        if self.data_cfg['name'] in ['KITTI2012', 'KITTI2015']:
            from data.reader.kitti_reader import KittiReader
            self.dataset = KittiReader(self.data_cfg['root'], self.data_cfg[f'{self.scope}_list'])
        elif self.data_cfg['name'] == 'FlyingThings3DSubset':
            from data.reader.sceneflow_reader import FlyingThings3DSubsetReader
            self.dataset = FlyingThings3DSubsetReader(self.data_cfg['root'], self.data_cfg[f'{self.scope}_list'])
        elif self.data_cfg['name'] == 'SceneFlow':
            from data.reader.sceneflow_reader import SceneFlowReader
            self.dataset = SceneFlowReader(self.data_cfg['root'], self.data_cfg[f'{self.scope}_list'])
        elif self.data_cfg['name'] == 'Middlebury':
            from data.reader.middlebury_reader import MiddleburyReader
            self.dataset = MiddleburyReader(self.data_cfg['root'], self.data_cfg[f'{self.scope}_list'])
        elif self.data_cfg['name'] == 'ETH3D':
            from data.reader.eth3d_reader import ETH3DReader
            self.dataset = ETH3DReader(self.data_cfg['root'], self.data_cfg[f'{self.scope}_list'])
        elif self.data_cfg['name'] == 'DrivingStereo':
            from data.reader.driving_reader import DrivingReader
            self.dataset = DrivingReader(self.data_cfg['root'], self.data_cfg[f'{self.scope}_list'])
        else:
            name = self.data_cfg['name']
            raise NotImplementedError(f'{name} dataset is not supported yet.')
        self.build_transform()

    def build_transform(self):
        transform_config = self.data_cfg['transform']
        config = transform_config['train'] if self.is_train else transform_config['test']
        self.transform = self.build_transform_by_cfg(config)

    def __getitem__(self, indexs):
        # set the image_size for this batch
        if self.batch_uniform:
            base_size = self.transform.transforms[self.random_crop_index].size
            size = self.get_size(base_size)
            self.transform.transforms[self.random_crop_index].size = size

        batch_result = {}
        for index in indexs:
            sample = self.dataset[index]
            result = self.transform(sample)
            for each_item in result:
                if isinstance(result[each_item], np.ndarray):
                    tmp = np.expand_dims(result[each_item], 0)
                    if each_item not in batch_result:
                        batch_result[each_item] = tmp
                    else:
                        batch_result[each_item] = np.concatenate([batch_result[each_item], tmp], 0)
                else:
                    tmp = torch.unsqueeze(result[each_item], 0)
                    if each_item not in batch_result:
                        batch_result[each_item] = tmp
                    else:
                        batch_result[each_item] = torch.cat([batch_result[each_item], tmp], 0)
        return batch_result

    def build_transform_by_cfg(self, transform_config):
        transform_compose = []
        for trans in transform_config:
            if trans['type'] == 'CenterCrop':
                transform_compose.append(ST.CenterCrop(trans['size']))
            if trans['type'] == 'TestCrop':
                transform_compose.append(ST.TestCrop(trans['size']))
            if trans['type'] == 'StereoPad':
                transform_compose.append(ST.StereoPad(trans['size']))
            elif trans['type'] == 'RandomCrop':
                transform_compose.append(ST.RandomCrop(trans['size']))
                self.random_crop_index = len(transform_compose) - 1
            if trans['type'] == 'RandomHorizontalFlip':
                transform_compose.append(ST.RandomHorizontalFlip(p=trans['prob']))
            elif trans['type'] == 'GetValidDispNOcc':
                transform_compose.append(ST.GetValidDispNOcc())
            elif trans['type'] == 'GetValidDisp':
                transform_compose.append(ST.GetValidDisp(trans['max_disp']))
            elif trans['type'] == 'TransposeImage':
                transform_compose.append(ST.TransposeImage())
            elif trans['type'] == 'ToTensor':
                transform_compose.append(ST.ToTensor())
            elif trans['type'] == 'NormalizeImage':
                transform_compose.append(ST.NormalizeImage(trans['mean'], trans['std']))

        return ST.Compose(transform_compose)

    def get_size(self, base_size):
        if self.random_type == 'range':
            w = random.randint(self.w_range[0] * base_size[1], self.w_range[1] * base_size[1])
            h = random.randint(self.h_range[0] * base_size[0], self.h_range[1] * base_size[0])
        elif self.random_type == 'choice':
            w = random.choice(self.w_range)
            h = random.choice(self.h_range)
        else:
            raise NotImplementedError
        return int(h), int(w)
