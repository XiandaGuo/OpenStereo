import random

import numpy as np
import torch

from . import stereo_trans as ST
from .stereo_dataset import StereoDataset


class StereoBatchDataset(StereoDataset):
    def __init__(self, data_cfg, scope='train'):
        super().__init__(data_cfg, scope)

    def build_dataset(self):
        if self.data_cfg['name'] in ['KITTI2012', 'KITTI2015']:
            from data.reader.kitti_reader import KittiReader
            self.dataset = KittiReader(self.data_cfg['root'], self.data_cfg[f'{self.scope}_list'])
        elif self.data_cfg['name'] == 'SceneFlow':
            from data.reader.sceneflow_reader import FlyingThings3DSubsetReader
            self.dataset = FlyingThings3DSubsetReader(self.data_cfg['root'], self.data_cfg[f'{self.scope}_list'])
        elif self.data_cfg['name'] == 'Middlebury':
            from data.reader.middlebury_reader import MiddleburyReader
            self.dataset = MiddleburyReader(self.data_cfg['root'], self.data_cfg[f'{self.scope}_list'])
        elif self.data_cfg['name'] == 'ETH3D':
            from data.reader.eth3d_reader import ETH3DReader
            self.dataset = ETH3DReader(self.data_cfg['root'], self.data_cfg[f'{self.scope}_list'])
        else:
            name = self.data_cfg['name']
            raise NotImplementedError(f'{name} dataset is not supported yet.')
        self.build_transform()

    def build_transform(self, size=None):
        transform_config = self.data_cfg['transform']
        config = transform_config['train'] if self.is_train else transform_config['test']
        mean, std = config['mean'], config['std']
        if size is not None:
            size = size
        else:
            size = config['size']
        if self.is_train:
            transform = ST.Compose([
                ST.RandomHorizontalFlip(),
                ST.RandomCrop(size),
                ST.GetValidDispNOcc(),  # must be used after Crop
                ST.GetValidDisp(192),
                ST.TransposeImage(),
                ST.ToTensor(),
                ST.NormalizeImage(mean, std),
            ])
        else:
            transform = ST.Compose([
                ST.CenterCrop(size),
                ST.GetValidDispNOcc(),  # must be used after Crop
                ST.GetValidDisp(192),
                ST.TransposeImage(),
                ST.ToTensor(),
                ST.NormalizeImage(mean, std),
            ])
        self.transform = transform

    def __getitem__(self, indexs):
        batch_result = {}
        if self.is_train:
            crop_height = random.randint(256, 512)
            crop_width = random.randint(640, 960)
            self.build_transform([crop_height, crop_width])
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
