import random

import numpy as np
import torch
from torch.utils.data import Dataset

from . import stereo_trans as ST


class StereoBatchDataset(Dataset):
    def __init__(self, data_cfg, scope='train'):
        super().__init__()
        self.data_cfg = data_cfg
        self.is_train = scope == 'train'
        self.scope = scope.lower()
        self.dataset = None
        self.transform = None
        self.image_reader_type = data_cfg['image_reader'] if 'image_reader' in data_cfg else "PIL"
        self.disp_reader_type = data_cfg['disp_reader'] if 'disp_reader' in data_cfg else "PIL"
        self.return_right_disp = data_cfg['return_right_disp'] if 'return_right_disp' in data_cfg else True
        self.return_occ_mask = data_cfg['return_occ_mask'] if 'return_occ_mask' in data_cfg else False

        # for batch uniform
        self.batch_uniform = data_cfg['batch_uniform'] if 'batch_uniform' in data_cfg else False
        self.random_type = data_cfg['random_type'] if 'random_type' in data_cfg else None
        self.w_range = data_cfg['w_range'] if 'w_range' in data_cfg else None
        self.h_range = data_cfg['h_range'] if 'h_range' in data_cfg else None
        self.random_crop_index = None  # for batch uniform random crop, record the index of the crop transform operator
        self.build_dataset()

    def build_dataset(self):
        if self.data_cfg['name'] in ['KITTI2012', 'KITTI2015']:
            if self.scope == 'test_kitti':
                from data.reader.kitti_reader import KittiTestReader
                self.disp_reader_type = 'PIL'
                self.dataset = KittiTestReader(
                    self.data_cfg['root'],
                    self.data_cfg['test_list'],
                    self.image_reader_type,
                    self.disp_reader_type,
                    right_disp=False,
                    use_noc=False,
                )
            else:
                from data.reader.kitti_reader import KittiReader
                self.disp_reader_type = 'PIL'
                self.dataset = KittiReader(
                    self.data_cfg['root'],
                    self.data_cfg[f'{self.scope}_list'],
                    self.image_reader_type,
                    self.disp_reader_type,
                    right_disp=self.return_right_disp,
                    use_noc=self.data_cfg['use_noc'] if 'use_noc' in self.data_cfg else False,  # NOC disp or OCC disp
                )
        elif self.data_cfg['name'] == 'FlyingThings3DSubset':
            from data.reader.sceneflow_reader import FlyingThings3DSubsetReader
            self.disp_reader_type = 'PFM'
            self.return_right_disp = True
            self.return_occ_mask = True
            self.dataset = FlyingThings3DSubsetReader(
                self.data_cfg['root'],
                self.data_cfg[f'{self.scope}_list'],
                self.image_reader_type,
                self.disp_reader_type,
                right_disp=self.return_right_disp,
                occ_mask=self.return_occ_mask
            )
        elif self.data_cfg['name'] == 'SceneFlow':
            from data.reader.sceneflow_reader import SceneFlowReader
            self.disp_reader_type = 'PFM'
            self.dataset = SceneFlowReader(
                self.data_cfg['root'],
                self.data_cfg[f'{self.scope}_list'],
                self.image_reader_type,
                self.disp_reader_type,
                right_disp=self.return_right_disp,
            )
        elif self.data_cfg['name'] == 'DrivingStereo':
            from data.reader.driving_reader import DrivingReader
            self.return_right_disp = False
            self.return_occ_mask = False
            self.disp_reader_type = 'PIL'
            self.dataset = DrivingReader(
                self.data_cfg['root'],
                self.data_cfg[f'{self.scope}_list'],
                self.image_reader_type,
                self.disp_reader_type
            )
        elif self.data_cfg['name'] == 'Middlebury':
            from data.reader.middlebury_reader import MiddleburyReader
            self.disp_reader_type = 'PFM'
            self.dataset = MiddleburyReader(
                self.data_cfg['root'],
                self.data_cfg[f'{self.scope}_list'],
                self.image_reader_type,
                self.disp_reader_type
            )
        elif self.data_cfg['name'] == 'ETH3D':
            from data.reader.eth3d_reader import ETH3DReader
            self.disp_reader_type = 'PFM'
            self.dataset = ETH3DReader(
                self.data_cfg['root'],
                self.data_cfg[f'{self.scope}_list'],
                self.image_reader_type,
                self.disp_reader_type
            )
        else:
            name = self.data_cfg['name']
            raise NotImplementedError(f'{name} dataset is not supported yet.')
        self.build_transform()

    def build_transform(self):
        transform_config = self.data_cfg['transform']
        if self.scope == 'test_kitti':
            config = transform_config['test_kitti']
        elif self.scope == 'train':
            config = transform_config['train']
        else:
            config = transform_config['test']
        self.transform = self.build_transform_by_cfg(config)

    def __getitem__(self, indexs):
        # set the image_size for this batch
        if self.batch_uniform:
            base_size = self.transform.transforms[self.random_crop_index].size
            size = self.get_crop_size(base_size)
            self.transform.transforms[self.random_crop_index].size = size

        batch_result = {}

        if isinstance(indexs, int):
            indexs = [indexs]

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
                    tmp = torch.unsqueeze(result[each_item], 0) if torch.is_tensor(result[each_item]) else result[
                        each_item]
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
            if trans['type'] == 'DivisiblePad':
                transform_compose.append(ST.DivisiblePad(trans['by']))
            elif trans['type'] == 'RandomCrop':
                transform_compose.append(ST.RandomCrop(trans['size']))
                self.random_crop_index = len(transform_compose) - 1
            if trans['type'] == 'RandomHorizontalFlip':
                assert self.return_right_disp, 'RandomHorizontalFlip is used, but return_right_disp is False.'
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

    def get_crop_size(self, base_size):
        if self.random_type == 'range':
            w = random.randint(self.w_range[0] * base_size[1], self.w_range[1] * base_size[1])
            h = random.randint(self.h_range[0] * base_size[0], self.h_range[1] * base_size[0])
        elif self.random_type == 'choice':
            w = random.choice(self.w_range)
            h = random.choice(self.h_range)
        else:
            raise NotImplementedError
        return int(h), int(w)

    def __len__(self):
        return len(self.dataset)
