import os

import numpy as np
import torch
from PIL import Image

from .base_reader import BaseReader


class KittiReader(BaseReader):
    def __init__(self, root, list_file):
        super().__init__(root, list_file)

    def item_loader(self, item):
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths
        left_img = np.asarray(Image.open(left_img_path), dtype=np.float32).transpose(2, 0, 1)
        right_img = np.asarray(Image.open(right_img_path), dtype=np.float32).transpose(2, 0, 1)
        disp_img = np.asarray(Image.open(disp_img_path), dtype=np.float32) / 256.0
        disp_img = disp_img[np.newaxis, ...]
        sample = {
            'left': torch.from_numpy(left_img),
            'right': torch.from_numpy(right_img),
            'disp': torch.from_numpy(disp_img),
            'original_size': left_img.shape[-2:],
        }
        return sample


if __name__ == '__main__':
    dataset = KittiReader(root='../../data/kitti12', list_file='../../../datasets/kitti12/kitti12_train165.txt')
    print(dataset)
    sample = dataset[0]
    print(sample['left'].shape, sample['right'].shape, sample['disp'].shape)
