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
        disp_img_right_path = disp_img_path.replace('disp_occ_0', 'disp_occ_1')
        left_img = np.asarray(Image.open(left_img_path), dtype=np.float32)
        right_img = np.asarray(Image.open(right_img_path), dtype=np.float32)
        disp_img = np.asarray(Image.open(disp_img_path), dtype=np.float32) / 256.0
        disp_img_right = np.asarray(Image.open(disp_img_right_path), dtype=np.float32) / 256.0
        # disp_img = disp_img[np.newaxis, ...]
        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
            'disp_right': disp_img_right,
        }
        return sample


if __name__ == '__main__':
    dataset = KittiReader(root='../../data/kitti12', list_file='../../../datasets/kitti12/kitti12_train165.txt')
    print(dataset)
    sample = dataset[0]
    print(sample['left'].shape, sample['right'].shape, sample['disp'].shape)
