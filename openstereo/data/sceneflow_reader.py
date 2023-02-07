import os

import numpy as np
import torch
from PIL import Image

from data.base_reader import BaseReader
from data.readpfm import readPFM


class SceneFlowReader(BaseReader):
    def __init__(self, root, list_file):
        super().__init__(root, list_file)

    def item_loader(self, item):
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32).transpose(2, 0, 1)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32).transpose(2, 0, 1)
        disp_img, _ = readPFM(disp_img_path)
        disp_img = disp_img.astype(np.float32)
        disp_img = disp_img[np.newaxis, ...]
        sample = {
            'left': torch.from_numpy(left_img),
            'right': torch.from_numpy(right_img),
            'disp': torch.from_numpy(disp_img),
            'original_size': left_img.shape[-2:],
        }
        return sample


if __name__ == '__main__':
    dataset = SceneFlowReader(root='../../data/sceneflow', list_file='../../datasets/sceneflow/train_val.txt')
    print(dataset)
    sample = dataset[0]
    print(sample['left'].shape, sample['right'].shape, sample['disp'].shape)
