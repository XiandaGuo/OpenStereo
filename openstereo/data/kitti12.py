import os

import numpy as np
import torch
from imageio.v3 import imread

from data.base_reader import BaseReader


class Kitti12(BaseReader):
    def __init__(self, root, list_file):
        super().__init__(root, list_file)

    def item_loader(self, item):
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths
        left_img = imread(left_img_path).transpose(2, 0, 1).astype(np.float32)
        right_img = imread(right_img_path).transpose(2, 0, 1).astype(np.float32)
        disp_img = imread(disp_img_path).astype(np.float32) / 256.0
        disp_img = disp_img[np.newaxis, ...]
        sample = {
            'left': torch.from_numpy(left_img),
            'right': torch.from_numpy(right_img),
            'disp': torch.from_numpy(disp_img)
        }
        return sample

    # def get_transform(self):
    #     if self.train:
    #         img_transform = transforms.Compose([
    #             transforms.Resize((384, 1248)),
    #             transforms.RandomCrop((256, 512)),
    #             transforms.RandomHorizontalFlip(),
    #             # transforms.ToTensor(),
    #         ])
    #
    #         disp_transform = transforms.Compose([
    #             transforms.Resize((384, 1248)),
    #             transforms.RandomCrop((256, 512)),
    #             transforms.RandomHorizontalFlip(),
    #             # transforms.ToTensor(),
    #         ])
    #
    #         def transform(sample):
    #             sample['left'] = img_transform(sample['left'])
    #             sample['right'] = img_transform(sample['right'])
    #             sample['disp'] = disp_transform(sample['disp']) / 256.0
    #             return sample
    #
    #         return transform
    #     else:
    #         img_transform = transforms.Compose([
    #             transforms.Resize((256, 512)),
    #             # transforms.ToTensor(),
    #         ])
    #
    #         disp_transform = transforms.Compose([
    #             transforms.Resize((256, 512)),
    #             # transforms.ToTensor(),
    #         ])
    #
    #         def transform(sample):
    #             sample['left'] = img_transform(sample['left'])
    #             sample['right'] = img_transform(sample['right'])
    #             sample['disp'] = disp_transform(sample['disp'])
    #             return sample
    #
    #         return transform


if __name__ == '__main__':
    dataset = Kitti12(root='../../data/kitti12', list_file='../../datasets/kitti12/train.txt')
    print(dataset)
    sample = dataset[0]
    print(sample)
    print(dataset[0]['left'].shape)
