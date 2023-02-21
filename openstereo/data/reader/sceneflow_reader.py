import os

import numpy as np
import torch
from PIL import Image

from data.reader.base_reader import BaseReader
from data.reader.readpfm import readPFM


class SceneFlowReader(BaseReader):
    def __init__(self, root, list_file):
        super().__init__(root, list_file)

    def item_loader(self, item):
        full_paths = [os.path.join(self.root, x) for x in item[0:3]]
        left_img_path, right_img_path, disp_img_path = full_paths
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        disp_img, _ = readPFM(disp_img_path)
        disp_img = disp_img.astype(np.float32)

        disp_img_right, _ = readPFM(disp_img_path.replace('left', 'right'))
        disp_img_right = disp_img_right.astype(np.float32)

        sample = {
            'left': left_img,  # [H, W, 3]
            'right': right_img,  # [H, W, 3]
            'disp': disp_img,  # [H, W]
            'disp_right': disp_img_right,  # [H, W]
        }
        return sample


class FlyingThings3DSubsetReader(BaseReader):
    def __init__(self, root, list_file, return_occ=True):
        super().__init__(root, list_file)
        self.return_occ = return_occ

    def item_loader(self, item):
        """
        :param item: [left_img_path, right_img_path, disp_img_path]
        :return:
        dict: {
            'left': left image, [H, W, 3]
            'right': right image, [H, W, 3]
            'disp': disparity map, [H, W]
            'disp_right': disparity map of right image, [H, W]
            'occ': occlusion map, [H, W]
            'occ_right': occlusion map of right image, [H, W]
            'original_size': original size of the image, [H, W]
        }
        """
        full_paths = [os.path.join(self.root, x) for x in item[0:6]]
        left_img_path, right_img_path, disp_img_path, disp_img_right_path, occ_path, occ_right_path = full_paths
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        disp_img, _ = readPFM(disp_img_path)
        disp_img = disp_img.astype(np.float32)
        disp_img = np.nan_to_num(disp_img, nan=0.0)  # replace nan with 0

        disp_img_right, _ = readPFM(disp_img_right_path)
        disp_img_right = disp_img_right.astype(np.float32)
        disp_img_right = np.nan_to_num(disp_img_right, nan=0.0)  # replace nan with 0
        sample = {
            'left': left_img,
            'right': right_img,
            'disp': -1 * disp_img,
            'disp_right': disp_img_right,
        }
        if self.return_occ:
            occ = np.array(Image.open(occ_path)).astype(np.bool)
            occ_right = np.array(Image.open(occ_right_path)).astype(np.bool)
            sample.update({
                'occ_mask': occ,
                'occ_mask_right': occ_right
            })
        return sample
