# @Time    : 2024/10/27 01:08
# @Author  : zhangchenming
import os
import random
import numpy as np
import cv2
import re
from PIL import Image
from pathlib import Path
from .dataset_template import DatasetTemplate
from .mono import WarpDataset


class RealfillDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.warp = WarpDataset()

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, disp_path = full_paths

        left_img = Image.open(left_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)
        right_img = Image.open(right_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)
        disp_img = np.load(disp_path).astype(np.float32)
        occ_mask = np.zeros_like(disp_img, dtype=bool)

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
            'occ_mask': occ_mask,
        }

        img = cv2.cvtColor(sample['left'], cv2.COLOR_RGB2BGR)
        lsc = cv2.ximgproc.createSuperpixelLSC(img, region_size=10, ratio=0.075)
        lsc.iterate(20)
        label = lsc.getLabels()
        sample['super_pixel_label'] = label.astype(np.int32)

        if self.transform is not None:
            sample = self.transform(sample)

        sample['valid'] = sample['disp'] < 512
        sample['index'] = idx
        sample['name'] = left_path

        return sample
