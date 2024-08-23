
import os
import numpy as np
from PIL import Image
import cv2
from .dataset_template import DatasetTemplate


class ArgoverseDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.return_right_disp = self.data_info.RETURN_RIGHT_DISP

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, disp_path = full_paths

        left_img = cv2.cvtColor(cv2.imread(left_path), cv2.COLOR_BGR2RGB)

        right_img = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)

        disp_img = cv2.imread(disp_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        disp_img = np.float32(disp_img) / 256.0


        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
        }
        sample = self.transform(sample)
        sample['index'] = idx
        sample['name'] = left_path

        return sample
