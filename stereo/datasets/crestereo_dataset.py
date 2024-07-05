
import os
import numpy as np
from PIL import Image
from .dataset_template import DatasetTemplate
import cv2


class CREStereoDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.return_right_disp = self.data_info.RETURN_RIGHT_DISP

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, left_disp_path = full_paths

        left_img = Image.open(left_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)

        right_img = Image.open(right_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)

        left_disp = cv2.imread(left_disp_path, cv2.IMREAD_UNCHANGED)
        left_disp = left_disp.astype(np.float32) / 32

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': left_disp
        }

        if self.return_right_disp:
            right_disp_path = left_disp_path.replace('left','right')
            right_disp = cv2.imread(right_disp_path, cv2.IMREAD_UNCHANGED)
            right_disp = right_disp.astype(np.float32) / 32
            sample['disp_right'] = right_disp     

        sample = self.transform(sample)
        sample['index'] = idx
        sample['name'] = left_path

        return sample
    