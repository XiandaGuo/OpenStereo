import os
import numpy as np
from PIL import Image
from .dataset_template import DatasetTemplate


class UnrealStereo4KDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.return_right_disp = self.data_info.RETURN_RIGHT_DISP

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, disp_left_path = full_paths

        left_img = Image.open(left_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)

        right_img = Image.open(right_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)

        disp_left = np.load(disp_left_path, mmap_mode='c')

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_left
        }

        if self.return_right_disp:
            disp_right_path = disp_left_path.replace('Image0','Image1')
            disp_right = np.load(disp_right_path, mmap_mode='c')
            sample['disp_right'] = disp_right

        sample = self.transform(sample)
        sample['index'] = idx
        sample['name'] = left_path

        return sample
