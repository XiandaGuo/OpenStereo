import os
import numpy as np
from PIL import Image
from .dataset_template import DatasetTemplate


class TartanAirDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, left_depth_path = full_paths
        # image
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        # disp
        depth = np.load(left_depth_path)
        disparity = 80.0 / depth

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disparity
        }

        sample = self.transform(sample)

        sample['index'] = idx
        sample['name'] = left_img_path
        return sample
