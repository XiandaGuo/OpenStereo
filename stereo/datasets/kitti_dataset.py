import os
import torch.utils.data as torch_data
import numpy as np
from PIL import Image
from .dataset_template import DatasetTemplate
from stereo.utils.common_utils import get_pos_fullres


class KittiDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.return_right_disp = self.data_info.RETURN_RIGHT_DISP
        self.use_noc = self.data_info.get('USE_NOC', False)
        if hasattr(self.data_info, 'RETURN_POS'):
            self.retrun_pos = self.data_info.RETURN_POS
        else:
            self.retrun_pos = False

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths
        if self.use_noc:
            disp_img_path = disp_img_path.replace('disp_occ', 'disp_noc')
        # image
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        # disp
        disp_img = np.array(Image.open(disp_img_path), dtype=np.float32) / 256.0

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
        }
        if self.return_right_disp:
            disp_img_right_path = disp_img_path.replace('c_0', 'c_1')
            disp_img_right = np.array(Image.open(disp_img_right_path), dtype=np.float32) / 256.0
            sample['disp_right'] = disp_img_right

        if self.retrun_pos and self.mode == 'training':
            sample['pos'] = get_pos_fullres(800, sample['left'].shape[1], sample['left'].shape[0])

        sample = self.transform(sample)
        sample['index'] = idx
        sample['name'] = left_img_path

        return sample
