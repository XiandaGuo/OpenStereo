import os
import numpy as np
from PIL import Image
from stereo.datasets.dataset_utils.readpfm import readpfm
from .dataset_template import DatasetTemplate
from stereo.utils.common_utils import get_pos_fullres


class MiddleburyDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        if hasattr(self.data_info, 'RETURN_POS'):
            self.retrun_pos = self.data_info.RETURN_POS
        else:
            self.retrun_pos = False

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths

        left_img = Image.open(left_img_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)
        right_img = Image.open(right_img_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)
        disp_img = readpfm(disp_img_path)[0].astype(np.float32)
        disp_img[disp_img == np.inf] = 0

        occ_mask_path = left_img_path.replace('im0.png', 'mask0nocc.png')
        occ_mask = Image.open(occ_mask_path).convert('L')
        occ_mask = np.array(occ_mask, dtype=np.float32)
        occ_mask = occ_mask != 255.0

        sample = {
            'left': left_img,  # [H, W, 3]
            'right': right_img,  # [H, W, 3]
            'disp': disp_img,  # [H, W]
            'occ_mask': occ_mask  # [H, W]
        }

        if self.retrun_pos and self.mode == 'training':
            sample['pos'] = get_pos_fullres(800, sample['left'].shape[1], sample['left'].shape[0])

        sample = self.transform(sample)
        sample['index'] = idx
        sample['name'] = left_img_path
        return sample
