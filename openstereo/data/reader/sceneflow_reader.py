import os

import numpy as np
from PIL import Image

from .base_reader import BaseReader


class SceneFlowReader(BaseReader):
    def __init__(self, root, list_file, image_reader='PIL', disp_reader='PFM', right_disp=True):
        super().__init__(root, list_file, image_reader, disp_reader, right_disp, occ_mask=False)
        assert disp_reader == 'PFM', 'SceneFlow Disp only support PFM format.'

    def item_loader(self, item):
        full_paths = [os.path.join(self.root, x) for x in item[0:3]]
        left_img_path, right_img_path, disp_img_path = full_paths
        left_img = self.image_loader(left_img_path)
        right_img = self.image_loader(right_img_path)
        disp_img = self.disp_loader(disp_img_path)
        disp_img = disp_img.astype(np.float32)
        sample = {
            'left': left_img,  # [H, W, 3]
            'right': right_img,  # [H, W, 3]
            'disp': disp_img,  # [H, W]
        }
        if self.return_right_disp:
            disp_img_right_path = disp_img_path.replace('left', 'right')
            disp_img_right = self.disp_loader(disp_img_right_path)
            disp_img_right = disp_img_right.astype(np.float32)
            sample['disp_right'] = disp_img_right
        return sample


class FlyingThings3DSubsetReader(BaseReader):
    def __init__(self, root, list_file, image_reader='PIL', disp_reader='PFM', right_disp=True, occ_mask=True):
        super().__init__(root, list_file, image_reader, disp_reader, right_disp, occ_mask)
        assert disp_reader == 'PFM', 'FlyingThings3DSubset Disp Reader only supports PFM format'

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
        left_img = self.image_loader(left_img_path)
        right_img = self.image_loader(right_img_path)
        disp_img = self.disp_loader(disp_img_path)
        disp_img = np.nan_to_num(disp_img, nan=0.0)  # replace nan with 0
        disp_img_right = self.disp_loader(disp_img_right_path)
        disp_img_right = np.nan_to_num(disp_img_right, nan=0.0)  # replace nan with 0
        sample = {
            'left': left_img,
            'right': right_img,
            'disp': -1 * disp_img,
            'disp_right': disp_img_right,
        }
        if self.return_occ_mask:
            occ = np.array(Image.open(occ_path)).astype(np.bool)
            occ_right = np.array(Image.open(occ_right_path)).astype(np.bool)
            sample.update({
                'occ_mask': occ,
                'occ_mask_right': occ_right
            })
        return sample
