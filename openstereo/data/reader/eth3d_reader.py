import os

import numpy as np
from PIL import Image

from .base_reader import BaseReader


class ETH3DReader(BaseReader):
    def __init__(self, root, list_file, image_reader='PIL', disp_reader='PFM'):
        super().__init__(root, list_file, image_reader, disp_reader)
        assert disp_reader == 'PFM', 'ETH3D Disp only support PFM format'

    def item_loader(self, item):
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths
        left_img = self.image_loader(left_img_path)
        right_img = self.image_loader(right_img_path)
        disp_img = self.disp_loader(disp_img_path)  # PFM
        # remove invalid values
        disp_img[disp_img == np.inf] = 0
        occ_img = disp_img_path.replace('disp0GT.pfm', 'mask0nocc.png')
        occ_mask = np.ascontiguousarray(Image.open(occ_img).convert('L'), dtype=np.float32).flatten() == 255
        occ_mask = np.reshape(occ_mask, disp_img.shape)
        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
            'occ_mask': occ_mask,
            'left_path': left_img_path,
        }
        return sample


if __name__ == '__main__':
    dataset = ETH3DReader(root='../../data/ETH3D', list_file='../../../datasets/ETH3D/ETH3D_train.txt')
    print(dataset)
    for i in range(len(dataset)):
        sample = dataset[i]
        disp = sample['disp']
        disp = disp.squeeze()
        print(disp.max(), disp.min())
