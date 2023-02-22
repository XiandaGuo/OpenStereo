import os

import numpy as np
import torch
from PIL import Image

from .base_reader import BaseReader
from .readpfm import readPFM


class MiddleburyReader(BaseReader):
    def __init__(self, root, list_file):
        super().__init__(root, list_file)

    def item_loader(self, item):
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        disp_img, _ = readPFM(disp_img_path)
        disp_img[disp_img == np.inf] = 0
        disp_img = disp_img.astype(np.float32)
        # disp_img = disp_img[np.newaxis, ...]
        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
        }
        return sample
