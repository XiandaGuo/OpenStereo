import os

from .base_reader import BaseReader


class DrivingReader(BaseReader):
    def __init__(self, root, list_file, image_reader='PIL', disp_reader='PIL'):
        super().__init__(root, list_file, image_reader, disp_reader)

    def item_loader(self, item):
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths
        left_img = self.image_loader(left_img_path)
        right_img = self.image_loader(right_img_path)
        disp_img = self.disp_loader(disp_img_path)
        # for validation, full resolution disp need to be divided by 128 instead of 256
        full = True if 'full' in disp_img_path else False
        scale = 128.0 if full else 256.0
        disp_img = disp_img / scale
        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img
        }
        return sample
