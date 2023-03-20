import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .readpfm import readPFM


def pil_loader(path):
    return np.array(Image.open(path).convert('RGB'), dtype=np.float32)


def cv2_loader(path):
    return cv2.imread(path)


def pfm_disp_loader(path):
    return readPFM(path)[0].astype(np.float32)


def png_disp_loader(path):
    return np.array(Image.open(path), dtype=np.float32)


class BaseReader(Dataset):
    def __init__(self, root, list_file, image_reader='PIL', disp_reader='PIL', right_disp=True, occ_mask=False):
        self.root = root
        self.list_file = list_file
        self.image_reader_type = image_reader
        self.disp_reader_type = disp_reader
        self.return_right_disp = right_disp
        self.return_occ_mask = occ_mask
        self.data_list = self.load_anno()
        self.image_loader = self.build_image_loader()
        self.disp_loader = self.build_disp_loader()

    def load_anno(self):
        data_list = []
        with open(self.list_file, 'r') as fp:
            data_list.extend([x.strip().split(' ') for x in fp.readlines()])
        return data_list

    def item_loader(self, item):
        """
        Load a single item from the dataset.
        Args:
            item: list of str, [left_img_path, right_img_path, disp_img_path] or
            any other formate you defined in load_anno.

        Returns:
            sample: dict, {'left': left_img, 'right': right_img, 'disp': disp_img}
            or any other key-value your net needs and be sure that your transform
            function can handle.
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        item = self.data_list[idx]
        sample = self.item_loader(item)
        return sample

    def __len__(self):
        return len(self.data_list)

    def __repr__(self):
        repr_str = '{}\n'.format(self.__class__.__name__)
        repr_str += ' ' * 4 + 'Data root: {}\n'.format(self.root)
        repr_str += ' ' * 4 + 'Anno file: {}\n'.format(self.list_file)
        repr_str += ' ' * 4 + 'Data length: {}\n'.format(self.__len__())
        return repr_str

    def build_image_loader(self):
        if self.image_reader_type == 'PIL':
            return pil_loader
        elif self.image_reader_type == 'CV2':
            return cv2_loader
        else:
            raise NotImplementedError('Image reader type not supported: {}'.format(self.image_reader_type))

    def build_disp_loader(self):
        if self.disp_reader_type == 'PIL':
            return png_disp_loader
        elif self.disp_reader_type == 'PFM':
            return pfm_disp_loader
        else:
            raise NotImplementedError('Disp reader type not supported: {}'.format(self.disp_reader_type))
