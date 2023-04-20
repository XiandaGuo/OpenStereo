import random

import numpy as np
import torch
import torchvision.transforms.functional as TF


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, sample):
        for k in sample.keys():
            if sample[k] is not None and isinstance(sample[k], np.ndarray):
                # Convert the numpy array to a PyTorch Tensor and set its datatype to float32
                sample[k] = torch.from_numpy(sample[k].copy()).to(torch.float32)
        return sample


class TestCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        crop_h, crop_w = self.size
        crop_h = min(h, crop_h)  # ensure crop_h is within the bounds of the image
        crop_w = min(w, crop_w)  # ensure crop_w is within the bounds of the image

        for k in sample.keys():
            # crop the specified arrays to the desired size
            if k in ['left', 'right', 'disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                sample[k] = sample[k][h - crop_h:h, w - crop_w: w]
        return sample


class CropOrPad(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        th, tw = self.size
        if th > h or tw > w:
            # pad the arrays with zeros to the desired size
            pad_left = 0
            pad_right = tw - w
            pad_top = th - h
            pad_bottom = 0
            for k in sample.keys():
                if k in ['left', 'right']:
                    sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant',
                                       constant_values=0)
                elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                    sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant',
                                       constant_values=0)
        else:
            # crop the arrays to the desired size
            for k in sample.keys():
                if k in ['left', 'right', 'disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                    sample[k] = sample[k][h - th:h, w - tw: w]
        return sample


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):

        h, w = sample['left'].shape[:2]
        th, tw = self.size
        tw = min(w, tw)  # ensure tw is within the bounds of the image
        th = min(h, th)  # ensure th is within the bounds of the image

        x1 = (w - tw) // 2  # compute the left edge of the centered rectangle
        y1 = (h - th) // 2  # compute the top edge of the centered rectangle

        for k in sample.keys():
            # crop the specified arrays to the centered rectangle
            if k in ['left', 'right']:
                sample[k] = sample[k][y1: y1 + th, x1: x1 + tw, :]
            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                sample[k] = sample[k][y1: y1 + th, x1: x1 + tw]
        return sample


class StereoPad(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        th, tw = self.size

        h = min(h, th)  # ensure h is within the bounds of the image
        w = min(w, tw)  # ensure w is within the bounds of the image

        pad_left = 0
        pad_right = tw - w
        pad_top = th - h
        pad_bottom = 0
        # apply pad for left, right, disp image, and occ mask
        for k in sample.keys():
            if k in ['left', 'right']:
                sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant',
                                   constant_values=0)
            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant',
                                   constant_values=0)
        return sample


class DivisiblePad(object):
    def __init__(self, by):
        self.by = by

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        if h % self.by != 0:
            pad_top = h + self.by - h % self.by - h
        else:
            pad_top = 0
        if w % self.by != 0:
            pad_right = w + self.by - w % self.by - w
        else:
            pad_right = 0
        pad_left = 0
        pad_bottom = 0

        # apply pad for left, right, disp image, and occ mask
        for k in sample.keys():
            if k in ['left', 'right']:
                sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant',
                                   constant_values=0)
            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant',
                                   constant_values=0)
        sample['pad'] = [pad_top, pad_right, 0, 0]
        return sample


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        return self.horizontal_flip(sample) if self.p < random.random() else sample

    @staticmethod
    def horizontal_flip(sample):
        img_left = sample['left']  # (3, H, W)
        img_right = sample['right']  # (3, H, W)
        disp_left = sample['disp']  # (H, W)
        disp_right = sample['disp_right']  # (H, W)

        left_flipped = img_left[:, ::-1]
        right_flipped = img_right[:, ::-1]
        img_left = right_flipped
        img_right = left_flipped
        disp = disp_right[:, ::-1]
        disp_right = disp_left[:, ::-1]

        sample['left'] = img_left
        sample['right'] = img_right
        sample['disp'] = disp
        sample['disp_right'] = disp_right

        if 'occ_mask' in sample.keys():
            occ_left = sample['occ_mask']  # (H, W)
            occ_right = sample['occ_mask_right']  # (H, W)
            occ = occ_right[:, ::-1]
            occ_right = occ_left[:, ::-1]
            sample['occ_mask'] = occ
            sample['occ_mask_right'] = occ_right
        return sample


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        crop_height, crop_width = self.size
        height, width = sample['left'].shape[:2]  # (H, W, 3)
        if crop_width >= width or crop_height >= height:
            return sample
        else:
            x1, y1, x2, y2 = self.get_random_crop_coords(height, width, crop_height, crop_width)
            for k in sample.keys():
                if k in ['left', 'right']:
                    sample[k] = sample[k][y1: y2, x1: x2, :]
                elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                    sample[k] = sample[k][y1: y2, x1: x2]

            return sample

    @staticmethod
    def get_random_crop_coords(height, width, crop_height, crop_width):
        """
        get coordinates for cropping

        :param height: image height, int
        :param width: image width, int
        :param crop_height: crop height, int
        :param crop_width: crop width, int
        :return: xy coordinates
        """
        y1 = random.randint(0, height - crop_height)
        y2 = y1 + crop_height
        x1 = random.randint(0, width - crop_width)
        x2 = x1 + crop_width
        return x1, y1, x2, y2

    @staticmethod
    def crop(img, x1, y1, x2, y2):
        """
        crop image given coordinates

        :param img: input image, [H,W,3]
        :param x1: coordinate, int
        :param y1: coordinate, int
        :param x2: coordinate, int
        :param y2: coordinate, int
        :return: cropped image
        """
        img = img[y1:y2, x1:x2]
        return img


class GetValidDisp(object):
    def __init__(self, max_disp):
        self.max_disp = max_disp

    def __call__(self, sample):
        disp = sample['disp']
        disp[disp > self.max_disp] = 0
        disp[disp < 0] = 0
        sample.update({
            'disp': disp,
        })
        if 'disp_right' in sample.keys():
            disp_right = sample['disp_right']
            disp_right[disp_right > self.max_disp] = 0
            disp_right[disp_right < 0] = 0
            sample.update({
                'disp_right': disp_right
            })

        return sample


class GetValidDispNOcc(object):
    def __call__(self, sample):
        w = sample['disp'].shape[-1]
        occ_mask = self.compute_left_occ_region(w, sample['disp'])
        sample['occ_mask'][occ_mask] = True  # update
        sample['occ_mask'] = np.ascontiguousarray(sample['occ_mask'])
        try:
            occ_mask = self.compute_right_occ_region(w, sample['disp_right'])
            sample['occ_mask_right'][occ_mask] = 1
            sample['occ_mask_right'] = np.ascontiguousarray(sample['occ_mask_right'])
        except KeyError:
            # print('No disp mask right, check if dataset is KITTI')
            sample['occ_mask_right'] = np.zeros_like(occ_mask).astype(np.bool)

        # remove invalid disp from occ mask
        occ_mask = sample['occ_mask']
        sample['disp'][occ_mask] = 0
        sample['disp'] = np.ascontiguousarray(sample['disp'], dtype=np.float32)
        return sample

    @staticmethod
    def compute_left_occ_region(w, disp):
        """
        Compute occluded region on the left image border

        :param w: image width
        :param disp: left disparity
        :return: occ mask
        """

        coord = np.linspace(0, w - 1, w)[None,]  # 1xW
        shifted_coord = coord - disp
        occ_mask = shifted_coord < 0  # occlusion mask, 1 indicates occ
        return occ_mask

    @staticmethod
    def compute_right_occ_region(w, disp):
        """
        Compute occluded region on the right image border

        :param w: image width
        :param disp: right disparity
        :return: occ mask
        """
        coord = np.linspace(0, w - 1, w)[None,]  # 1xW
        shifted_coord = coord + disp
        occ_mask = shifted_coord > w  # occlusion mask, 1 indicates occ

        return occ_mask


class TransposeImage(object):
    def __call__(self, sample):
        sample['left'] = sample['left'].transpose((2, 0, 1))
        sample['right'] = sample['right'].transpose((2, 0, 1))
        return sample


class NormalizeImage(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['left'] = self.normalize(sample['left'], self.mean, self.std)
        sample['right'] = self.normalize(sample['right'], self.mean, self.std)
        return sample

    @staticmethod
    def normalize(img, mean, std):
        return TF.normalize(img / 255.0, mean=mean, std=std)
