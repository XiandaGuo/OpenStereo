import numbers
import random

import numpy as np
import torch
import torch.nn.functional as F
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
    """
    convert numpy.ndarray to torch.floatTensor, in [Channels, Height, Width]
    """

    def __call__(self, sample):
        for k in sample.keys():
            if sample[k] is not None and isinstance(sample[k], np.ndarray):
                sample[k] = torch.from_numpy(sample[k].copy())

            sample['left'] = sample['left'].float() / 255.0
            sample['right'] = sample['right'].float() / 255.0
        return sample


class CenterCrop(object):
    """Crops the given image at central location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):

        h, w = sample['left'].shape[-2:]
        th, tw = self.size
        if w == tw and h == th:
            return sample

        x1 = (w - tw) // 2
        y1 = (h - th) // 2

        for k in sample.keys():
            if sample[k] is not None and isinstance(sample[k], (np.ndarray, torch.Tensor)):
                sample[k] = sample[k][:, y1: y1 + th, x1: x1 + tw]
        return sample


class RandomCrop(object):
    """Crops the given image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):

        h, w = sample['left'].shape[-2:]
        th, tw = self.size
        if w == tw and h == th:
            return sample

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for k in sample.keys():
            if sample[k] is not None and isinstance(sample[k], (np.ndarray, torch.Tensor)):
                sample[k] = sample[k][:, y1: y1 + th, x1: x1 + tw]
        return sample


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['left'] = TF.normalize(sample['left'], mean=self.mean, std=self.std)
        sample['right'] = TF.normalize(sample['right'], mean=self.mean, std=self.std)
        return sample


class StereoPad(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        h, w = sample['left'].shape[-2:]
        th, tw = self.size
        if w == tw and h == th:
            return sample
        pad_left = 0
        pad_right = tw - w
        pad_top = th - h
        pad_bottom = 0

        # apply pad for left, right, disp image
        for k in sample.keys():
            if sample[k] is not None and isinstance(sample[k], (np.ndarray, torch.Tensor)):
                sample[k] = F.pad(sample[k], (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return sample
