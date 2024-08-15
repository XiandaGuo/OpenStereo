# @Time    : 2023/8/29 02:01
# @Author  : zhangchenming
import random
import torch
import numpy as np
import cv2
from torchvision.transforms.functional import normalize
from PIL import Image
from torchvision.transforms import ColorJitter


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class TransposeImage(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        sample['left'] = sample['left'].transpose((2, 0, 1))
        sample['right'] = sample['right'].transpose((2, 0, 1))
        return sample


class ToTensor(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        for k in sample.keys():
            if isinstance(sample[k], np.ndarray):
                sample[k] = torch.from_numpy(sample[k].copy()).to(torch.float32)
        return sample


class NormalizeImage(object):
    def __init__(self, config):
        self.mean = config.MEAN
        self.std = config.STD

    def __call__(self, sample):
        sample['left'] = normalize(sample['left'] / 255.0, self.mean, self.std)
        sample['right'] = normalize(sample['right'] / 255.0, self.mean, self.std)
        return sample


class RandomCrop(object):
    def __init__(self, config):
        self.crop_size = config.SIZE
        self.base_size = config.SIZE
        self.y_jitter = config.get('Y_JITTER', False)

    def __call__(self, sample):
        crop_height, crop_width = self.crop_size
        height, width = sample['left'].shape[:2]  # (H, W, 3)
        # crop_height = min(height, crop_height)
        # crop_width = min(width, crop_width)
        if crop_width > width or crop_height > height:
            return sample

        n_pixels = 2 if (self.y_jitter and np.random.rand() < 0.5) else 0
        y1 = random.randint(n_pixels, height - crop_height - n_pixels)
        x1 = random.randint(0, width - crop_width)
        y2 = y1 + np.random.randint(-n_pixels, n_pixels + 1)

        for k in sample.keys():
            if k in ['right']:
                sample[k] = sample[k][y2: y2 + crop_height, x1: x1 + crop_width]
            elif k in ['pos']:
                sample[k] = sample[k][:, y1: y1 + crop_height, x1: x1 + crop_width]
            else:
                sample[k] = sample[k][y1: y1 + crop_height, x1: x1 + crop_width]

        return sample


class RandomScale(object):
    def __init__(self, config):
        self.config = config
        self.crop_size = config.SIZE
        self.min_scale = config.MIN_SCALE
        self.max_scale = config.MAX_SCALE
        self.scale_prob = config.SCALE_PROB
        self.stretch_prob = config.STRETCH_PROB

        self.max_stretch = 0.2

    def __call__(self, sample):
        ht, wd = sample['left'].shape[:2]
        min_scale = np.maximum((self.crop_size[0] + 8) / float(ht), (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.scale_prob:
            for k in sample.keys():
                if k in ['left', 'right']:
                    sample[k] = cv2.resize(sample[k], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

                elif k in ['disp', 'disp_right']:
                    sample[k] = cv2.resize(sample[k], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                    sample[k] = sample[k] * scale_x

        return sample


class RandomSparseScale(object):
    def __init__(self, config):
        self.crop_size = config.SIZE
        self.min_scale = config.MIN_SCALE
        self.max_scale = config.MAX_SCALE
        self.scale_prob = config.SCALE_PROB

    def __call__(self, sample):
        ht, wd = sample['left'].shape[:2]
        min_scale = np.maximum((self.crop_size[0] + 1) / float(ht), (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.scale_prob:
            for k in sample.keys():
                if k in ['left', 'right']:
                    sample[k] = cv2.resize(sample[k], None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

                elif k in ['disp', 'disp_right']:
                    sample[k] = self.sparse_disp_map_reisze(sample[k], fx=scale_x, fy=scale_y)

        return sample

    @staticmethod
    def sparse_disp_map_reisze(disp, fx=1.0, fy=1.0):
        ht, wd = disp.shape[:2]
        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))
        valid = disp > 0.0
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)
        coords = coords.reshape(-1, 2).astype(np.float32)  # 坐标[w, h]

        disp = disp.reshape(-1).astype(np.float32)
        valid = valid.reshape(-1)
        coords = coords[valid]  # disp > 0 的坐标
        disp = disp[valid]  # disp > 0 的值
        coords = coords * [fx, fy]  # resize 后的坐标
        disp = disp * fx  # risize 后的值

        xx = np.round(coords[:, 0]).astype(np.int32)
        yy = np.round(coords[:, 1]).astype(np.int32)
        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        disp = disp[v]

        resized_disp = np.zeros([ht1, wd1], dtype=np.float32)
        resized_disp[yy, xx] = disp

        return resized_disp


class RandomErase(object):
    def __init__(self, config):
        self.eraser_aug_prob = config.PROB
        self.max_erase_time = config.MAX_TIME
        self.bounds = config.BOUNDS

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, self.max_erase_time + 1)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(self.bounds[0], self.bounds[1])
                dy = np.random.randint(self.bounds[0], self.bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        sample['left'] = img1
        sample['right'] = img2
        return sample


class StereoColorJitter(object):
    def __init__(self, config):
        self.brightness = config.BRIGHTNESS
        self.contrast = config.CONTRAST
        self.saturation = config.SATURATION
        self.hue = config.HUE
        self.asymmetric_color_aug_prob = config.ASYMMETRIC_PROB
        self.color_jitter = ColorJitter(brightness=self.brightness, contrast=self.contrast,
                                        saturation=self.saturation, hue=self.hue / 3.14)

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.color_jitter(Image.fromarray(img1.astype(np.uint8))), dtype=np.uint8)
            img2 = np.array(self.color_jitter(Image.fromarray(img2.astype(np.uint8))), dtype=np.uint8)
        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0).astype(np.uint8)
            image_stack = np.array(self.color_jitter(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        sample['left'] = img1
        sample['right'] = img2

        return sample


class RightTopPad(object):
    def __init__(self, config):
        self.size = config.SIZE

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
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                sample[k] = np.pad(sample[k], pad_width, 'edge')

            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right]])
                sample[k] = np.pad(sample[k], pad_width, 'constant', constant_values=0)

        return sample


class DivisiblePad(object):
    def __init__(self, config):
        self.by = config.BY
        self.mode = config.get('MODE', 'tr')

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        if h % self.by != 0:
            pad_h = self.by - h % self.by
        else:
            pad_h = 0
        if w % self.by != 0:
            pad_w = self.by - w % self.by
        else:
            pad_w = 0

        if self.mode == 'round':
            pad_top = pad_h // 2
            pad_right = pad_w // 2
            pad_bottom = pad_h - (pad_h // 2)
            pad_left = pad_w - (pad_w // 2)
        elif self.mode == 'tr':
            pad_top = pad_h
            pad_right = pad_w
            pad_bottom = 0
            pad_left = 0
        else:
            raise Exception('no DivisiblePad mode')

        # apply pad for left, right, disp image, and occ mask
        for k in sample.keys():
            if k in ['left', 'right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                sample[k] = np.pad(sample[k], pad_width, 'edge')

            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right]])
                sample[k] = np.pad(sample[k], pad_width, 'constant', constant_values=0)

        sample['pad'] = [pad_top, pad_right, pad_bottom, pad_left]
        return sample


class RandomFlip(object):
    def __init__(self, config):
        self.config = config
        self.flip_type = config.FLIP_TYPE
        self.prob = config.PROB

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']
        disp_right = sample['disp_right']

        if np.random.rand() < self.prob and self.flip_type == 'horizontal':  # 水平翻转
            img1 = np.ascontiguousarray(img1[:, ::-1])
            img2 = np.ascontiguousarray(img2[:, ::-1])
            disp = np.ascontiguousarray(disp[:, ::-1] * -1.0)

        if np.random.rand() < self.prob and self.flip_type == 'horizontal_swap':  # 水平翻转并交换
            tmp = np.ascontiguousarray(img1[:, ::-1])
            img1 = np.ascontiguousarray(img2[:, ::-1])
            disp = np.ascontiguousarray(disp_right[:, ::-1])
            img2 = tmp

        if np.random.rand() < self.prob and self.flip_type == 'vertical':  # 垂直翻转
            img1 = np.ascontiguousarray(img1[::-1, :])
            img2 = np.ascontiguousarray(img2[::-1, :])
            disp = np.ascontiguousarray(disp[::-1, :])

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = disp
        return sample


class RightBottomCrop(object):
    def __init__(self, config):
        self.size = config.SIZE

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        crop_h, crop_w = self.size
        crop_h = min(h, crop_h)
        crop_w = min(w, crop_w)

        for k in sample.keys():
            sample[k] = sample[k][h - crop_h:, w - crop_w:]
        return sample


class CropOrPad(object):
    def __init__(self, config):
        self.size = config.SIZE
        self.crop_fn = RightBottomCrop(config)
        self.pad_fn = RightTopPad(config)

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        th, tw = self.size
        if th > h or tw > w:
            sample = self.pad_fn(sample)
        else:
            sample = self.crop_fn(sample)

        return sample