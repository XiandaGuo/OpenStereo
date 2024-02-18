import random
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import ColorJitter, functional


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
                    sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'edge')
                elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                    sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant',
                                       constant_values=0)
        else:
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
                sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'edge')
            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)
        return sample


class DivisiblePad(object):
    def __init__(self, by, mode='single'):
        self.by = by
        self.mode = mode

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        if h % self.by != 0:
            pad_h = h + self.by - h % self.by - h
        else:
            pad_h = 0
        if w % self.by != 0:
            pad_w = w + self.by - w % self.by - w
        else:
            pad_w = 0

        if self.mode == 'single':
            pad_top = pad_h
            pad_right = pad_w
            pad_left = 0
            pad_bottom = 0
        elif self.mode == 'double':
            pad_top = pad_h // 2
            pad_right = pad_w // 2
            pad_bottom = pad_h - (pad_h // 2)
            pad_left = pad_w - (pad_w // 2)
        else:
            raise ValueError('Invalid mode: {}'.format(self.mode))

        # apply pad for left, right, disp image, and occ mask
        for k in sample.keys():
            if k in ['left', 'right']:
                sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'edge')
            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                sample[k] = np.pad(sample[k], ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant',
                                   constant_values=0)
        sample['pad'] = [pad_top, pad_right, pad_bottom, pad_left]
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


class ImageColorJitter(object):
    def __init__(self, brightness=0.3, contrast=0.3, saturation=[0.7, 1.3], hue=0.3 / 3.14):
        self.argumentor = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, sample):
        sample['left'] = self.argumentor(sample['left'])
        sample['right'] = self.argumentor(sample['right'])
        return sample


class RandomEraser(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        if np.random.rand() < self.prob:
            img1, img2 = self.eraser_transform(img1, img2)
        sample['left'] = img1
        sample['right'] = img2
        return sample

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        mean_color = np.mean(img2.reshape(-1, 3), axis=0)
        for _ in range(np.random.randint(1, 3)):
            x0 = np.random.randint(0, wd)
            y0 = np.random.randint(0, ht)
            dx = np.random.randint(50, 100)
            dy = np.random.randint(50, 100)
            img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color
        return img1, img2


class ImageAdjustGamma(object):
    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.argumentor = AdjustGamma(gamma_min, gamma_max, gain_min, gain_max)

    def __call__(self, sample):
        sample['left'] = self.argumentor(sample['left'])
        sample['right'] = self.argumentor(sample['right'])
        return sample


class AdjustGamma(object):

    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)


class FlowAugmentor(object):
    def __init__(self, crop_size=[256, 512], min_scale=-0.2, max_scale=0.5, do_flip=False, yjitter=False,
                 saturation_range=[0.6, 1.4], gamma=[1, 1, 1, 1]):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose(
            [ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5 / 3.14),
             AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1.astype(np.uint8))), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2.astype(np.uint8))), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0).astype(np.uint8)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow, flow_right):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow_right = cv2.resize(flow_right, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

            flow = flow * [scale_x, scale_y]
            flow_right = flow_right * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf':  # h-flip
                img1 = np.ascontiguousarray(img1[:, ::-1])
                img2 = np.ascontiguousarray(img2[:, ::-1])
                flow = np.ascontiguousarray(flow[:, ::-1] * [-1.0, 1.0])

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h':  # h-flip for stereo
                tmp = np.ascontiguousarray(img1[:, ::-1])
                img1 = np.ascontiguousarray(img2[:, ::-1])
                img2 = tmp
                flow = np.ascontiguousarray(flow_right[:, ::-1])

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v':  # v-flip
                img1 = np.ascontiguousarray(img1[::-1, :])
                img2 = np.ascontiguousarray(img2[::-1, :])
                flow = np.ascontiguousarray(flow[::-1, :] * [1.0, -1.0])

        if self.yjitter:
            y0 = np.random.randint(2, img1.shape[0] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, img1.shape[1] - self.crop_size[1] - 2)

            y1 = y0 + np.random.randint(-2, 2 + 1)
            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y1:y1 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

            img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']
        disp_right = sample['disp_right']
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)
        flow_right = np.stack([disp_right, np.zeros_like(disp_right)], axis=-1)

        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow, flow_right)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = flow[:1][0]
        return sample


# only for kitti
class SparseFlowAugmentor(object):
    def __init__(self, crop_size=[256, 512], min_scale=-0.2, max_scale=0.5, do_flip=False, yjitter=False,
                 saturation_range=[0.7, 1.3], gamma=[1, 1, 1, 1]):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose(
            [ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3 / 3.14),
             AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf':  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            # no usage in kitti dataset
            # if np.random.rand() < self.h_flip_prob and self.do_flip == 'h':  # h-flip for stereo
            #     tmp = img1[:, ::-1]
            #     img1 = img2[:, ::-1]
            #     img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v':  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow, valid

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']
        valid = sample['disp'] > 0.0

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        img1 = img1[..., :3]
        img2 = img2[..., :3]

        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        valid = torch.from_numpy(valid).float()

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = flow[:1][0]
        sample['flow'] = flow[:1][0]
        sample['valid'] = valid
        return sample


class ColorTransform(object):
    def __init__(self, saturation_range=[0.7, 1.3], gamma=[1, 1, 1, 1]):
        self.photo_aug = Compose(
            [ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3 / 3.14),
             AdjustGamma(*gamma)])

    def __call__(self, sample):
        img1 = sample['left']
        img2 = sample['right']
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(image_stack, dtype=np.uint8)
        image_stack = Image.fromarray(image_stack)
        image_stack = self.photo_aug(image_stack)
        image_stack = np.array(image_stack, dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        sample['left'] = img1
        sample['right'] = img2
        return sample


class EraserTransform(object):
    def __init__(self, eraser_aug_prob=0.5):
        self.eraser_aug_prob = eraser_aug_prob

    def __call__(self, sample):
        ht, wd = sample['left'].shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(sample['right'].reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                sample['right'][y0:y0 + dy, x0:x0 + dx, :] = mean_color
        return sample


class SpatialTransform(object):
    def __init__(self, size, min_scale=0.2, max_scale=0.5, do_flip=False, h_flip_prob=0.5, v_flip_prob=0.1):
        self.crop_size = size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        self.do_flip = do_flip
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def __call__(self, sample):

        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']

        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

            if self.do_flip:
                if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf':  # h-flip
                    img1 = img1[:, ::-1]
                    img2 = img2[:, ::-1]
                    flow = flow[:, ::-1] * [-1.0, 1.0]

                # if np.random.rand() < self.h_flip_prob and self.do_flip == 'h':  # h-flip for stereo
                #     tmp = img1[:, ::-1]
                #     img1 = img2[:, ::-1]
                #     img2 = tmp

                if np.random.rand() < self.v_flip_prob and self.do_flip == 'v':  # v-flip
                    img1 = img1[::-1, :]
                    img2 = img2[::-1, :]
                    flow = flow[::-1, :] * [1.0, -1.0]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        # valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = flow[:, :, :1].squeeze(-1)

        return sample


class RandomFlip(object):
    def __init__(self, do_flip_type='h', h_flip_prob=0.5, v_flip_prob=0.1):
        self.h_flip_prob = h_flip_prob
        self.v_flip_prob = v_flip_prob
        self.do_flip_type = do_flip_type

    def __call__(self, sample):

        img1 = sample['left']
        img2 = sample['right']
        disp = sample['disp']

        if np.random.rand() < self.h_flip_prob and self.do_flip_type == 'hf':  # h-flip
            img1 = img1[:, ::-1]
            img2 = img2[:, ::-1]
            disp = disp[:, ::-1] * -1.0

        if np.random.rand() < self.h_flip_prob and self.do_flip_type == 'h':  # h-flip for stereo
            tmp = img1[:, ::-1]
            img1 = img2[:, ::-1]
            img2 = tmp

        if np.random.rand() < self.v_flip_prob and self.do_flip_type == 'v':  # v-flip
            img1 = img1[::-1, :]
            img2 = img2[::-1, :]
            disp = disp[::-1, :]

        sample['left'] = img1
        sample['right'] = img2
        sample['disp'] = disp
        return sample
