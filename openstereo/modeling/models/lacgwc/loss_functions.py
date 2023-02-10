import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def disp2distribute(disp_gt, max_disp, b=2):
    disp_gt = disp_gt.unsqueeze(1)
    disp_range = torch.arange(0, max_disp).view(1, -1, 1, 1).float().cuda()
    gt_distribute = torch.exp(-torch.abs(disp_range - disp_gt) / b)
    gt_distribute = gt_distribute / (torch.sum(gt_distribute, dim=1, keepdim=True) + 1e-8)
    return gt_distribute


def CEloss(disp_gt, max_disp, gt_distribute, pred_distribute):
    mask = (disp_gt > 0) & (disp_gt < max_disp)

    pred_distribute = torch.log(pred_distribute + 1e-8)

    ce_loss = torch.sum(-gt_distribute * pred_distribute, dim=1)
    ce_loss = torch.mean(ce_loss[mask])
    return ce_loss


class DispAffinity(nn.Module):
    def __init__(self, win_w, win_h, dilation, max_disp):
        super(DispAffinity, self).__init__()
        self.win_w = win_w
        self.win_h = win_h
        self.dilation = dilation
        self.max_disp = max_disp

    def forward(self, disp):
        B, _, H, W = disp.size()
        disp_mask = (disp > 0) & (disp < self.max_disp)

        affinity = []
        valid_mask = []
        shift = []

        for d in self.dilation:
            pad_t = (self.win_w // 2 * d, self.win_w // 2 * d, self.win_h // 2 * d, self.win_h // 2 * d)
            pad_disp = F.pad(disp, pad_t, mode='constant')

            for i in range(self.win_w):
                for j in range(self.win_h):
                    if (i == self.win_w // 2) & (j == self.win_h // 2):
                        continue

                    if ((j-self.win_h//2)*d, (i-self.win_w//2)*d) in shift:
                        continue
                    else:
                        rel_dif = torch.abs(pad_disp[:, :, d * j: d * j + H, d * i: d * i + W] - disp)

                        # whether the neighbor is valid
                        pad_mask = (pad_disp[:, :, d*j: d*j+H, d*i: d*i+W] > 0) & \
                                   (pad_disp[:, :, d*j: d*j+H, d*i: d*i+W] < self.max_disp)

                        # both are valid, the disparity distance is valid
                        mask = disp_mask & pad_mask
                        rel_dif = rel_dif * mask.float()

                        affinity.append(rel_dif)
                        valid_mask.append(mask)

                        shift.append(((j-self.win_h//2)*d, (i-self.win_w//2)*d))

        affinity = torch.stack(affinity, dim=1)
        valid_mask = torch.stack(valid_mask, dim=1)

        return affinity, valid_mask


def random_noise(img, type):
    if type == 'illumination':
        yuv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YUV)
        # yuv_rimg[:, :, 0] = yuv_rimg[:, :, 0] - 20
        illu_mask = yuv_img[:, :, 0] > 50
        yuv_img[:, :, 0][illu_mask] = yuv_img[:, :, 0][illu_mask] - 50
        img = Image.fromarray(cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB))
    elif type == 'color':
        rgb_img = np.array(img)
        color_mask = rgb_img[:, :, 2] > 50
        rgb_img[:, :, 2][color_mask] = rgb_img[:, :, 2][color_mask] - 50
        color_mask = rgb_img[:, :, 0] < 195
        rgb_img[:, :, 0][color_mask] = rgb_img[:, :, 0][color_mask] + 50
        img = Image.fromarray(rgb_img)
    elif type == 'noise':
        rgb_img = np.array(img)
        shape = rgb_img.shape
        noise = np.random.randint(-20, 20, size=shape).astype('uint8')
        rgb_img = rgb_img + noise
        rgb_img[rgb_img > 255] = 255
        rgb_img[rgb_img < 0] = 0
        img = Image.fromarray(rgb_img)
    elif type == 'haze':
        rgb_img = np.array(img)
        A = np.random.uniform(0.6, 0.95) * 255
        t = np.random.uniform(0.3, 0.95) * 255
        img = rgb_img * t + A * (1 - t)
        img = Image.fromarray(img.astype('uint8'))

    return img


def gradient_x(img):
    img = F.pad(img, [0, 1, 0, 0], mode='replicate')
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gx


def gradient_y(img):
    img = F.pad(img, [0, 0, 0, 1], mode='replicate')
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gy


def smooth_loss(img, disp):
    img_gx = gradient_x(img)
    img_gy = gradient_y(img)
    disp_gx = gradient_x(gradient_x(disp))
    disp_gy = gradient_y(gradient_y(disp))

    weight_x = torch.exp(-torch.mean(torch.abs(img_gx), dim=1, keepdim=True))
    weight_y = torch.exp(-torch.mean(torch.abs(img_gy), dim=1, keepdim=True))
    smoothness_x = torch.abs(disp_gx * weight_x)
    smoothness_y = torch.abs(disp_gy * weight_y)
    loss = smoothness_x + smoothness_y

    return torch.mean(loss)
