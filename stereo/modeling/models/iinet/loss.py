from collections import OrderedDict

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import kornia
import numpy as np
from dataclasses import dataclass

import sys
sys.path.append('.')
from .metrics import cal_metric


class VolumeBiFocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pt, target, weight):
        loss = - weight * self.alpha * ((1 - pt) ** self.gamma) * (target * torch.log(pt)) - (1 - self.alpha) * (
                        pt ** self.gamma) * ((1 - target) * torch.log(1 - pt))
        return loss

class Criterion(nn.Module):
    """
    Compute loss and evaluation metrics
    """
    def __init__(self, max_disp: int = -1, loss_weight: dict = None, out_scales=4, disp_scale=16):
        super(Criterion, self).__init__()

        if loss_weight is None:
            loss_weight = {}

        self.num_scales = out_scales
        self.max_disp = max_disp
        self.depth_scale = disp_scale
        self.weights = loss_weight

        self.kernel_size = 5
        self.std = 2.0

        self.sml1_crit = nn.SmoothL1Loss()
        self.l1_crit = nn.L1Loss()
        self.focal_loss = VolumeBiFocalLoss(alpha=0.8, gamma=2)
        self.klloss = torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='none', log_target=False)

    @staticmethod
    def to_homogeneous(input_tensor: Tensor, dim: int = 0) -> Tensor:
        """
        Converts tensor to homogeneous coordinates by adding ones to the specified
        dimension
        """
        ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
        output_bkN = torch.cat([input_tensor, ones], dim=dim)
        return output_bkN

    @torch.no_grad()
    def gen_disp_pyr(self, input_tensor: torch.Tensor):
        """ Creates a downscale pyramid for the input tensor. """
        output = [None] * self.num_scales
        output[0] = input_tensor
        for i in range(1, self.num_scales):
            _,_,h,w = output[i - 1].shape
            down = F.interpolate(output[i-1], size=(h // 2, w // 2), mode='nearest')
            output[i] = down
        return output

    @torch.no_grad()
    def gen_validmask_pyr(self, disp_pyr: list):
        validmask_pyr = [None] * self.num_scales
        for i in range(self.num_scales):
            #TODO::SPRING SPECIAL
            #validmask_pyr[i] = torch.logical_and(disp_pyr[i] >= 0.0, disp_pyr[i] < self.max_disp)
            validmask_pyr[i] = torch.logical_and(disp_pyr[i] > 0.0, disp_pyr[i] < self.max_disp)
        return validmask_pyr

    def cal_normal_loss(self, inputs: dict, outputs: dict, valid_mask: Tensor):

        def cal_normal(disp_b1hw: Tensor, cam_grid:Tensor, kernel_size:int, std:float) -> Tensor:
            """
            First smoothes incoming depth maps with a gaussian blur, backprojects
            those depth points into world space (see BackprojectDepth), estimates
            the spatial gradient at those points, and finally uses normalized cross
            correlation to estimate a normal vector at each location.

            """
            disp_smooth_b1hw = kornia.filters.gaussian_blur2d(
                disp_b1hw,
               (kernel_size, kernel_size),
                (std, std),
            )
            cam_points_b3hw = cam_grid * disp_smooth_b1hw

            gradients_b32hw = kornia.filters.spatial_gradient(cam_points_b3hw)

            return F.normalize(
                torch.cross(
                    gradients_b32hw[:, :, 0],
                    gradients_b32hw[:, :, 1],
                    dim=1,
                ),
                dim=1,
            )

        disp_pred = outputs["disp_pred"]
        # disp_pred = outputs["disp_pred_s0"]
        disp_gt = (inputs['disp_pyr'][0] / self.depth_scale)

        normals_gt = cal_normal(disp_gt, inputs['pos'], self.kernel_size, self.std)
        normals_pred = cal_normal(disp_pred, inputs['pos'], self.kernel_size, self.std)

        normals_mask = torch.logical_and(
            normals_gt.isfinite().all(dim=1, keepdim=True),
            normals_pred.isfinite().all(dim=1, keepdim=True))
        final_mask = torch.logical_and(normals_mask, valid_mask)

        normals_pred = normals_pred.masked_fill(~normals_mask, 1.0)
        normals_gt = normals_gt.masked_fill(~normals_mask, 1.0)

        normals_dot_b1hw = (1.0 - (normals_pred * normals_gt).sum(dim=1)).unsqueeze(1)
        normals_loss = normals_dot_b1hw.masked_select(final_mask).mean()

        return normals_loss

    def cal_l1_loss(self, inputs: dict, outputs: dict, validmask_pyr: list) -> list:
            l1disp_loss = [0.0] * self.num_scales
            for i in range(len(validmask_pyr)):
                disp_gt = inputs['disp_pyr'][i] / self.depth_scale
                validmask = validmask_pyr[i]
                if i == 0:
                    disp_pred = outputs["disp_pred"]
                else:
                    disp_pred = outputs[f"disp_pred_s{i}"]
                l1disp_loss[i] = self.l1_crit(disp_gt[validmask], disp_pred[validmask])

            return l1disp_loss

    def cal_msgrad_loss(self, inputs: dict, outputs: dict, validmask_pyr: list) -> list:

        grad_loss = [0.0] * self.num_scales
        for i in range(len(validmask_pyr)):
            disp_gt = inputs['disp_pyr'][i] / self.depth_scale
            if i == 0:
                disp_pred = outputs["disp_pred"]
            else:
                disp_pred = outputs[f"disp_pred_s{i}"]
            disp_gt_grad = kornia.filters.spatial_gradient(disp_gt)
            disp_pred_grad = kornia.filters.spatial_gradient(disp_pred)
            validmask = torch.logical_and(disp_gt_grad.isfinite().all(dim=1, keepdim=True), validmask_pyr[i])
            validmask = torch.logical_and(disp_pred_grad.isfinite().all(dim=1, keepdim=True), validmask)
            grad_error = torch.abs(disp_pred_grad.masked_select(validmask) -
                                   disp_gt_grad.masked_select(validmask))
            grad_loss[i] = torch.mean(grad_error)

        return grad_loss


    def cal_focal_loss(self, input, output):
        def gen_gt_volume(disp_gt, disp_index, scale=1):
            invalid_mask = None
            if disp_index is None:
                b, d, h, w = disp_gt.shape[0], self.max_disp // scale, disp_gt.shape[2] // scale, disp_gt.shape[3] // scale
                disp_index = torch.arange(0, d, device=disp_gt.device)[None, :, None, None].expand(b, -1, h, w)
                w_pos = torch.arange(0, w, device=disp_gt.device)[None, None, None, :]
                invalid_mask = (w_pos < disp_index).unsqueeze(1).expand(-1, scale ** 2, -1, -1, -1)
            b,d,h,w = disp_index.shape #b,d,h,w
            unfold_size = scale
            unfolded_patch = torch.nn.functional.unfold(disp_gt, unfold_size, dilation=1,
                                                              padding=0, stride=unfold_size)
            unfolded_patch = unfolded_patch.reshape(b,unfold_size ** 2, 1, h, w) #b, s, 1, h, w
            distance = torch.abs(unfolded_patch / scale - disp_index.unsqueeze(1)) #b, s, d, h, w
            hist = 1 -distance
            distance_mask = distance > 1
            if invalid_mask is not None:
                hist[invalid_mask] = 0
            hist[distance_mask] = 0
            gt_volume = hist.sum(dim=1)
            gt_volume = gt_volume / gt_volume.sum(dim=1, keepdim=True)
            gt_volume[torch.isnan(gt_volume)] = 0
            mask_volume = (gt_volume > 0).float()

            return gt_volume, mask_volume

        disp_gt = input['disp_pyr'][0]
        klloss_weights = [5, 5, 10]
        bceloss_weights = [5, 5, 10]
        bceloss = 0.0
        klloss = 0.0

        cost_volume = output['cost_volume']
        ns = len(cost_volume)
        klloss_weights = klloss_weights[3-ns:]
        bceloss_weights = bceloss_weights[3-ns:]
        for i in range(len(cost_volume)):
            pred_volume = torch.softmax(cost_volume[i], dim=1).clamp(min=1e-7)
            sigmoid_volume = torch.sigmoid(cost_volume[i] * 2).clamp(min=1e-7, max=1-1e-7)
            if i < ns - 1:
                pred_index = output['hypos'][i]
                gt_volume, mask_volume = gen_gt_volume(disp_gt, pred_index,scale=2 ** (4 - ns + i))
            else:
                gt_volume, mask_volume = gen_gt_volume(disp_gt, None, scale=2 ** (4 - ns + i))

            covered_mask = (gt_volume.sum(dim=1, keepdim=True) > 0).expand(-1, gt_volume.shape[1], -1, -1)
            occ_hypos_count = mask_volume.sum(dim=1, keepdim=True).expand(-1, gt_volume.shape[1], -1, -1)
            edge_weight = occ_hypos_count

            est = torch.masked_select(pred_volume, covered_mask)
            gt = torch.masked_select(gt_volume, covered_mask) #b, d, h, w
            tmp_loss = self.klloss(est.log(), gt)
            edge_weight = torch.masked_select(edge_weight, covered_mask)
            # Apply edge weight
            tmp_loss = tmp_loss * edge_weight
            tmp_loss = klloss_weights[i]*tmp_loss
            klloss += tmp_loss.mean()

            tmp_loss = self.focal_loss(sigmoid_volume, mask_volume, gt_volume)
            tmp_loss = tmp_loss.masked_select(covered_mask)
            tmp_loss = bceloss_weights[i] * edge_weight * tmp_loss
            bceloss += tmp_loss.mean()

        return klloss + bceloss

    def aggregate_loss(self, loss_dict: dict):
        """
        compute weighted sum of loss

        :param loss_dict: dictionary of losses
        """
        loss = 0.0
        for key,value in loss_dict.items():
            if isinstance(value, list):
                for i in range(0, self.num_scales):
                    loss += loss_dict[key][i] * self.weights[key][i]
            else:
                loss += loss_dict[key] * self.weights[key]
        loss_dict['aggregated'] = loss
        return

    def forward(self, inputs: dict, outputs: dict, uncer_only=False):
        """
        :param inputs: input data
        :param outputs: output from the network, dictionary
        :return: loss dictionary
        """
        loss = {}

        with torch.no_grad():
            inputs['disp_pyr'] = self.gen_disp_pyr(inputs['disp_pyr'])
            validmask_pyr = self.gen_validmask_pyr(inputs['disp_pyr'])

        if not uncer_only:
            loss['l1'] = self.cal_l1_loss(inputs, outputs, validmask_pyr)
            loss['grad'] = self.cal_msgrad_loss(inputs, outputs, validmask_pyr)
            loss['normal'] = self.cal_normal_loss(inputs, outputs, validmask_pyr[0])
            loss['focal'] = torch.tensor(0., device=validmask_pyr[0].device)
        else:
            loss['grad'] = [torch.tensor(0., device=validmask_pyr[0].device)] * 4
            loss['l1'] = [torch.tensor(0., device=validmask_pyr[0].device)] * 4
            loss['normal'] = torch.tensor(0., device=validmask_pyr[0].device)
            loss['focal'] = self.cal_focal_loss(inputs, outputs)

        outputs['disp'] = inputs['disp_pyr'][0][0:1]
        outputs['displ'] = inputs['disp_pyr'][1][0:1]
        # TODO::SPRING SPECIAL
        #validmask = torch.logical_and(inputs['disp_pyr'][0] >= 0.0, inputs['disp_pyr'][0] < 192)
        validmask = torch.logical_and(inputs['disp_pyr'][0] > 0.0, inputs['disp_pyr'][0] < 192)
        outputs['validmask'] = validmask

        self.aggregate_loss(loss)

        if ((validmask_pyr[0].float().flatten(1).mean(dim=-1) / (inputs['disp_pyr'][0] > 0).float().flatten(
                1).mean(dim=-1)) < 0.1).any():
            loss['aggregated'] = loss['aggregated'] * 0.0

        # for benchmarking
        if not uncer_only:
            # cal_metric(outputs['disp_pred_s0'] * self.depth_scale, inputs['disp_pyr'][0], loss, validmask)
            cal_metric(outputs['disp_pred'] * self.depth_scale, inputs['disp_pyr'][0], loss, validmask)
        else:
            margin = torch.where(inputs['disp_pyr'][-3] % 2 < 1, inputs['disp_pyr'][-3] % 2,
                                                      inputs['disp_pyr'][-3] % 2 - 2)
            cal_metric(outputs['coarse_disp'] * 16, inputs['disp_pyr'][-3] - margin, loss, validmask_pyr[-3])

        return OrderedDict(loss)


def build_criterion(opts, loss_weight):

    return Criterion(opts.MAX_DISP, loss_weight, opts.OUT_SCALE, opts.DISP_SCALE)
