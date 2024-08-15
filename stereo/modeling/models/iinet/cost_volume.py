import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

import sys
sys.path.append('.')

from .networks import MLP

import math


class MsCostVolumeManager(nn.Module):
    def __init__(self,
                 num_depth_bins=64,
                 multiscale=1,
                 disp_scale=2,
                 matching_dim_size=16,
                 dot_dim=1):
        super().__init__()
        self.num_depth_bins = [4, 6, num_depth_bins]
        self.disp_scale = disp_scale
        self.stride = [1, 2, 4, 8]
        self.num_cv = multiscale + 1
        self.num_depth_bins = self.num_depth_bins[-self.num_cv:]
        self.stride = self.stride[:self.num_cv][::-1]
        self.in_channels = 2 * matching_dim_size + 1
        self.feat_scale = np.sqrt(matching_dim_size)
        channel_list = [self.in_channels, 64, 32, 1]
        self.mlp = MLP(channel_list=channel_list)
        self.beta = - np.log(0.5)
        self.pt = 0.90
        self.alpha = 1 / (-self.pt * np.log(self.pt) - (1 - self.pt) * np.log(1 - self.pt) - self.beta)

    def forward(self, left_feats, right_feats):
        """ Runs the cost volume and gets the lowest cost result """

        num_stage = len(left_feats)

        # loop through depth planes
        cost_volume = [None] * num_stage
        confidence = [None] * num_stage

        coarse_disp = [None] * num_stage
        hypos = [None] * (num_stage - 1)
        
        for k in range(num_stage - 1, -1, -1):
            batch_size, num_feat_channels, matching_height, matching_width = left_feats[k].shape
            volume = left_feats[k].new_zeros([batch_size, self.num_depth_bins[k], matching_height, matching_width])

            if k == num_stage - 1:

                dot_feat = torch.sum(left_feats[k] * right_feats[k], dim=1, keepdim=True) / self.feat_scale
                mlp_feat = torch.cat((left_feats[k][:, :, :, :], right_feats[k][:, :, :, :], dot_feat), dim=1)
                volume[:, 0, :, :] = self.mlp(mlp_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).squeeze(1)
                for depth_id in range(1, self.num_depth_bins[k]):
                    mlp_feat = left_feats[0].new_zeros([batch_size, self.in_channels, matching_height, matching_width])
                    mlp_feat[:, :num_feat_channels] = left_feats[k]
                    mlp_feat[:, num_feat_channels:2*num_feat_channels, :, depth_id:] = right_feats[k][:, :, :, :-depth_id]
                    mlp_feat[:, -1:, :, depth_id:] = torch.sum(left_feats[k][:, :, :, depth_id:] * right_feats[k][:, :, :, :-depth_id], dim=1, keepdim=True) / self.feat_scale
                    volume[:, depth_id, :, :] = self.mlp(mlp_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).squeeze(1)

                cost_volume[k] = volume
                p = torch.sigmoid(cost_volume[k] * 2).clamp(min=1e-7, max=1 - 1e-7)
                q = 1 - p

                entropy = - p * torch.log(p) - q * torch.log(q)
                uncertainty = torch.mean(entropy, dim=1, keepdim=True)
                confidence[k] = torch.clamp(
                    self.alpha * (uncertainty - self.beta) * (torch.max(p, dim=1)[0].unsqueeze(1)), max=1.0)
                prob_volume = torch.softmax(volume, dim=1)
                _, ind = prob_volume.sort(dim=1, descending=True)
                coarse_disp[k] = ind[:, :1].float() * (confidence[k] > 0.1) / self.disp_scale
                if len(hypos) > 0:
                    hypos[k - 1] = F.interpolate(ind[:, :self.num_depth_bins[k - 1] // 2].float(), scale_factor=2)
            else:

                mlp_feat = left_feats[0].new_zeros(
                   [batch_size, self.in_channels, self.num_depth_bins[k] * matching_height, matching_width])

                prev_ind = hypos[k]  # b, d//2,h ,w

                hpos = torch.arange(0, matching_height, device=volume.device).view(1, 1, matching_height, 1).expand(
                    batch_size, self.num_depth_bins[k], matching_height, matching_width)
                wpos = torch.arange(0, matching_width, device=volume.device).view(1, 1, 1, matching_width)  # (H *W)
                dpos = torch.stack([prev_ind * 2, prev_ind * 2 + 1], dim=2).view(batch_size, -1, matching_height,
                                                                                 matching_width)
                hypos[k] = dpos

                tgt_wpos = wpos - dpos  # b,d,h,w
                coords_x = tgt_wpos / ((matching_width - 1) / 2) - 1
                coords_y = hpos / ((matching_height - 1) / 2) - 1
                grid = torch.stack([coords_x, coords_y], dim=4)
                tgt_feats = F.grid_sample(right_feats[k],
                                          grid.view(batch_size, self.num_depth_bins[k] * matching_height,
                                                    matching_width, 2),
                                          mode='nearest',
                                          padding_mode='zeros',
                                          align_corners=False)
                left_feat = left_feats[k].repeat(1, 1, self.num_depth_bins[k], 1)
                mlp_feat[:, :num_feat_channels] = left_feat
                mlp_feat[:, num_feat_channels:2*num_feat_channels] = tgt_feats
                mlp_feat[:, -1:] = torch.sum(left_feat * tgt_feats, dim=1, keepdim=True) / self.feat_scale
                volume = self.mlp(mlp_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).reshape(batch_size,
                                                                                            self.num_depth_bins[k],
                                                                                            matching_height,
                                                                                            matching_width)

                cost_volume[k] = volume

                p = torch.sigmoid(cost_volume[k] * 2).clamp(min=1e-7, max=1 - 1e-7)
                q = 1 - p
                entropy = - p * torch.log(p) - q * torch.log(q)
                uncertainty = torch.mean(entropy, dim=1, keepdim=True)
                confidence[k] = torch.clamp(
                    self.alpha * (uncertainty - self.beta) * (torch.max(p, dim=1)[0].unsqueeze(1)), max=1.0)
                max_ind = torch.max(volume, dim=1)[1].unsqueeze(1)
                # TODO:Ablation on filter
                coarse_disp[k] = torch.gather(dpos, 1, max_ind) * (confidence[k] > 0.1) / (self.disp_scale * 2 ** (num_stage - 1 - k))
                if k > 0:
                    _, ind = volume.sort(dim=1, descending=True)
                    hypos[k - 1] = torch.gather(dpos, 1, ind[:, :self.num_depth_bins[k - 1] // 2])
                    hypos[k - 1] = F.interpolate(hypos[k - 1].float(), scale_factor=2)

        return cost_volume, hypos, {'cdisp': coarse_disp, 'cconf': confidence}



