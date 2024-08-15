import logging
import os.path

import timm
import torch
from torch import nn
import numpy as np

import sys
sys.path.append('.')
from .cost_volume import MsCostVolumeManager
from .networks import (CVEncoder, ResnetMatchingEncoder,UnetMatchingEncoder, DepthDecoderMSR)
from .loss import build_criterion

from easydict import EasyDict
from stereo.utils import common_utils

class IINet(nn.Module):

    def __init__(self, opts):

        super().__init__()

        self.run_opts = opts

        self.num_ch_enc = [16, 24, 40, 112, 160]


        # iniitalize the first half of the U-Net, encoding the cost volume
        # and image prior image feautres
        if self.run_opts.CV_ENCODER_TYPE == "multi_scale_encoder":

            self.cost_volume_net = CVEncoder(
                num_ch_cv=self.run_opts.MAX_DISP // 2 ** (self.run_opts.MATCHING_SCALE + 1),
                num_ch_encs=self.num_ch_enc,
                num_ch_outs=[24, 64, 128, 192, 256],
                lrcv_level=self.run_opts.MATCHING_SCALE,
                multi_scale=self.run_opts.MULTISCALE
            )

            dec_num_input_ch = (self.num_ch_enc[:self.run_opts.MATCHING_SCALE - self.run_opts.MULTISCALE]
                                + self.cost_volume_net.num_ch_enc)
        else:
            raise ValueError("Unrecognized option for cost volume encoder type!")

        # iniitalize the final depth decoder
        if self.run_opts.DEPTH_DECODER_NAME == "unet_pp":
            self.depth_decoder = DepthDecoderMSR(dec_num_input_ch,
                                               scales=self.run_opts.OUT_SCALE,
                                               lrcv_scale=self.run_opts.MATCHING_SCALE + 1,
                                               multiscale=self.run_opts.MULTISCALE)
        else:
            raise ValueError("Unrecognized option for depth decoder name!")

        if self.run_opts.FEATURE_VOLUME_TYPE == "ms_cost_volume":
            cost_volume_class = MsCostVolumeManager
        else:
            raise ValueError("Unrecognized option for feature volume type {}!".format(self.run_opts.FEATURE_VOLUME_TYPE))

        scale_factor = 2 ** (self.run_opts.MATCHING_SCALE + 1)
        self.cost_volume = cost_volume_class(
            num_depth_bins=self.run_opts.MAX_DISP // scale_factor,
            dot_dim=self.run_opts.DOT_DIM,
            disp_scale=self.run_opts.DISP_SCALE // scale_factor,
            multiscale=self.run_opts.MULTISCALE
        )

        # init the matching encoder. resnet is fast and is the default for
        # results in the paper, fpn is more accurate but much slower.
        if "resnet" == self.run_opts.MATCHING_ENCODER_TYPE:
            #prefp = opts.pre_weight_name
            prefp = None
            self.matching_model = ResnetMatchingEncoder(18,
                                                        self.run_opts.MATCHING_FEATURE_DIMS,
                                                        self.run_opts.MATCHING_SCALE,
                                                        self.run_opts.MULTISCALE,
                                                        pretrainedfp=prefp)
        elif "unet" == self.run_opts.MATCHING_ENCODER_TYPE:
            #prefp = opts.pre_weight_name
            prefp = None
            self.matching_model = UnetMatchingEncoder(self.run_opts.MATCHING_FEATURE_DIMS,
                                                      self.run_opts.MATCHING_SCALE,
                                                      self.run_opts.MULTISCALE,
                                                      pretrainedfp=prefp)
        else:
            raise ValueError("Unrecognized option for matching encoder type!")

    def forward(self, input:dict, only_uncer=False):

        left_image, right_image = input['left'], input['right']
        assert left_image.shape[2]%32 == 0 and right_image.shape[2]%32 == 0, "Image size must be divisible by 32!"

        matching_left_feats, left_feats = self.matching_model(left_image)
        matching_right_feats, _ = self.matching_model(right_image)

        cost_volumes, hypos, priority = self.cost_volume(
                                                left_feats=matching_left_feats,
                                               right_feats=matching_right_feats,
                                           )

        if not only_uncer:
            filter_volumes = [cost_volume * confidence for cost_volume, confidence in zip(cost_volumes, priority['cconf'])]
            # Encode the cost volume and current image features
            cost_volume_features = self.cost_volume_net(
                                    filter_volumes,
                                    left_feats[self.run_opts.MATCHING_SCALE - self.run_opts.MULTISCALE:],
                                )
            left_feats = left_feats[:self.run_opts.MATCHING_SCALE - self.run_opts.MULTISCALE] + cost_volume_features

            # Decode into depth at multiple resolutions.
            depth_outputs = self.depth_decoder(left_feats, priority)
        else:
            depth_outputs = {}

        depth_outputs["coarse_disp"] = priority['cdisp'][0]
        depth_outputs["cost_volume"] = cost_volumes
        depth_outputs["hypos"] = hypos
        depth_outputs["confidence"] = priority['cconf'][0][0:1]

        return depth_outputs

    def get_loss(self, config, input, output):
        criterion = build_criterion(config.MODEL, config.TRAINER.LOSS_WEIGHT)
        losses = criterion(input,  output, config.TRAINER.UNCER_ONLY)
        loss_info = {'scalar/train/loss_focal': losses['focal'].item(),
                     'scalar/train/normal': losses['normal'].item(),
                     'scalar/train/aggregated': losses['aggregated'].item()}
        for i in range(len(losses['l1'])):
            loss_info[f'scalar/train/l1_{i}'] = losses['l1'][i].item()
            loss_info[f'scalar/train/grad_{i}'] = losses['grad'][i].item()
        return losses['aggregated'], loss_info




