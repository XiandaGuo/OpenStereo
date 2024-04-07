# @Time    : 2024/4/2 11:48
# @Author  : zhangchenming
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fadnet_backbone import FadnetBackbone
from .fadnet_cost_processor import FADAggregator
from .fadnet_disp_predictor import DispNetRes
from .submodule import *


class FADNet(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.maxdisp = cfgs.MAX_DISP
        resBlock = cfgs.RESBLOCK
        input_channel = cfgs.INPUT_CHANNEL
        encoder_ratio = cfgs.ENCODER_RATIO
        decoder_ratio = cfgs.DECODER_RATIO
        in_planes = cfgs.IN_PLANES

        self.backbone = FadnetBackbone(resBlock=resBlock, maxdisp=self.maxdisp, input_channel=input_channel,
                                       encoder_ratio=encoder_ratio, decoder_ratio=decoder_ratio)
        self.cost_processor = FADAggregator(resBlock=resBlock, maxdisp=self.maxdisp, input_channel=input_channel,
                                            encoder_ratio=encoder_ratio, decoder_ratio=decoder_ratio)
        self.disp_predictor = DispNetRes(in_planes, resBlock=resBlock, input_channel=input_channel,
                                         encoder_ratio=encoder_ratio, decoder_ratio=decoder_ratio)

    def build_corr(self,img_left, img_right, max_disp=40, zero_volume=None):
        B, C, H, W = img_left.shape
        if zero_volume is not None:
            tmp_zero_volume = zero_volume #* 0.0
            #print('tmp_zero_volume: ', mean)
            volume = tmp_zero_volume
        else:
            volume = img_left.new_zeros([B, max_disp, H, W])
        for i in range(max_disp):
            if (i > 0) & (i < W):
                volume[:, i, :, i:] = (img_left[:, :, :, i:] * img_right[:, :, :, :W-i]).mean(dim=1)
            else:
                volume[:, i, :, :] = (img_left[:, :, :, :] * img_right[:, :, :, :]).mean(dim=1)

        volume = volume.contiguous()
        return volume

    def forward(self, inputs, enabled_tensorrt=False):
        # dispnetc_flows, dispnetres_flows, dispnetres_final_flow=self.fadnet(inputs)
        # parse batch
        ref_img = inputs["left"]
        tgt_img = inputs["right"]

        # extract image feature
        conv1_l, conv2_l, conv3a_l, conv3a_r = self.backbone(ref_img, tgt_img)

        out_corr = self.build_corr(conv3a_l, conv3a_r, max_disp=self.maxdisp // 8 + 16)

        # generate first-stage flows

        dispnetc_flows = self.cost_processor(ref_img, tgt_img, conv1_l, conv2_l, conv3a_l, out_corr)
        dispnetc_final_flow = dispnetc_flows[0]

        # warp img1 to img0; magnitude of diff between img0 and warped_img1,
        resampled_img1 = warp_right_to_left(tgt_img, -dispnetc_final_flow)
        diff_img0 = ref_img - resampled_img1
        norm_diff_img0 = channel_length(diff_img0)

        inputs_net2 = torch.cat((ref_img, tgt_img, resampled_img1, dispnetc_final_flow, norm_diff_img0), dim=1)

        if enabled_tensorrt:
            dispnetres_flows = self.disp_predictor(inputs_net2, dispnetc_final_flow)
        else:
            dispnetres_flows = self.disp_predictor(inputs_net2, dispnetc_flows)

        index = 0
        dispnetres_final_flow = dispnetres_flows[index]

        if self.training:
            output = {
                "training_disp": {
                    "dispnetc_flows": {
                        "disp_ests": dispnetc_flows,
                        "disp_gt": inputs["disp_gt"],
                        "mask": inputs["mask"]
                    },
                    "dispnetres_flows": {
                        "disp_ests": dispnetres_flows,
                        "disp_gt": inputs["disp_gt"],
                        "mask": inputs["mask"]
                    }

                },
                "visual_summary": {}
            }
        else:
            if dispnetres_final_flow.dim() == 4:
                dispnetres_final_flow = dispnetres_final_flow.squeeze(1)

            if 'disp_gt' in inputs:
                output = {
                    "inference_disp": {
                        "disp_est": dispnetres_final_flow,
                        "disp_gt": inputs["disp_gt"],
                        "mask": inputs["mask"]

                    },
                    "visual_summary": {}
                }
            else:
                output = {
                    "inference_disp": {
                        "disp_est": dispnetres_final_flow
                    },
                    "visual_summary": {}
                }

        return {'disp_pred': dispnetres_final_flow}
