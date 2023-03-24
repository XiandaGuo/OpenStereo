import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from modeling.base_model import BaseModel
from base_trainer import BaseTrainer
import torch.optim as optim

from .feature import (StereoNetFeature, PSMNetFeature, GANetFeature, GCNetFeature,
                          FeaturePyrmaid, FeaturePyramidNetwork)
from .resnet import AANetFeature
from .cost import CostVolume, CostVolumePyramid
from .aggregation import (StereoNetAggregation, GCNetAggregation, PSMNetBasicAggregation,
                              PSMNetHGAggregation, AdaptiveAggregation)
from .estimation import DisparityEstimation
from .refinement import StereoNetRefinement, StereoDRNetRefinement, HourglassRefinement
from utils import get_attr_from,get_valid_args
from modeling.common.lamb import Lamb

class AANet(BaseModel):
    def __init__(self, *args, **kwargs):
        super(AANet, self).__init__(*args, **kwargs)
        self.Trainer = AANet_Trainer

    

    def build_network(self):
        model_cfg=self.model_cfg
        
        self.max_disp=model_cfg['base_config']['max_disp']
        self.refinement_type = model_cfg['base_config']['refinement_type']
        self.feature_type = model_cfg['base_config']['feature_type']
        self.feature_pyramid = model_cfg['base_config']['feature_pyramid']
        self.feature_pyramid_network = model_cfg['base_config']['feature_pyramid_network']
        self.num_downsample = model_cfg['base_config']['num_downsample']
        self.aggregation_type = model_cfg['base_config']['aggregation_type']
        self.num_scales = model_cfg['base_config']['num_scales']
        self.no_feature_mdconv=model_cfg['base_config']['no_feature_mdconv']
        self.feature_similarity=model_cfg['base_config']['feature_similarity']

        self.num_fusions=model_cfg['base_config']['num_fusions']
        self.deformable_groups=model_cfg['base_config']['deformable_groups']
        self.mdconv_dilation=model_cfg['base_config']['mdconv_dilation']
        self.no_intermediate_supervision=model_cfg['base_config']['no_intermediate_supervision']
        self.num_stage_blocks=model_cfg['base_config']['num_stage_blocks']
        self.num_deform_blocks=model_cfg['base_config']['num_deform_blocks']

        



        # Feature extractor
        if self.feature_type == 'stereonet':
            self.max_disp = self.max_disp // (2 ** self.num_downsample)
            self.num_downsample = self.num_downsample
            self.feature_extractor = StereoNetFeature(self.num_downsample)
        elif self.feature_type == 'psmnet':
            self.feature_extractor = PSMNetFeature()
            self.max_disp = self.max_disp // (2 ** self.num_downsample)
        elif self.feature_type == 'gcnet':
            self.feature_extractor = GCNetFeature()
            self.max_disp = self.max_disp // 2
        elif self.feature_type == 'ganet':
            self.feature_extractor = GANetFeature(feature_mdconv=(not self.no_feature_mdconv))
            self.max_disp = self.max_disp // 3
        elif self.feature_type == 'aanet':
            self.feature_extractor = AANetFeature(feature_mdconv=(not self.no_feature_mdconv))
            self.max_disp = self.max_disp // 3
        else:
            raise NotImplementedError

        if self.feature_pyramid_network:
            if self.feature_type == 'aanet':
                in_channels = [32 * 4, 32 * 8, 32 * 16, ]
            else:
                in_channels = [32, 64, 128]
            self.fpn = FeaturePyramidNetwork(in_channels=in_channels,
                                             out_channels=32 * 4)
        elif self.feature_pyramid:
            self.fpn = FeaturePyrmaid()

        # Cost volume construction
        if self.feature_type == 'aanet' or self.feature_pyramid or self.feature_pyramid_network:
            cost_volume_module = CostVolumePyramid
        else:
            cost_volume_module = CostVolume
        self.cost_volume = cost_volume_module(self.max_disp,
                                              feature_similarity=self.feature_similarity)

        # Cost aggregation
        max_disp = self.max_disp
        if self.feature_similarity == 'concat':
            in_channels = 64
        else:
            in_channels = 32  # StereoNet uses feature difference

        if self.aggregation_type == 'adaptive':
            self.aggregation = AdaptiveAggregation(max_disp=self.max_disp,
                                                   num_scales=self.num_scales,
                                                   num_fusions=self.num_fusions,
                                                   num_stage_blocks=self.num_stage_blocks,
                                                   num_deform_blocks=self.num_deform_blocks,
                                                   mdconv_dilation=self.mdconv_dilation,
                                                   deformable_groups=self.deformable_groups,
                                                   intermediate_supervision=not self.no_intermediate_supervision)
        elif self.aggregation_type == 'psmnet_basic':
            self.aggregation = PSMNetBasicAggregation(max_disp=self.max_disp)
        elif self.aggregation_type == 'psmnet_hourglass':
            self.aggregation = PSMNetHGAggregation(max_disp=self.max_disp)
        elif self.aggregation_type == 'gcnet':
            self.aggregation = GCNetAggregation()
        elif self.aggregation_type == 'stereonet':
            self.aggregation = StereoNetAggregation(in_channels=in_channels)
        else:
            raise NotImplementedError

        match_similarity = False if self.feature_similarity in ['difference', 'concat'] else True

        if 'psmnet' in self.aggregation_type:
            max_disp = self.max_disp * 4  # PSMNet directly upsamples cost volume
            match_similarity = True  # PSMNet learns similarity for concatenation

        # Disparity estimation
        self.disparity_estimation = DisparityEstimation(max_disp, match_similarity)

        # Refinement
        if self.refinement_type is not None and self.refinement_type != 'None':
            if self.refinement_type in ['stereonet', 'stereodrnet', 'hourglass']:
                refine_module_list = nn.ModuleList()
                for i in range(self.num_downsample):
                    if self.refinement_type == 'stereonet':
                        refine_module_list.append(StereoNetRefinement())
                    elif self.refinement_type == 'stereodrnet':
                        refine_module_list.append(StereoDRNetRefinement())
                    elif self.refinement_type == 'hourglass':
                        refine_module_list.append(HourglassRefinement())
                    else:
                        raise NotImplementedError

                self.refinement = refine_module_list
            else:
                raise NotImplementedError

    def feature_extraction(self, img):
        feature = self.feature_extractor(img)
        if self.feature_pyramid_network or self.feature_pyramid:
            feature = self.fpn(feature)
        return feature

    def cost_volume_construction(self, left_feature, right_feature):
        cost_volume = self.cost_volume(left_feature, right_feature)

        if isinstance(cost_volume, list):
            if self.num_scales == 1:
                cost_volume = [cost_volume[0]]  # ablation purpose for 1 scale only
        elif self.aggregation_type == 'adaptive':
            cost_volume = [cost_volume]
        return cost_volume

    def disparity_computation(self, aggregation):
        if isinstance(aggregation, list):
            disparity_pyramid = []
            length = len(aggregation)  # D/3, D/6, D/12
            for i in range(length):
                disp = self.disparity_estimation(aggregation[length - 1 - i])  # reverse
                disparity_pyramid.append(disp)  # D/12, D/6, D/3
        else:
            disparity = self.disparity_estimation(aggregation)
            disparity_pyramid = [disparity]

        return disparity_pyramid

    def disparity_refinement(self, left_img, right_img, disparity):
        disparity_pyramid = []
        if self.refinement_type is not None and self.refinement_type != 'None':
            if self.refinement_type == 'stereonet':
                for i in range(self.num_downsample):
                    # Hierarchical refinement
                    scale_factor = 1. / pow(2, self.num_downsample - i - 1)
                    if scale_factor == 1.0:
                        curr_left_img = left_img
                        curr_right_img = right_img
                    else:
                        curr_left_img = F.interpolate(left_img,
                                                      scale_factor=scale_factor,
                                                      mode='bilinear', align_corners=False)
                        curr_right_img = F.interpolate(right_img,
                                                       scale_factor=scale_factor,
                                                       mode='bilinear', align_corners=False)
                    inputs = (disparity, curr_left_img, curr_right_img)
                    disparity = self.refinement[i](*inputs)
                    disparity_pyramid.append(disparity)  # [H/2, H]

            elif self.refinement_type in ['stereodrnet', 'hourglass']:
                for i in range(self.num_downsample):
                    scale_factor = 1. / pow(2, self.num_downsample - i - 1)

                    if scale_factor == 1.0:
                        curr_left_img = left_img
                        curr_right_img = right_img
                    else:
                        curr_left_img = F.interpolate(left_img,
                                                      scale_factor=scale_factor,
                                                      mode='bilinear', align_corners=False)
                        curr_right_img = F.interpolate(right_img,
                                                       scale_factor=scale_factor,
                                                       mode='bilinear', align_corners=False)
                    inputs = (disparity, curr_left_img, curr_right_img)
                    disparity = self.refinement[i](*inputs)
                    disparity_pyramid.append(disparity)  # [H/2, H]

            else:
                raise NotImplementedError

        return disparity_pyramid

    def forward(self, inputs):
        # split left image and right image
        #imgs = torch.chunk(inputs, 2, dim = 1)
        left_img = inputs['ref_img']
        right_img = inputs['tgt_img']

        left_feature = self.feature_extraction(left_img)
        right_feature = self.feature_extraction(right_img)
        cost_volume = self.cost_volume_construction(left_feature, right_feature)
        aggregation = self.aggregation(cost_volume)
        disparity_pyramid = self.disparity_computation(aggregation)
        disparity_pyramid += self.disparity_refinement(left_img, right_img,
                                                       disparity_pyramid[-1])


        if self.training:
            output = {
                "training_disp": {
                    "disparity_pyramid": {
                        "disp_ests": disparity_pyramid,
                        "disp_gt": inputs["disp_gt"],
                        "mask": inputs["mask"]
                    },
                },
                "visual_summary": {}
            }
        else:
            if  disparity_pyramid[-1].dim()==4:
                disparity_pyramid[-1]=disparity_pyramid[-1].squeeze(1)
            output = {
                "inference_disp": {   
                    "disp_est": disparity_pyramid[-1],
                    "disp_gt": inputs["disp_gt"],
                    "mask": inputs["mask"]            
                },
                "visual_summary": {}
            }
        return output
    

class AANet_Trainer(BaseTrainer):
    def __init__(
            self,
            model: nn.Module = None,
            trainer_cfg: dict = None,
            data_cfg: dict = None,
            is_dist: bool = True,
            rank: int = None,
            device: torch.device = torch.device('cpu'),
            **kwargs
    ):
        super(AANet_Trainer, self).__init__(model,trainer_cfg,data_cfg,
            is_dist,rank,device, **kwargs)


    def build_optimizer(self, optimizer_cfg):
        if optimizer_cfg['solver']=='lamb':
            self.msg_mgr.log_info(optimizer_cfg)
            optimizer = Lamb(filter(lambda p: p.requires_grad, self.model.parameters()), lr=optimizer_cfg['lr'])
        else:
            self.msg_mgr.log_info(optimizer_cfg)
            optimizer = get_attr_from([optim], optimizer_cfg['solver'])
            valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
            optimizer = optimizer(params=[p for p in self.model.parameters() if p.requires_grad], **valid_arg)
        self.optimizer = optimizer
    
    
    