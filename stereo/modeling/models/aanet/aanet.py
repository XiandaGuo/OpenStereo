from .submodule import *


class aanet(nn.Module):
    def __init__(self, model_cfg):
        super(aanet, self).__init__()

        self.model_cfg = model_cfg
        self.max_disp = model_cfg.MAX_DISP
        self.refinement_type = model_cfg.REFINEMENT_TYPE
        self.num_downsample = model_cfg.NUM_DOWNSAMPLE
        self.aggregation_type = model_cfg.AGGREGATION_TYPE
        self.num_scales = model_cfg.NUM_SCALES
        self.no_feature_mdconv = model_cfg.NO_FEATURE_MDCONV
        self.num_fusions = model_cfg.NUM_FUSIONS
        self.deformable_groups = model_cfg.DEFORMABLE_GROUPS
        self.mdconv_dilation = model_cfg.MDCONV_DILATION
        self.no_intermediate_supervision = model_cfg.NO_INTERMEDIATE_SUPERVISION
        self.num_stage_blocks = model_cfg.NUM_STAGE_BLOCKS
        self.num_deform_blocks = model_cfg.NUM_DEFORM_BLOCKS

        # Feature extractor
        self.feature_extractor = AANetFeature(feature_mdconv=(not self.no_feature_mdconv))
        self.max_disp = self.max_disp // 3
        in_channels = [32 * 4, 32 * 8, 32 * 16, ]
        self.fpn = FeaturePyramidNetwork(in_channels=in_channels,
                                         out_channels=32 * 4)

        # Cost volume construction
        cost_volume_module = CostVolumePyramid
        self.cost_volume = cost_volume_module(self.max_disp, feature_similarity='correlation')

        # Cost aggregation
        self.aggregation = AdaptiveAggregation(max_disp=self.max_disp,
                                               num_scales=self.num_scales,
                                               num_fusions=self.num_fusions,
                                               num_stage_blocks=self.num_stage_blocks,
                                               num_deform_blocks=self.num_deform_blocks,
                                               mdconv_dilation=self.mdconv_dilation,
                                               deformable_groups=self.deformable_groups,
                                               intermediate_supervision=not self.no_intermediate_supervision)
        match_similarity = True
        # Disparity estimation
        self.disparity_estimation = DisparityEstimation(self.max_disp, match_similarity)

        # Refinement
        refine_module_list = nn.ModuleList()
        for i in range(self.num_downsample):
            refine_module_list.append(StereoDRNetRefinement())
        self.refinement = refine_module_list

    def feature_extraction(self, img):
        feature = self.feature_extractor(img)
        feature = self.fpn(feature)
        return feature

    def cost_volume_construction(self, left_feature, right_feature):
        cost_volume = self.cost_volume(left_feature, right_feature)
        if isinstance(cost_volume, list):
            if self.num_scales == 1:
                cost_volume = [cost_volume[0]]  # ablation purpose for 1 scale only
        else:
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

        return disparity_pyramid

    def forward(self, data):
        left_img = data['left']
        right_img = data['right']

        left_feature = self.feature_extraction(left_img)
        right_feature = self.feature_extraction(right_img)
        cost_volume = self.cost_volume_construction(left_feature, right_feature)
        aggregation = self.aggregation(cost_volume)
        disparity_pyramid = self.disparity_computation(aggregation)
        disparity_pyramid += self.disparity_refinement(left_img, right_img, disparity_pyramid[-1])

        return {'disp_pred': disparity_pyramid[-1],
                'disp_preds': disparity_pyramid}

    def get_loss(self, model_preds, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]
        disp_gt = disp_gt.unsqueeze(1)
        mask = (disp_gt < self.model_cfg.MAX_DISP) & (disp_gt > 0)  # [bz, 1, h, w]
        mask.detach_()

        weights = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]

        loss = 0.0
        for i, input_ in enumerate(model_preds['disp_preds']):
            input_ = input_.unsqueeze(1)
            input_ = F.interpolate(input_,
                                   size=(disp_gt.size(-2), disp_gt.size(-1)), mode='bilinear',
                                   align_corners=False) * (disp_gt.size(-1) / input_.size(-1))

            loss += weights[i] * F.smooth_l1_loss(input_[mask], disp_gt[mask], size_average=True)

        loss_info = {'scalar/train/loss_disp': loss.item()}
        return loss, loss_info
