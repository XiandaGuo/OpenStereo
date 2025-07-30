from .extractor import MultiBasicEncoder, Feature
from .geometry import Combined_Geo_Encoding_Volume
from .submodule import *
from .update import BasicMultiUpdateBlock
from .utils import Map


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(in_channels, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv2 = nn.Sequential(
            BasicConv(in_channels * 2, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv3 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=2, dilation=1),
            BasicConv(in_channels * 6, in_channels * 6, is_3d=True, bn=True, relu=True, kernel_size=3,
                      padding=1, stride=1, dilation=1))

        self.conv3_up = BasicConv(in_channels * 6, in_channels * 4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels * 4, in_channels * 2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels * 2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(
            BasicConv(in_channels * 8, in_channels * 4, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 4, in_channels * 4, is_3d=True, kernel_size=3, padding=1, stride=1), )

        self.agg_1 = nn.Sequential(
            BasicConv(in_channels * 4, in_channels * 2, is_3d=True, kernel_size=1, padding=0, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1),
            BasicConv(in_channels * 2, in_channels * 2, is_3d=True, kernel_size=3, padding=1, stride=1))

        self.feature_att_8 = FeatureAtt(in_channels * 2, 64)
        self.feature_att_16 = FeatureAtt(in_channels * 4, 192)
        self.feature_att_32 = FeatureAtt(in_channels * 6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels * 4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels * 2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv


class IGEVPPStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        context_dims = args.HIDDEN_DIMS

        self.cnet = MultiBasicEncoder(output_dim=[args.HIDDEN_DIMS, context_dims], norm_fn="batch",
                                      downsample=args.N_DOWNSAMPLE)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.HIDDEN_DIMS)
        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], args.HIDDEN_DIMS[i] * 3, 3, padding=3 // 2) for i in
             range(self.args.N_GRU_LAYERS)])
        self.feature = Feature()

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
        )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
        )
        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2x(64, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64), nn.ReLU()
        )

        self.spx_2_gru = Conv2x(64, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )

        self.conv = BasicConv(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)
        self.patch0 = nn.Conv3d(8, 8, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=False)
        self.patch1 = nn.Conv3d(8, 8, kernel_size=(4, 1, 1), stride=(4, 1, 1), bias=False)
        self.cost_agg0 = hourglass(8)
        self.cost_agg1 = hourglass(8)
        self.cost_agg2 = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)
        self.disp_conv = nn.Sequential(
            BasicConv(3, 64, kernel_size=1, stride=1, padding=0),
            BasicConv(64, 64, kernel_size=3, stride=1, padding=1),
        )
        self.selective_conv = nn.Sequential(
            BasicConv(96 + 64, 128, kernel_size=1, stride=1, padding=0),
            BasicConv(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 3, 3, 1, 1, bias=False),
        )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp * 4., spx_pred).unsqueeze(1)
        return up_disp

    def forward(self, data):
        image1 = data['left']
        image2 = data['right']
        """ Estimate disparity between pair of frames """
        test_mode = not self.training
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        features_left = self.feature(image1)
        features_right = self.feature(image2)
        stem_2x = self.stem_2(image1)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(image2)
        stem_4y = self.stem_4(stem_2y)
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))
        all_disp_volume = build_gwc_volume(match_left, match_right, self.args.MAX_DISP // 4, 8)

        disp_volume0 = all_disp_volume[:, :, :self.args.S_DISP_RANGE]
        disp_volume1 = self.patch0(all_disp_volume[:, :, :self.args.M_DISP_RANGE])
        disp_volume2 = self.patch1(all_disp_volume)

        geo_encoding_volume0 = self.cost_agg0(disp_volume0, features_left)
        geo_encoding_volume1 = self.cost_agg1(disp_volume1, features_left)
        geo_encoding_volume2 = self.cost_agg2(disp_volume2, features_left)

        cost_volume0 = self.classifier(geo_encoding_volume0)
        prob_volume0 = F.softmax(cost_volume0.squeeze(1), dim=1)
        agg_disp0 = disparity_regression(prob_volume0, self.args.S_DISP_RANGE, self.args.S_DISP_INTERVAL)

        cost_volume1 = self.classifier(geo_encoding_volume1)
        prob_volume1 = F.softmax(cost_volume1.squeeze(1), dim=1)
        agg_disp1 = disparity_regression(prob_volume1, self.args.M_DISP_RANGE, self.args.M_DISP_INTERVAL)

        cost_volume2 = self.classifier(geo_encoding_volume2)
        prob_volume2 = F.softmax(cost_volume2.squeeze(1), dim=1)
        agg_disp2 = disparity_regression(prob_volume2, self.args.L_DISP_RANGE, self.args.L_DISP_INTERVAL)

        disp_feature = self.disp_conv(torch.cat([agg_disp0, agg_disp1, agg_disp2], dim=1))
        selective_weights = torch.sigmoid(self.selective_conv(torch.cat([features_left[0], disp_feature], dim=1)))
        cnet_list = self.cnet(image1, num_layers=self.args.N_GRU_LAYERS)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in
                    zip(inp_list, self.context_zqr_convs)]

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(geo_encoding_volume0.float(), geo_encoding_volume1.float(), geo_encoding_volume2.float(),
                           match_left.float(), match_right.float(), radius=self.args.CORR_RADIUS)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp = agg_disp0
        iter_preds = []

        # GRUs iterations to update disparity
        iters = self.args.VALID_ITERS if test_mode else self.args.TRAIN_ITERS
        for itr in range(iters):
            disp = disp.detach()
            geo_feat0, geo_feat1, geo_feat2, init_corr = geo_fn(disp, coords)
            # with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat0, geo_feat1, geo_feat2,
                                                                  init_corr, selective_weights, disp,
                                                                  iter16=self.args.N_GRU_LAYERS == 3,
                                                                  iter08=self.args.N_GRU_LAYERS >= 2)

            disp = disp + delta_disp
            if test_mode and itr < iters - 1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            iter_preds.append(disp_up)

        if test_mode:
            return disp_up

        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        agg_disp0 = context_upsample(agg_disp0 * 4., spx_pred.float())
        agg_disp1 = context_upsample(agg_disp1 * 4., spx_pred.float())
        agg_disp2 = context_upsample(agg_disp2 * 4., spx_pred.float())
        agg_preds = [agg_disp0, agg_disp1, agg_disp2]

        return {'init_disp': agg_preds,
                'disp_preds': iter_preds,
                'disp_pred': iter_preds[-1]}

    def get_loss(self, model_pred, input_data):
        disp_gt = input_data["disp"]
        mask = (disp_gt < self.max_disp) & (disp_gt > 0)
        valid = mask.float()
        max_disp0 = 192
        max_disp1 = 384
        max_disp = 700

        disp_gt = disp_gt.unsqueeze(1)
        mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
        valid = ((valid >= 0.5) & (mag < self.max_disp)).unsqueeze(1)
        assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
        assert not torch.isinf(disp_gt[valid.bool()]).any()
        disp_loss = 0.0
        mag = torch.sum(disp_gt**2, dim=1).sqrt()
        mask0 = ((valid >= 0.5) & (mag < max_disp0)).unsqueeze(1)
        mask1 = ((valid >= 0.5) & (mag < max_disp1)).unsqueeze(1)
        mask = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)

        disp_init_pred = model_pred['init_disp']
        disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[0][mask0.bool()], disp_gt[mask0.bool()], reduction='mean')
        disp_loss += 0.5 * F.smooth_l1_loss(disp_init_pred[1][mask1.bool()], disp_gt[mask1.bool()], reduction='mean')
        disp_loss += 0.2 * F.smooth_l1_loss(disp_init_pred[2][mask.bool()], disp_gt[mask.bool()], reduction='mean')

        # gru loss
        loss_gamma = 0.9
        disp_preds = model_pred['disp_preds']
        n_predictions = len(disp_preds)
        assert n_predictions >= 1
        for i in range(n_predictions):
            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            i_loss = (disp_preds[i] - disp_gt).abs()
            assert i_loss.shape == mask.shape, [i_loss.shape, mask.shape, disp_gt.shape, disp_preds[i].shape]
            disp_loss += i_weight * i_loss[mask.bool()].mean()

        # epe = torch.sum((disp_preds[-1] - disp_gt) ** 2, dim=1).sqrt()
        # epe = epe.view(-1)[mask.view(-1)]
        #
        # metrics = {
        #     'epe': epe.mean().item(),
        #     '1px': (epe < 1).float().mean().item(),
        #     '3px': (epe < 3).float().mean().item(),
        #     '5px': (epe < 5).float().mean().item(),
        # }

        loss_info = {'scalar/train/loss_disp': disp_loss.item()}
        return disp_loss, loss_info
