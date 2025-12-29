import torch
import torch.nn as nn
import torch.nn.functional as F
from .update import BasicMultiUpdateBlock, BasicMultiUpdateBlock_mix2
from .geometry import Combined_Geo_Encoding_Volume
from .submodule import *
from .refinement import REMP
from .warp import disp_warp
import matplotlib.pyplot as plt

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass
from .depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2_decoder

    
def compute_scale_shift(monocular_depth, gt_depth, mask=None):
    """
    计算 monocular depth 和 ground truth depth 之间的 scale 和 shift.
    
    参数:
    monocular_depth (torch.Tensor): 单目深度图，形状为 (H, W) 或 (N, H, W)
    gt_depth (torch.Tensor): ground truth 深度图，形状为 (H, W) 或 (N, H, W)
    mask (torch.Tensor, optional): 有效区域的掩码，形状为 (H, W) 或 (N, H, W)
    
    返回:
    scale (float): 计算得到的 scale
    shift (float): 计算得到的 shift
    """
    
    flattened_depth_maps = monocular_depth.clone().view(-1).contiguous()
    sorted_depth_maps, _ = torch.sort(flattened_depth_maps)
    percentile_10_index = int(0.2 * len(sorted_depth_maps))
    threshold_10_percent = sorted_depth_maps[percentile_10_index]

    if mask is None:
        mask = (gt_depth > 0) & (monocular_depth > 1e-2) & (monocular_depth > threshold_10_percent)
    
    monocular_depth_flat = monocular_depth[mask]
    gt_depth_flat = gt_depth[mask]
    
    X = torch.stack([monocular_depth_flat, torch.ones_like(monocular_depth_flat)], dim=1)
    y = gt_depth_flat
    
    # 使用最小二乘法计算 [scale, shift]
    A = torch.matmul(X.t(), X) + 1e-6 * torch.eye(2, device=X.device)
    b = torch.matmul(X.t(), y)
    params = torch.linalg.solve(A, b)
    
    scale, shift = params[0].item(), params[1].item()
    
    return scale, shift


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))



        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

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

class Feat_transfer_cnet(nn.Module):
    def __init__(self, dim_list, output_dim):
        super(Feat_transfer_cnet, self).__init__()

        self.res_16x = nn.Conv2d(dim_list[0]+192, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0]+96, output_dim, kernel_size=3, padding=1, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0]+48, output_dim, kernel_size=3, padding=1, stride=1)

    def forward(self, features, stem_x_list):
        features_list = []
        feat_16x = self.res_16x(torch.cat((features[2], stem_x_list[0]), 1))
        feat_8x = self.res_8x(torch.cat((features[1], stem_x_list[1]), 1))
        feat_4x = self.res_4x(torch.cat((features[0], stem_x_list[2]), 1))
        features_list.append([feat_4x, feat_4x])
        features_list.append([feat_8x, feat_8x])
        features_list.append([feat_16x, feat_16x])
        return features_list



class Feat_transfer(nn.Module):
    def __init__(self, dim_list):
        super(Feat_transfer, self).__init__()
        self.conv4x = nn.Sequential(
            nn.Conv2d(in_channels=int(48+dim_list[0]), out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(48), nn.ReLU()
            )
        self.conv8x = nn.Sequential(
            nn.Conv2d(in_channels=int(64+dim_list[0]), out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(64), nn.ReLU()
            )
        self.conv16x = nn.Sequential(
            nn.Conv2d(in_channels=int(192+dim_list[0]), out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(192), nn.ReLU()
            )
        self.conv32x = nn.Sequential(
            nn.Conv2d(in_channels=dim_list[0], out_channels=160, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(160), nn.ReLU()
            )
        self.conv_up_32x = nn.ConvTranspose2d(160,
                                192,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_16x = nn.ConvTranspose2d(192,
                                64,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_8x = nn.ConvTranspose2d(64,
                                48,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        
        self.res_16x = nn.Conv2d(dim_list[0], 192, kernel_size=1, padding=0, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0], 64, kernel_size=1, padding=0, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0], 48, kernel_size=1, padding=0, stride=1)




    def forward(self, features):
        features_mono_list = []
        feat_32x = self.conv32x(features[3])
        feat_32x_up = self.conv_up_32x(feat_32x)
        feat_16x = self.conv16x(torch.cat((features[2], feat_32x_up), 1)) + self.res_16x(features[2])
        feat_16x_up = self.conv_up_16x(feat_16x)
        feat_8x = self.conv8x(torch.cat((features[1], feat_16x_up), 1)) + self.res_8x(features[1])
        feat_8x_up = self.conv_up_8x(feat_8x)
        feat_4x = self.conv4x(torch.cat((features[0], feat_8x_up), 1)) + self.res_4x(features[0])
        features_mono_list.append(feat_4x)
        features_mono_list.append(feat_8x)
        features_mono_list.append(feat_16x)
        features_mono_list.append(feat_32x)
        return features_mono_list





class MonSter(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.max_disp = args.max_disp
        context_dims = args.hidden_dims

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        mono_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        dim_list_ = mono_model_configs[self.args.encoder]['features']
        dim_list = []
        dim_list.append(dim_list_)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.feat_transfer = Feat_transfer(dim_list)
        self.feat_transfer_cnet = Feat_transfer_cnet(dim_list, output_dim=args.hidden_dims[0])


        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.stem_8 = nn.Sequential(
            BasicConv_IN(48, 96, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 96, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(96), nn.ReLU()
            )

        self.stem_16 = nn.Sequential(
            BasicConv_IN(96, 192, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(192, 192, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(192), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

        depth_anything = DepthAnythingV2(**mono_model_configs[args.encoder])
        depth_anything_decoder = DepthAnythingV2_decoder(**mono_model_configs[args.encoder])
        state_dict_dpt = torch.load(args.depth_anything_path, map_location='cpu')
        # state_dict_dpt = torch.load(f'/home/cjd/cvpr2025/fusion/Depth-Anything-V2-list3/depth_anything_v2_{args.encoder}.pth', map_location='cpu')
        depth_anything.load_state_dict(state_dict_dpt, strict=True)
        depth_anything_decoder.load_state_dict(state_dict_dpt, strict=False)
        self.mono_encoder = depth_anything.pretrained
        self.mono_decoder = depth_anything.depth_head
        self.feat_decoder = depth_anything_decoder.depth_head
        self.mono_encoder.requires_grad_(False)
        self.mono_decoder.requires_grad_(False)

        del depth_anything, state_dict_dpt, depth_anything_decoder
        self.REMP = REMP()


        self.update_block_mix_stereo = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims)
        self.update_block_mix_mono = BasicMultiUpdateBlock_mix2(self.args, hidden_dims=args.hidden_dims)


        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def infer_mono(self, image1, image2):
        height_ori, width_ori = image1.shape[2:]
        resize_image1 = F.interpolate(image1, scale_factor=14 / 16, mode='bilinear', align_corners=True)
        resize_image2 = F.interpolate(image2, scale_factor=14 / 16, mode='bilinear', align_corners=True)

        patch_h, patch_w = resize_image1.shape[-2] // 14, resize_image1.shape[-1] // 14
        features_left_encoder = self.mono_encoder.get_intermediate_layers(resize_image1, self.intermediate_layer_idx[self.args.encoder], return_class_token=True)
        features_right_encoder = self.mono_encoder.get_intermediate_layers(resize_image2, self.intermediate_layer_idx[self.args.encoder], return_class_token=True)
        depth_mono = self.mono_decoder(features_left_encoder, patch_h, patch_w)
        depth_mono = F.relu(depth_mono)
        depth_mono = F.interpolate(depth_mono, size=(height_ori, width_ori), mode='bilinear', align_corners=False)
        features_left_4x, features_left_8x, features_left_16x, features_left_32x = self.feat_decoder(features_left_encoder, patch_h, patch_w)
        features_right_4x, features_right_8x, features_right_16x, features_right_32x = self.feat_decoder(features_right_encoder, patch_h, patch_w)

        return depth_mono, [features_left_4x, features_left_8x, features_left_16x, features_left_32x], [features_right_4x, features_right_8x, features_right_16x, features_right_32x]

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            if isinstance(m, nn.SyncBatchNorm):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        # with autocast(enabled=self.args.mixed_precision):
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        spx_pred = self.spx_gru(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp


    def _forward_pair(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """

        # image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        # image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with torch.autocast(device_type='cuda', dtype=torch.float32): 
            depth_mono, features_mono_left,  features_mono_right = self.infer_mono(image1, image2)

        scale_factor = 0.25
        size = (int(depth_mono.shape[-2] * scale_factor), int(depth_mono.shape[-1] * scale_factor))

        disp_mono_4x = F.interpolate(depth_mono, size=size, mode='bilinear', align_corners=False)

        features_left = self.feat_transfer(features_mono_left)
        features_right = self.feat_transfer(features_mono_right)
        stem_2x = self.stem_2(image1)
        stem_4x = self.stem_4(stem_2x)
        stem_8x = self.stem_8(stem_4x)
        stem_16x = self.stem_16(stem_8x)
        stem_2y = self.stem_2(image2)
        stem_4y = self.stem_4(stem_2y)

        stem_x_list = [stem_16x, stem_8x, stem_4x]
        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))
        gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8)
        gwc_volume = self.corr_stem(gwc_volume)
        gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
        geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

        # Init disp from geometry encoding volume
        prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
        init_disp = disparity_regression(prob, self.args.max_disp//4)
        
        del prob, gwc_volume

        if not test_mode:
            xspx = self.spx_4(features_left[0])
            xspx = self.spx_2(xspx, stem_2x)
            spx_pred = self.spx(xspx)
            spx_pred = F.softmax(spx_pred, 1)

        # cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
        cnet_list = self.feat_transfer_cnet(features_mono_left, stem_x_list)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [torch.relu(x) for x in inp_list]
        inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]
        net_list_mono = [x.clone() for x in net_list]

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(match_left.float(), match_right.float(), geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1).contiguous()
        disp = init_disp
        disp_preds = []
        for itr in range(iters):
            disp = disp.detach()
            if itr >= int(1):
                disp_mono_4x = disp_mono_4x.detach()
            geo_feat = geo_fn(disp, coords)
            if itr > int(iters-8):
                if itr == int(iters-7):
                    bs, _, _, _ = disp.shape
                    for i in range(bs):
                        with torch.autocast(device_type='cuda', dtype=torch.float32): 
                            scale, shift = compute_scale_shift(disp_mono_4x[i].clone().squeeze(1).to(torch.float32), disp[i].clone().squeeze(1).to(torch.float32))
                        disp_mono_4x[i] = scale * disp_mono_4x[i] + shift
                
                warped_right_mono = disp_warp(features_right[0], disp_mono_4x.clone().to(features_right[0].dtype))[0]  
                flaw_mono = warped_right_mono - features_left[0] 

                warped_right_stereo = disp_warp(features_right[0], disp.clone().to(features_right[0].dtype))[0]  
                flaw_stereo = warped_right_stereo - features_left[0] 
                geo_feat_mono = geo_fn(disp_mono_4x, coords)

            if itr <= int(iters-8):
                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)
            else:
                net_list, mask_feat_4, delta_disp = self.update_block_mix_stereo(net_list, inp_list, flaw_stereo, disp, geo_feat, flaw_mono, disp_mono_4x, geo_feat_mono, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)
                net_list_mono, mask_feat_4_mono, delta_disp_mono = self.update_block_mix_mono(net_list_mono, inp_list, flaw_mono, disp_mono_4x, geo_feat_mono, flaw_stereo, disp, geo_feat, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2)
                disp_mono_4x = disp_mono_4x + delta_disp_mono
                disp_mono_4x_up = self.upsample_disp(disp_mono_4x, mask_feat_4_mono, stem_2x)
                disp_preds.append(disp_mono_4x_up)

            disp = disp + delta_disp
            if test_mode and itr < iters-1:
                continue

            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)

            if itr == iters - 1:
                refine_value = self.REMP(disp_mono_4x_up, disp_up, image1, image2)
                disp_up = disp_up + refine_value
            disp_preds.append(disp_up)
        
        if test_mode:
            return {'disp_pred': disp_up}

        init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
    
        return {
            "init_disp": init_disp,
            "disp_preds": disp_preds, 
            "depth_mono": depth_mono,
            "disp_pred": disp_preds[-1]
        }
    
    def forward(self, data):
        image1 = data['left']
        image2 = data['right']
        test_mode = not self.training
        iters = self.args.train_iters if self.training else self.args.valid_iters

        return self._forward_pair(image1, image2, iters=iters, test_mode=test_mode)


    def get_loss(self, model_pred, input_data):
        disp_preds = model_pred["disp_preds"]
        disp_init_pred = model_pred["init_disp"]
        disp_gt = input_data["disp"]
        loss_gamma = self.args.loss_gamma
        if 'valid' in input_data.keys():
            valid = input_data['valid']
        else:
            mask = (disp_gt < self.max_disp) & (disp_gt > 0)
            valid = mask.float()

        disp_gt = disp_gt.unsqueeze(1)
        mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
        valid = ((valid >= 0.5) & (mag < self.max_disp)).unsqueeze(1)
        assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
        assert not torch.isinf(disp_gt[valid.bool()]).any()

        """ Loss function defined over sequence of flow predictions """
        n_predictions = len(disp_preds)
        assert n_predictions >= 1
        disp_loss = 0.0

        # quantile = torch.quantile((disp_init_pred - disp_gt).abs(), 0.9)
        init_valid = valid.bool() & ~torch.isnan(disp_init_pred)#  & ((disp_init_pred - disp_gt).abs() < quantile)
        disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[init_valid], disp_gt[init_valid], reduction='mean')
        for i in range(n_predictions):
            adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
            i_loss = (disp_preds[i] - disp_gt).abs()
            # quantile = torch.quantile(i_loss, 0.9)
            assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
            disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()

        loss_info = {'scalar/train/loss_disp': disp_loss.item()}
        return disp_loss, loss_info