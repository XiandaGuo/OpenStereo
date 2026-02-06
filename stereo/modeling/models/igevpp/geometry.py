import torch
import torch.nn.functional as F
from .utils import bilinear_sampler


class Combined_Geo_Encoding_Volume:
    def __init__(self, geo_volume0, geo_volume1, geo_volume2, init_fmap1, init_fmap2, radius=4, num_levels=2):
        self.num_levels = num_levels
        self.radius = radius
        # self.geo_volume_pyramid = []
        self.init_corr_pyramid = []
        self.geo_volume0_pyramid = []

        # all pairs correlation
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w1, _, w2 = init_corr.shape
        b, c, d0, h, w = geo_volume0.shape
        d1 = geo_volume1.shape[2]
        d2 = geo_volume2.shape[2]
        geo_volume0 = geo_volume0.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d0)
        self.geo_volume1 = geo_volume1.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d1)
        self.geo_volume2 = geo_volume2.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d2)

        init_corr = init_corr.reshape(b*h*w1, 1, 1, w2)
        self.init_corr_pyramid.append(init_corr)
        self.geo_volume0_pyramid.append(geo_volume0)
        for i in range(self.num_levels-1):
            geo_volume0 = F.avg_pool2d(geo_volume0, [1,2], stride=[1,2])
            self.geo_volume0_pyramid.append(geo_volume0)

            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)

    def __call__(self, disp, coords):
        r = self.radius
        b, _, h, w = disp.shape
        init_corr_pyramid = []
        geo_feat0_pyramid = []
        dx = torch.linspace(-r, r, 2*r+1)
        dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)

        x1 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2
        y0 = torch.zeros_like(x1)
        disp_lvl1 = torch.cat([x1, y0], dim=-1)
        geo_feat1 = bilinear_sampler(self.geo_volume1, disp_lvl1)
        geo_feat1 = geo_feat1.view(b, h, w, -1)

        x2 = dx + disp.reshape(b*h*w, 1, 1, 1) / 4
        y0 = torch.zeros_like(x2)
        disp_lvl2 = torch.cat([x2, y0], dim=-1)
        geo_feat2 = bilinear_sampler(self.geo_volume2, disp_lvl2)
        geo_feat2 = geo_feat2.view(b, h, w, -1)

        for i in range(self.num_levels):
            geo_volume0 = self.geo_volume0_pyramid[i]
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)
            disp_lvl0 = torch.cat([x0,y0], dim=-1)
            geo_feat0 = bilinear_sampler(geo_volume0, disp_lvl0)
            geo_feat0 = geo_feat0.view(b, h, w, -1)
            geo_feat0_pyramid.append(geo_feat0)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1)
            init_corr_pyramid.append(init_corr)

        init_corr = torch.cat(init_corr_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous().float()
        geo_feat0 = torch.cat(geo_feat0_pyramid, dim=-1)
        geo_feat0 = geo_feat0.permute(0, 3, 1, 2).contiguous().float()
        geo_feat1 = geo_feat1.permute(0, 3, 1, 2).contiguous().float()
        geo_feat2 = geo_feat2.permute(0, 3, 1, 2).contiguous().float()

        return geo_feat0, geo_feat1, geo_feat2, init_corr
 
    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr