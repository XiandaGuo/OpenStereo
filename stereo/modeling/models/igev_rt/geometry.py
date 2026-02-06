import torch
import torch.nn.functional as F
from .utils import bilinear_sampler


class Geo_Encoding_Volume:
    def __init__(self, geo_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume_pyramid = []
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)
        self.geo_volume_pyramid.append(geo_volume)
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

    def __call__(self, disp):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        dx = torch.linspace(-r, r, 2*r+1)
        dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)
            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)
            out_pyramid.append(geo_volume)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()
