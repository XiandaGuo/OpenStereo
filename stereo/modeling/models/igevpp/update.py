import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class DispHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=1):
        super(DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + cq)
        h = (1 - z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, *x):
        # horizontal
        x = torch.cat(x, dim=1)
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h

class GeoEncoder(nn.Module):
    def __init__(self, geo_planes):
        super(GeoEncoder, self).__init__()
        self.convg1 = nn.Conv2d(geo_planes, 128, 1, padding=0)
        self.convg2 = nn.Conv2d(128, 96, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, geo):
        return self.convg2(self.relu(self.convg1(geo)))

class BasicDispEncoder(nn.Module):
    def __init__(self, args):
        super(BasicDispEncoder, self).__init__()
        self.args = args
        geo_planes = (2*args.CORR_RADIUS + 1) * 2 + 96
        self.convc1 = nn.Conv2d(geo_planes, 128, 1, padding=0)
        self.convc2 = nn.Conv2d(128, 96, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 32, 7, padding=3)
        self.convd2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv = nn.Conv2d(96+32, 128-1, 3, padding=1)

    def forward(self, disp, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        disp_ = F.relu(self.convd1(disp))
        disp_ = F.relu(self.convd2(disp_))
        cor_disp = torch.cat([cor, disp_], dim=1)
        out = F.relu(self.conv(cor_disp))
        return torch.cat([out, disp], dim=1)


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)


def interp(x, dest):
    original_dtype = x.dtype
    x_fp32 = x.float()
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    with torch.cuda.amp.autocast(enabled=False):
        output_fp32 = F.interpolate(x_fp32, dest.shape[2:], **interp_args)
    if original_dtype != torch.float32:
        output = output_fp32.to(original_dtype)
    else:
        output = output_fp32
    return output


class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=[]):
        super().__init__()
        self.args = args
        self.geo_encoder0 = GeoEncoder(geo_planes=2 * (2*args.CORR_RADIUS + 1) * 8)
        self.geo_encoder1 = GeoEncoder(geo_planes=(2*args.CORR_RADIUS + 1) * 8)
        self.geo_encoder2 = GeoEncoder(geo_planes=(2*args.CORR_RADIUS + 1) * 8)
        self.encoder = BasicDispEncoder(args)
        encoder_output_dim = 128

        self.gru04 = ConvGRU(hidden_dims[2], encoder_output_dim + hidden_dims[1] * (args.N_GRU_LAYERS > 1))
        self.gru08 = ConvGRU(hidden_dims[1], hidden_dims[0] * (args.N_GRU_LAYERS == 3) + hidden_dims[2])
        self.gru16 = ConvGRU(hidden_dims[0], hidden_dims[1])
        self.disp_head = DispHead(hidden_dims[2], hidden_dim=256, output_dim=1)
        factor = 2**self.args.N_DOWNSAMPLE

        self.mask_feat_4 = nn.Sequential(
            nn.Conv2d(hidden_dims[2], 64, 3, padding=1),
            nn.ReLU(inplace=True))

    def forward(self, net, inp, geo_feat0=None, geo_feat1=None, geo_feat2=None, init_corr=None, selective_weights=None, disp=None, iter04=True, iter08=True, iter16=True, update=True):

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), pool2x(net[1]))
        if iter08:
            if self.args.N_GRU_LAYERS > 2:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]), interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), pool2x(net[0]))
        if iter04:
            geo_feat0 = self.geo_encoder0(geo_feat0)
            geo_feat1 = self.geo_encoder1(geo_feat1)
            geo_feat2 = self.geo_encoder2(geo_feat2)
            geo_feat = selective_weights[:,0:1]*geo_feat0 + selective_weights[:,1:2]*geo_feat1 + selective_weights[:,2:3]*geo_feat2
            geo_feat = torch.cat([geo_feat, init_corr], dim=1)
            disp_features = self.encoder(disp, geo_feat)
            if self.args.N_GRU_LAYERS > 1:
                net[0] = self.gru04(net[0], *(inp[0]), disp_features, interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), disp_features)

        if not update:
            return net

        delta_disp = self.disp_head(net[0])
        mask_feat_4 = self.mask_feat_4(net[0])
        return net, mask_feat_4, delta_disp
