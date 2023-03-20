import torch
import torch.nn as nn
import torch.nn.functional as F


class AffinityFeature(nn.Module):
    def __init__(self, win_h, win_w, dilation, cut):
        super(AffinityFeature, self).__init__()
        self.win_w = win_w
        self.win_h = win_h
        self.dilation = dilation
        self.cut = 0

    def padding(self, x, win_h, win_w, dilation):
        pad_t = (win_w // 2 * dilation, win_w // 2 * dilation,
                 win_h // 2 * dilation, win_h // 2 * dilation)
        out = F.pad(x, pad_t, mode='constant')
        return out

    def forward(self, feature):
        B, C, H, W = feature.size()
        feature = F.normalize(feature, dim=1, p=2)

        # affinity = []
        # pad_feature = self.padding(feature, win_w=self.win_w, win_h=self.win_h, dilation=self.dilation)
        # for i in range(self.win_w):
        #     for j in range(self.win_h):
        #         if (i == self.win_w // 2) & (j == self.win_h // 2):
        #             continue
        #         simi = self.cal_similarity(
        #             pad_feature[:, :, self.dilation*j:self.dilation*j+H, self.dilation*i:self.dilation*i+W],
        #             feature, self.simi_type)
        #
        #         affinity.append(simi)
        # affinity = torch.stack(affinity, dim=1)
        #
        # affinity[affinity < self.cut] = self.cut

        unfold_feature = nn.Unfold(
            kernel_size=(self.win_h, self.win_w), dilation=self.dilation, padding=self.dilation)(feature)
        all_neighbor = unfold_feature.reshape(B, C, -1, H, W).transpose(1, 2)
        num = (self.win_h * self.win_w) // 2
        neighbor = torch.cat((all_neighbor[:, :num], all_neighbor[:, num+1:]), dim=1)
        feature = feature.unsqueeze(1)
        affinity = torch.sum(neighbor * feature, dim=2)
        affinity[affinity < self.cut] = self.cut

        return affinity


