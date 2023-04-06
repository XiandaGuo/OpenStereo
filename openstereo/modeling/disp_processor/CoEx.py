import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.common.model_basics.CoEx import BasicConv, Conv2x


def upfeat(input, prob, up_h=2, up_w=2):
    b, c, h, w = input.shape

    feat = F.unfold(input, 3, 1, 1).reshape(b, -1, h, w)
    feat = F.interpolate(
        feat, (h * up_h, w * up_w), mode='nearest').reshape(
        b, -1, 9, h * up_h, w * up_w)
    feat_sum = (feat * prob.unsqueeze(1)).sum(2)
    return feat_sum


class Regression(nn.Module):
    def __init__(self, max_disparity=192, top_k=2):
        super(Regression, self).__init__()
        self.D = int(max_disparity // 4)
        self.top_k = top_k
        self.ind_init = False

    def forward(self, cost, spg):
        b, _, h, w = spg.shape

        corr, disp = self.topkpool(cost, self.top_k)
        corr = F.softmax(corr, 2)

        disp_4 = torch.sum(corr * disp, 2, keepdim=True)
        disp_4 = disp_4.reshape(b, 1, disp_4.shape[-2], disp_4.shape[-1])
        disp_1 = upfeat(disp_4, spg, 4, 4)
        disp_1 = disp_1.squeeze(1) * 4  # + 1.5

        if self.training:
            disp_4 = disp_4.squeeze(1) * 4  # + 1.5
            return [disp_1, disp_4]
        else:
            return [disp_1]

    def topkpool(self, cost, k):
        if k == 1:
            _, ind = cost.sort(2, True)
            pool_ind_ = ind[:, :, :k]
            b, _, _, h, w = pool_ind_.shape
            pool_ind = pool_ind_.new_zeros((b, 1, 3, h, w))
            pool_ind[:, :, 1:2] = pool_ind_
            pool_ind[:, :, 0:1] = torch.max(
                pool_ind_ - 1, pool_ind_.new_zeros(pool_ind_.shape))
            pool_ind[:, :, 2:] = torch.min(
                pool_ind_ + 1, self.D * pool_ind_.new_ones(pool_ind_.shape))
            cv = torch.gather(cost, 2, pool_ind)

            disp = pool_ind

        else:
            _, ind = cost.sort(2, True)
            pool_ind = ind[:, :, :k]
            cv = torch.gather(cost, 2, pool_ind)

            disp = pool_ind

        return cv, disp


class CoExDispProcessor(nn.Module):
    def __init__(self, max_disp=192, regression_topk=2, chans =[16, 24, 32, 96, 160]):
        super().__init__()
        self.max_disp = max_disp
        self.regression_topk = regression_topk
        self.chans = chans
        self.spixel_branch_channels = [32, 48]
        self.spx = nn.Sequential(nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1), )
        self.spx_2 = Conv2x(self.chans[1], 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(self.chans[1] * 2 + self.spixel_branch_channels[1], self.chans[1], kernel_size=3, stride=1,
                      padding=1),
            nn.Conv2d(self.chans[1], self.chans[1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.chans[1]),
            nn.ReLU()
        )
        self.regression = Regression(
            max_disparity=self.max_disp,
            top_k=self.regression_topk
        )

    def forward(self, inputs):
        shape = inputs['ref_img'].shape
        x = inputs['ref_feature']
        cost = inputs['cost_volume']
        stem_2x = inputs['stem_2x']

        xspx = self.spx_4(x[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)
        # Regression
        disp_pred = self.regression(cost, spx_pred)
        if self.training:
            r1 = disp_pred[1]
            disp_pred[1] = F.interpolate(r1.unsqueeze(1), size=(shape[2], shape[3]), mode='bilinear').squeeze(1)

        ref_img = inputs["ref_img"]
        tgt_img = inputs["tgt_img"]

        if self.training:
            output = {
                "training_disp": {
                    "disp": {
                        "disp_ests": disp_pred,
                        "disp_gt": inputs['disp_gt'],
                        "mask": inputs['mask']
                    },
                },
                "visual_summary": {
                    'image/train/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/train/disp_c': torch.cat([inputs['disp_gt'][0], disp_pred[0][0]], dim=0),
                },
            }

        else:
            disp_pred = disp_pred[0]
            output = {
                "inference_disp": {
                    "disp_est": disp_pred,
                },
                "visual_summary": {
                    'image/test/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/test/disp_c': disp_pred[0],
                }
            }
            if 'disp_gt' in inputs:
                output['visual_summary'] = {
                    'image/val/image_c': torch.cat([ref_img[0], tgt_img[0]], dim=1),
                    'image/val/disp_c': torch.cat([inputs['disp_gt'][0], disp_pred[0]], dim=0),
                }
        return output
