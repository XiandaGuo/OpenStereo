from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from torch.cuda.amp import autocast

from data.kitti15 import KITTI2015
from ..base_model import BaseModel


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # +conv2

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # +x

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class PSMNet(BaseModel):
    def __init__(self, *args, **kargs):
        super(PSMNet, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        self.maxdisp = model_cfg['maxdisp']

        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.Conv3d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm3d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):

        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        # matching
        cost = Variable(
            torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2, self.maxdisp // 4, refimg_fea.size()[2],
                              refimg_fea.size()[3]).zero_()).cuda()

        for i in range(self.maxdisp // 4):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3

    @staticmethod
    def run_train(model):
        while (model.iteration < model.engine_cfg['total_iter']):
            for inputs in model.train_loader:
                imgL, imgR, disp_L = model.inputs_pretreament(inputs)
                mask = (disp_L > 0)
                mask.detach_()
                with autocast(enabled=model.engine_cfg['enable_float16']):
                    output1, output2, output3 = model(imgL, imgR)
                    output1 = torch.squeeze(output1, 1)
                    output2 = torch.squeeze(output2, 1)
                    output3 = torch.squeeze(output3, 1)
                    # training_feat, visual_summary = retval['training_feat'], retval['visual_summary']
                    # del retval

                # loss_sum, loss_info = model.loss_aggregator(training_feat)
                loss_sum = 0.5 * F.smooth_l1_loss(output1[mask], disp_L[mask], size_average=True) \
                           + 0.7 * F.smooth_l1_loss(output2[mask], disp_L[mask], size_average=True) \
                           + F.smooth_l1_loss(output3[mask], disp_L[mask], size_average=True)
                print("per-iter training loss  = %.3f", loss_sum)
                ok = model.train_step(loss_sum)
                if not ok:
                    continue
                #
                # visual_summary.update(loss_info)
                # visual_summary['scalar/learning_rate'] = model.optimizer.param_groups[0]['lr']

                # model.msg_mgr.train_step(loss_info, visual_summary)
                # model.train_step(loss_sum)
                if model.iteration % model.engine_cfg['save_iter'] == 0:
                    # save the checkpoint
                    model.save_ckpt(model.iteration)

                    # run test if with_test = true
                    if model.engine_cfg['with_test']:
                        model.msg_mgr.log_info("Running test...")
                        model.eval()
                        result_dict = model.run_val(model)
                        model.train()
                        if model.cfgs['trainer_cfg']['fix_BN']:
                            model.fix_BN()
                        model.msg_mgr.write_to_tensorboard(result_dict)
                        model.msg_mgr.reset_time()
                if model.iteration >= model.engine_cfg['total_iter']:
                    break

    @staticmethod
    def run_val(model):
        total_size = len(model.test_loader)
        # rank = torch.distributed.get_rank()
        # if rank == 0:
        #     pbar = tqdm(total=total_size, desc='Transforming')
        # else:
        #     pbar = NoOp()
        # batch_size = model.test_loader.batch_sampler.batch_size
        # rest_size = total_size
        # info_dict = Odict()
        total_test_loss = 0
        for inputs in model.test_loader:
            imgL, imgR, disp_L = model.inputs_pretreament(inputs)
            with autocast(enabled=model.engine_cfg['enable_float16']):
                output3 = model.forward(imgL, imgR)

            pred_disp = output3.data.cpu()

            # computing 3-px error#
            true_disp = copy.deepcopy(disp_L)
            index = np.argwhere(true_disp > 0)
            disp_L[index[0][:], index[1][:], index[2][:]] = np.abs(
                true_disp[disp_L[0][:], index[1][:], index[2][:]] - pred_disp[index[0][:], index[1][:], index[2][:]])
            correct = (disp_L[index[0][:], index[1][:], index[2][:]] < 3) | (
                    disp_L[index[0][:], index[1][:], index[2][:]] < true_disp[
                index[0][:], index[1][:], index[2][:]] * 0.05)
            torch.cuda.empty_cache()

            test_loss = 1 - (float(torch.sum(correct)) / float(len(index[0])))
            # print('Iter %d 3-px error in val = %.3f' % (batch_idx, test_loss * 100))
            total_test_loss += test_loss

        # print('epoch %d total 3-px error in val = %.3f' % (epoch, total_test_loss / len(TestImgLoader) * 100))
        # if total_test_loss / len(TestImgLoader) * 100 > max_acc:
        #     max_acc = total_test_loss / len(TestImgLoader) * 100
        #     max_epo = epoch
        # print('MAX epoch %d total test error = %.3f' % (max_epo, max_acc))

        print("total 3-px error in val = %.3f", total_test_loss / total_size)

        # return info_dict

    def get_loader(self, data_cfg, train=True):

        all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = KITTI2015.Imageloader(
            data_cfg['dataset_root'])
        print("len_all_left_img", len(all_left_img))

        if train:

            loader = torch.utils.data.DataLoader(
                KITTI2015(all_left_img, all_right_img, all_left_disp, True),
                batch_size=12, shuffle=True, num_workers=8, drop_last=False)

        else:
            loader = torch.utils.data.DataLoader(
                KITTI2015(test_left_img, test_right_img, test_left_disp, False),
                batch_size=8, shuffle=False, num_workers=4, drop_last=False)
        return loader

    @staticmethod
    def inputs_pretreament(inputs):
        # processed=get_transform(augment=False)
        imgL, imgR, disp_L = inputs
        # imgL=processed(imgL)
        # imgR=processed(imgR)

        return imgL.cuda(), imgR.cuda(), disp_L.cuda()


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])).cuda()

    def forward(self, x):
        out = torch.sum(x * self.disp.data, 1, keepdim=True)
        return out


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature
