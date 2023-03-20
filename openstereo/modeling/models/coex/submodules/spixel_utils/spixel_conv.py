import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import mark_boundaries
import cv2

import sys
# sys.path.append('./third_party/cython')
# from connectivity import enforce_connectivity

from ..util_conv import BasicConv, Conv2x

import pdb

class spixel_conv2_5d(nn.Module):
    def __init__(self, in_chan, out_chan, spixel_downsize=4):
        super(spixel_conv2_5d, self).__init__()
        '''
        inputs : 
            - spixel downsize
            - 
        '''
        self.spixel_downsize = spixel_downsize
        sp = spixel_downsize

        self.conv_stem = BasicConv(9*in_chan,out_chan//sp,is_3d=True,kernel_size=1,padding=0,stride=1)
        self.conv = nn.ModuleList()
        while True:
            self.conv.append(
                nn.Sequential(
                    BasicConv(out_chan//sp,2*out_chan//sp,is_3d=True,kernel_size=(3,2,2),padding=(1,0,0),stride=2),
                    BasicConv(2*out_chan//sp,2*out_chan//sp,is_3d=True,kernel_size=(3,1,1),padding=(1,0,0),stride=1),
                )
            )
            sp = sp//2
            if sp == 1: break

    def forward(self, cv, sp, xy):
        '''
        inputs :
            - cost volume cv (b,c,d,h,w)
            - spixel confidence sp (b,9,h,w)
            - xy (b,2,h,w)
        outputs :
            - reduced cost volume 
        '''
        b, c, d, h, w = cv.shape
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        feat = cv.new_zeros((b,c,9,d,h,w))
        prob = sp.new_zeros(sp.shape)
        uv = xy.new_zeros((b,2,9,h,w))

        p = sp[:,0,self.spixel_downsize:,self.spixel_downsize:]
        f = cv[:,:,:,self.spixel_downsize:,self.spixel_downsize:]
        u = xy[:,:,self.spixel_downsize:,self.spixel_downsize:]
        prob[:,0,:-self.spixel_downsize,:-self.spixel_downsize] = p
        feat[:,:,0,:,:-self.spixel_downsize,:-self.spixel_downsize] = f*p.unsqueeze(1).unsqueeze(1)
        uv[:,:,0,:-self.spixel_downsize,:-self.spixel_downsize] = u*p.unsqueeze(1)

        p = sp[:,1,self.spixel_downsize:,:]
        f = cv[:,:,:,self.spixel_downsize:,:]
        u = xy[:,:,self.spixel_downsize:,:]
        prob[:,1,:-self.spixel_downsize,:] = p
        feat[:,:,1,:,:-self.spixel_downsize,:] = f*p.unsqueeze(1).unsqueeze(1)
        uv[:,:,0,:-self.spixel_downsize,:] = u*p.unsqueeze(1)

        p = sp[:,2,self.spixel_downsize:,:-self.spixel_downsize]
        f = cv[:,:,:,self.spixel_downsize:,:-self.spixel_downsize]
        u = xy[:,:,self.spixel_downsize:,:-self.spixel_downsize]
        prob[:,2,:-self.spixel_downsize,self.spixel_downsize:] = p
        feat[:,:,2,:,:-self.spixel_downsize,self.spixel_downsize:] = f*p.unsqueeze(1).unsqueeze(1)
        uv[:,:,0,:-self.spixel_downsize,self.spixel_downsize:] = u*p.unsqueeze(1)

        p = sp[:,3,:,self.spixel_downsize:]
        f = cv[:,:,:,:,self.spixel_downsize:]
        u = xy[:,:,:,self.spixel_downsize:]
        prob[:,3,:,:-self.spixel_downsize] = p
        feat[:,:,3,:,:,:-self.spixel_downsize] = f*p.unsqueeze(1).unsqueeze(1)
        uv[:,:,0,:,:-self.spixel_downsize] = u*p.unsqueeze(1)

        p = sp[:,4,:,:]
        f = cv[:,:,:,:,:]
        u = xy[:,:,:,:]
        prob[:,4,:,:] = p
        feat[:,:,4,:,:,:] = f*p.unsqueeze(1).unsqueeze(1)
        uv[:,:,0,:,:] = u*p.unsqueeze(1)

        p = sp[:,5,:,:-self.spixel_downsize]
        f = cv[:,:,:,:,:-self.spixel_downsize]
        u = xy[:,:,:,:-self.spixel_downsize]
        prob[:,5,:,self.spixel_downsize:] = p
        feat[:,:,5,:,:,self.spixel_downsize:] = f*p.unsqueeze(1).unsqueeze(1)
        uv[:,:,0,:,self.spixel_downsize:] = u*p.unsqueeze(1)

        p = sp[:,6,:-self.spixel_downsize,self.spixel_downsize:]
        f = cv[:,:,:,:-self.spixel_downsize,self.spixel_downsize:]
        u = xy[:,:,:-self.spixel_downsize,self.spixel_downsize:]
        prob[:,6,self.spixel_downsize:,:-self.spixel_downsize] = p
        feat[:,:,6,:,self.spixel_downsize:,:-self.spixel_downsize] = f*p.unsqueeze(1).unsqueeze(1)
        uv[:,:,0,self.spixel_downsize:,:-self.spixel_downsize] = u*p.unsqueeze(1)

        p = sp[:,7,:-self.spixel_downsize,:]
        f = cv[:,:,:,:-self.spixel_downsize,:]
        u = xy[:,:,:-self.spixel_downsize,:]
        prob[:,7,self.spixel_downsize:,:] = p
        feat[:,:,7,:,self.spixel_downsize:,:] = f*p.unsqueeze(1).unsqueeze(1)
        uv[:,:,0,self.spixel_downsize:,:] = u*p.unsqueeze(1)

        p = sp[:,8,:-self.spixel_downsize,:-self.spixel_downsize]
        f = cv[:,:,:,:-self.spixel_downsize,:-self.spixel_downsize]
        u = xy[:,:,:-self.spixel_downsize,:-self.spixel_downsize]
        prob[:,8,self.spixel_downsize:,self.spixel_downsize:] = p
        feat[:,:,8,:,self.spixel_downsize:,self.spixel_downsize:] = f*p.unsqueeze(1).unsqueeze(1)
        uv[:,:,0,self.spixel_downsize:,self.spixel_downsize:] = u*p.unsqueeze(1)

        tot = F.avg_pool2d(prob.sum(1,keepdim=True),self.spixel_downsize,self.spixel_downsize)
        tot_ = F.interpolate(tot,None,self.spixel_downsize).unsqueeze(1)
        feat = feat.reshape(b,-1,d,h,w)/tot_

        feat = self.conv_stem(feat)
        cost_feat = []
        spd = self.spixel_downsize
        j = 0
        while True:
            feat = self.conv[j](feat)
            cost_feat.append(feat)
            j += 1
            spd /= 2
            if spd == 1: break
        
        uv = F.avg_pool2d(uv.sum(2),self.spixel_downsize,self.spixel_downsize)/tot
        uv = uv.reshape(b,2,-1)
        # end.record()
        # torch.cuda.synchronize()
        # print(start.elapsed_time(end))

        return cost_feat, uv

class spixel_upsample2_5d(nn.Module):
    def __init__(self, chan, out_dim, bnrelu=True, spixel_downsize=4):
        super(spixel_upsample2_5d, self).__init__()
        self.spixel_downsize = spixel_downsize
        sp = spixel_downsize
        chan = chan//sp

        self.conv = nn.ModuleList()
        cv_mul = 1
        while True:
            if sp == 2: 
                out_chan = out_dim
                bn = bnrelu
                relu = bnrelu
            else: 
                out_chan = sp*chan//2
                bn = True
                relu = True
            self.conv.append(
                nn.Sequential(
                    BasicConv(cv_mul*sp*chan,sp*chan,is_3d=True,deconv=True,kernel_size=(3,1,1),stride=1,padding=(1,0,0)),
                    BasicConv(sp*chan,out_chan,is_3d=True,deconv=True,kernel_size=(4,2,2),stride=2,padding=(1,0,0),bn=bn,relu=relu),
                )
            )
            cv_mul = 2
            sp = sp//2
            if sp == 1: break

    def forward(self, cv, sp):
        ###
        # cv (b,c,d,h,w)
        # sp (b,9,4*h,4*w)
        ###
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        b, c, d, h, w = cv[-1].shape
        cost = cv[-1]
        spd = self.spixel_downsize

        j = 0
        while True:
            if spd == 2:
                cost = self.conv[-1](cost)
                break
            else:
                cost = torch.cat((self.conv[j](cost),cv[-2-j]),1)
                j += 1
                spd /= 2

        agg = cost.new_zeros((b,cost.shape[1],self.spixel_downsize*d,self.spixel_downsize*h,self.spixel_downsize*w))
        sp = sp.unsqueeze(1)

        agg[:,:,:,self.spixel_downsize:,self.spixel_downsize:]  += cost[:,:,:,:-self.spixel_downsize,:-self.spixel_downsize]*sp[:,:,0:1,self.spixel_downsize:,self.spixel_downsize:]
        agg[:,:,:,self.spixel_downsize:,:]                      += cost[:,:,:,:-self.spixel_downsize,:]*sp[:,:,1:2,self.spixel_downsize:,:]
        agg[:,:,:,self.spixel_downsize:,:-self.spixel_downsize] += cost[:,:,:,:-self.spixel_downsize,self.spixel_downsize:]*sp[:,:,2:3,self.spixel_downsize:,:-self.spixel_downsize]
        agg[:,:,:,:,self.spixel_downsize:]                      += cost[:,:,:,:,:-self.spixel_downsize]*sp[:,:,3:4,:,self.spixel_downsize:]
        agg[:,:,:,:,:]                                          += cost[:,:,:,:,:]*sp[:,:,4:5,:,:]
        agg[:,:,:,:,:-self.spixel_downsize]                     += cost[:,:,:,:,self.spixel_downsize:]*sp[:,:,5:6,:,:-self.spixel_downsize]
        agg[:,:,:,:-self.spixel_downsize,self.spixel_downsize:] += cost[:,:,:,self.spixel_downsize:,:-self.spixel_downsize]*sp[:,:,6:7,:-self.spixel_downsize,self.spixel_downsize:]
        agg[:,:,:,:-self.spixel_downsize,:]                     += cost[:,:,:,self.spixel_downsize:,:]*sp[:,:,7:8,:-self.spixel_downsize,:]
        agg[:,:,:,:-self.spixel_downsize,:-self.spixel_downsize] += cost[:,:,:,self.spixel_downsize:,self.spixel_downsize:]*sp[:,:,8:,:-self.spixel_downsize,:-self.spixel_downsize]
        
        # end.record()
        # torch.cuda.synchronize()
        # print(start.elapsed_time(end))
        return agg

class spixel_upsample2d(nn.Module):
    def __init__(self, ):
        super(spixel_upsample2d, self).__init__()


    def forward(self, cv, sp):
        ###
        # cv (b,c,1,h,w)
        # sp (b,9*c,4*h,4*w)
        ###
        b, c, _, h, w = cv.shape
        
        agg = F.unfold(cv.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
        agg = F.interpolate(agg,(h*4,w*4),mode='nearest').reshape(b,c,9,h*4,w*4)
        prob = sp.reshape(b,c,9,4*h,4*w)

        agg = (agg*prob).sum(1).sum(1)
        
        return agg.unsqueeze(1)
