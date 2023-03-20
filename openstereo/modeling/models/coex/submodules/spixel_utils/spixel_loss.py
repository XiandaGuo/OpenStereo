import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .spixel import *

import pdb
'''
Loss function
author:Fengting Yang 
Mar.1st 2019

We only use "compute_semantic_pos_loss" func. in our final version, best result achieved with weight = 3e-3
'''

def compute_semantic_pos_loss(prob_in, labxy_feat,  pos_weight = 0.003,  kernel_size=4):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*50+2*H*W
    # output : B*9*H*w
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    # pdb.set_trace()
    S = kernel_size
    m = pos_weight
    prob = prob_in.clone()

    b, c, h, w = labxy_feat.shape
    pooled_labxy = poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = upfeat(pooled_labxy, prob, kernel_size, kernel_size)

    loss_map = reconstr_feat - labxy_feat

    # self def cross entropy  -- the official one combined softmax
    loss_sem = torch.norm(loss_map[:,:-2,:,:], p=2, dim=1).mean() 
    loss_pos = torch.norm(loss_map[:,-2:,:,:], p=2, dim=1).mean() * m / S

    # empirically we find timing 0.005 tend to better performance
    loss_sum =  1 * (loss_sem + loss_pos)
    loss_sem_sum =  1 * loss_sem
    loss_pos_sum = 1 * loss_pos

    return loss_sum, loss_sem_sum, loss_pos_sum

