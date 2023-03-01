import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import mark_boundaries
import cv2

import sys
# sys.path.append('./third_party/cython')
# from connectivity import enforce_connectivity

import pdb

class Args:
    def __init__(self):
        self.train_img_height = 256
        self.train_img_width = 512
        self.downsize = 16
        self.batch_size = 8
args = Args()
b_train = True

def init_spixel_grid(train_img_height=256, train_img_width=512, 
                    downsize=16, batch_size=8, 
                    b_train=True):
    if b_train:
        img_height, img_width = train_img_height, train_img_width
    else:
        img_height, img_width = input_img_height, input_img_width

    # get spixel id for the final assignment
    n_spixl_h = int(np.floor(img_height/downsize))
    n_spixl_w = int(np.floor(img_width/downsize))

    spixel_height = int(img_height / (1. * n_spixl_h))
    spixel_width = int(img_width / (1. * n_spixl_w))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    spix_idx_tensor =  np.repeat(
        np.repeat(spix_idx_tensor_, spixel_height,axis=1), spixel_width, axis=2)

    torch_spix_idx_tensor = torch.from_numpy(
                np.tile(spix_idx_tensor, (batch_size, 1, 1, 1))).type(torch.float).cuda()


    curr_img_height = int(np.floor(img_height))
    curr_img_width = int(np.floor(img_width))

    # pixel coord
    all_h_coords = np.arange(0, curr_img_height, 1)
    all_w_coords = np.arange(0, curr_img_width, 1)
    curr_pxl_coord = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing='ij'))

    coord_tensor = np.concatenate([curr_pxl_coord[1:2, :, :], curr_pxl_coord[:1, :, :]])

    all_XY_feat = (torch.from_numpy(
        np.tile(coord_tensor, (batch_size, 1, 1, 1)).astype(np.float32)).cuda())

    return  torch_spix_idx_tensor, all_XY_feat

#===================== pooling and upsampling feature ==========================================

def shift9pos(input, h_shift_unit=1,  w_shift_unit=1):
    # input should be padding as (c, 1+ height+1, 1+width+1)
    input_pd = np.pad(input, ((h_shift_unit, h_shift_unit), (w_shift_unit, w_shift_unit)), mode='edge')
    input_pd = np.expand_dims(input_pd, axis=0)

    # assign to ...
    top     = input_pd[:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
    bottom  = input_pd[:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
    left    = input_pd[:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
    right   = input_pd[:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

    center = input_pd[:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]

    bottom_right    = input_pd[:, 2 * h_shift_unit:,   2 * w_shift_unit:]
    bottom_left     = input_pd[:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
    top_right       = input_pd[:, :-2 * h_shift_unit,  2 * w_shift_unit:]
    top_left        = input_pd[:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

    shift_tensor = np.concatenate([     top_left,    top,      top_right,
                                        left,        center,      right,
                                        bottom_left, bottom,    bottom_right], axis=0)
    return shift_tensor


def poolfeat(input, prob, sp_h=2, sp_w=2):

    def feat_prob_sum(feat_sum, prob_sum, shift_feat):
        feat_sum += shift_feat[:, :-1, :, :]
        prob_sum += shift_feat[:, -1:, :, :]
        return feat_sum, prob_sum

    b, _, h, w = input.shape

    h_shift_unit = 1
    w_shift_unit = 1
    p2d = (w_shift_unit, w_shift_unit, h_shift_unit, h_shift_unit)
    feat_ = torch.cat([input, torch.ones([b, 1, h, w]).cuda()], dim=1)  # b* (n+1) *h*w
    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 0, 1), kernel_size=(sp_h, sp_w),stride=(sp_h, sp_w)) # b * (n+1) * h* w
    send_to_top_left =  F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, 2 * w_shift_unit:]
    feat_sum = send_to_top_left[:, :-1, :, :].clone()
    prob_sum = send_to_top_left[:, -1:, :, :].clone()

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 1, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum,prob_sum,top )

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 2, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    top_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, 2 * h_shift_unit:, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, top_right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 3, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 4, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    center = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, h_shift_unit:-h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, center)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 5, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    right = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, h_shift_unit:-h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, right)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 6, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_left = F.pad(prob_feat, p2d, mode='constant', value=0)[:,  :, :-2 * h_shift_unit, 2 * w_shift_unit:]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_left)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 7, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, w_shift_unit:-w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom)

    prob_feat = F.avg_pool2d(feat_ * prob.narrow(1, 8, 1), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))  # b * (n+1) * h* w
    bottom_right = F.pad(prob_feat, p2d, mode='constant', value=0)[:, :, :-2 * h_shift_unit, :-2 * w_shift_unit]
    feat_sum, prob_sum = feat_prob_sum(feat_sum, prob_sum, bottom_right)


    pooled_feat = feat_sum / (prob_sum + 1e-8)

    return pooled_feat

def poolfeat_(input, prob, sp_h=2, sp_w=2):

    b, c, h, w = input.shape

    feat_ = torch.cat([input, input.new_ones([b, 1, h, w])], dim=1)  
    agg = F.unfold(feat_,3,(sp_h,sp_w),(sp_h,sp_w)).reshape(b,c+1,9,h,w)
    sum_all = (torch.flip(agg,[2])*prob.unsqueeze(1)).sum(2)

    sum_all = F.avg_pool2d(sum_all, kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w))
    pooled_feat = sum_all[:,:-1] / (sum_all[:,-1:] + 1e-8)

    return pooled_feat

def poolfeat_head(input, prob, sp_h=2, sp_w=2):

    b, c, d, h, w = input.shape

    feat_ = torch.cat([input.permute(0,2,1,3,4).reshape(b, -1, h, w), 
                        input.new_ones([b, c, h, w])], dim=1)  
    agg = F.unfold(feat_,3,(sp_h,sp_w),(sp_h,sp_w)).reshape(b,(d+1),c,9,h,w)
    sum_all = (torch.flip(agg,[3])*F.softmax(prob.reshape(b,1,c,9,h,w),3)).sum(3)

    sum_all = F.avg_pool2d(sum_all.reshape(b,-1,h,w), kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w)).reshape(b,-1,c,h//sp_h,w//sp_w)
    pooled_feat = sum_all[:,:-1] / (sum_all[:,-1:] + 1e-8)

    return pooled_feat.permute(0,2,1,3,4)

def poolfeat3d(input, prob, sp_h=2, sp_w=2):

    b, c, d, h, w = input.shape
    
    feat_ = torch.cat([input, input.new_ones([b, c, 1, h, w])], dim=2).reshape(b,-1,h,w)  
    agg = F.unfold(feat_,3,(sp_h,sp_w),(sp_h,sp_w)).reshape(b,1,c,d+1,9,h,w)
    sum_all = (agg*prob.reshape(b,-1,c,1,9,h,w)).sum(2).sum(3).reshape(b,-1,h,w)

    sum_all = F.avg_pool2d(sum_all, kernel_size=(sp_h, sp_w), stride=(sp_h, sp_w)).reshape(b,-1,d+1,h//sp_h,w//sp_w)
    pooled_feat = sum_all[:,:,:-1] / (sum_all[:,:,-1:] + 1e-8)

    return pooled_feat

def upfeat_original(input, prob, up_h=2, up_w=2):
    # input b*n*H*W  downsampled
    # prob b*9*h*w
    b, c, h, w = input.shape

    h_shift = 1
    w_shift = 1

    p2d = (w_shift, w_shift, h_shift, h_shift)
    feat_pd = F.pad(input, p2d, mode='constant', value=0)

    gt_frm_top_left = F.interpolate(feat_pd[:, :, :-2 * h_shift, :-2 * w_shift], size=(h * up_h, w * up_w),mode='nearest')
    feat_sum = gt_frm_top_left * prob.narrow(1,0,1)

    top = F.interpolate(feat_pd[:, :, :-2 * h_shift, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top * prob.narrow(1, 1, 1)

    top_right = F.interpolate(feat_pd[:, :, :-2 * h_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += top_right * prob.narrow(1,2,1)

    left = F.interpolate(feat_pd[:, :, h_shift:-w_shift, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += left * prob.narrow(1, 3, 1)

    center = F.interpolate(input, (h * up_h, w * up_w), mode='nearest')
    feat_sum += center * prob.narrow(1, 4, 1)

    right = F.interpolate(feat_pd[:, :, h_shift:-w_shift, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += right * prob.narrow(1, 5, 1)

    bottom_left = F.interpolate(feat_pd[:, :, 2 * h_shift:, :-2 * w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_left * prob.narrow(1, 6, 1)

    bottom = F.interpolate(feat_pd[:, :, 2 * h_shift:, w_shift:-w_shift], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom * prob.narrow(1, 7, 1)

    bottom_right =  F.interpolate(feat_pd[:, :, 2 * h_shift:, 2 * w_shift:], size=(h * up_h, w * up_w), mode='nearest')
    feat_sum += bottom_right * prob.narrow(1, 8, 1)

    return feat_sum


def upfeat(input, prob, up_h=2, up_w=2):
    b, c, h, w = input.shape

    feat = F.unfold(input, 3, 1, 1).reshape(b, -1, h, w)
    feat = F.interpolate(
        feat, (h*up_h, w*up_w), mode='nearest').reshape(
            b, -1, 9, h*up_h, w*up_w)
    feat_sum = (feat*prob.unsqueeze(1)).sum(2)

    return feat_sum


def upfeatHW(input, prob, tgt_h=2, tgt_w=2):
    b, c, h, w = input.shape

    feat = F.unfold(input, 3, 1, 1).reshape(b, -1, h, w)
    feat = F.interpolate(
        feat, size=(tgt_h, tgt_w), mode='nearest').reshape(
            b, -1, 9, tgt_h, tgt_w)
    feat_sum = (feat*prob.unsqueeze(1)).sum(2)

    return feat_sum


def upfeat_slant(input, prob, slant, up_h=2, up_w=2):
    """
    Superpixel upsampling with slant
    Inputs:
        - slant : [b, 2, h, w]
    """
    b, c, h, w = input.shape

    feat = F.unfold(input, 3, 1, 1).reshape(b, -1, h, w)
    feat = F.interpolate(
        feat, (h*up_h, w*up_w), mode='nearest').reshape(
            b, -1, 9, h*up_h, w*up_w)
    feat_sum = (feat*prob.unsqueeze(1)).sum(2)

    return feat_sum


def upfeat_center(input, up_h=2, up_w=2):
    b,c,h,w = input.shape

    input = nn.ReplicationPad2d(1)(input)
    feat = F.unfold(input,3,1,0).reshape(b,-1,h,w)
    feat = F.interpolate(feat,(h*up_h,w*up_w),mode='nearest')

    return feat

def upfeat3d(input, prob, up_h=2, up_w=2):
    b,c,d,h,w = input.shape

    feat = input.reshape(b,-1,h,w)  
    feat = F.unfold(feat,3,1,1).reshape(b,-1,h,w)
    feat = F.interpolate(feat,(h*up_h,w*up_w),mode='nearest').reshape(b,c,d,9,h*up_h,w*up_w)
    feat_sum = (feat*prob.reshape(b,c,1,9,h*up_h,w*up_w)).sum(1).sum(2)
    
    return feat_sum

# ================= - spixel related -=============
def assign2uint8(assign):
    #red up, green mid, blue down, for debug only
    b,c,h,w = assign.shape

    red = torch.cat([torch.ones(size=assign.shape),  torch.zeros(size=[b,2,h,w])],dim=1).cuda()

    green = torch.cat([ torch.zeros(size=[b,1,h,w]),
                      torch.ones(size=assign.shape),
                      torch.zeros(size=[b,1,h,w])],dim=1).cuda()

    blue  = torch.cat([torch.zeros(size=[b,2,h,w]),
                       torch.ones(size=assign.shape)],dim=1).cuda()

    black = torch.zeros(size=[b,3,h,w]).cuda()
    white = torch.ones(size=[b,3,h,w]).cuda()
    # up probablity
    mat_vis = torch.where(assign.type(torch.float) < 0. , white, black)
    mat_vis = torch.where(assign.type(torch.float) >= 0. , red* (assign.type(torch.float)+1)/3, mat_vis)
    mat_vis = torch.where(assign.type(torch.float) >= 3., green*(assign.type(torch.float)-2)/3, mat_vis)
    mat_vis = torch.where(assign.type(torch.float) >= 6., blue * (assign.type(torch.float) - 5.) / 3, mat_vis)

    return (mat_vis * 255.).type(torch.uint8)

def val2uint8(mat,maxVal):
    maxVal_mat = torch.ones(mat.shape).cuda() * maxVal
    mat_vis = torch.where(mat > maxVal_mat, maxVal_mat, mat)
    return (mat_vis * 255. / maxVal).type(torch.uint8)


def update_spixl_map (spixl_map_idx_in, assig_map_in):
    assig_map = assig_map_in.clone()

    b,_,h,w = assig_map.shape
    _, _, id_h, id_w = spixl_map_idx_in.shape

    if (id_h == h) and (id_w == w):
        spixl_map_idx = spixl_map_idx_in
    else:
        spixl_map_idx = F.interpolate(spixl_map_idx_in, size=(h,w), mode='nearest')

    # assig_max,_ = torch.max(assig_map, dim=1, keepdim= True)
    # assignment_ = torch.where(assig_map == assig_max, torch.ones(assig_map.shape).cuda(),torch.zeros(assig_map.shape).cuda())
    # new_spixl_map_ = spixl_map_idx * assignment_ # winner take all
    # new_spixl_map = torch.sum(new_spixl_map_,dim=1,keepdim=True).type(torch.int)
    new_spixl_map = torch.gather(spixl_map_idx,1,torch.argmax(assig_map,1,True))

    return new_spixl_map


def get_spixel_image(given_img, spix_index, n_spixels = 600, b_enforce_connect = False):

    if not isinstance(given_img, np.ndarray):
        given_img_np_ = given_img.detach().cpu().numpy().transpose(1,2,0)
    else: # for cvt lab to rgb case
        given_img_np_ = given_img

    if not isinstance(spix_index, np.ndarray):
        spix_index_np = spix_index.detach().cpu().numpy().transpose(0,1)
    else:
        spix_index_np = spix_index


    h, w = spix_index_np.shape
    given_img_np = cv2.resize(given_img_np_, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

    cur_max = np.max(given_img_np)
    spixel_bd_image = mark_boundaries(given_img_np/cur_max, spix_index_np.astype(int), color = (1,1,0)) #cyna
    return (cur_max*spixel_bd_image).astype(np.float32).transpose(2,0,1), spix_index_np #

# ============ accumulate Q =============================
def spixlIdx(args, b_train = False):
    # code modified from ssn
    if b_train:
        n_spixl_h = int(np.floor(args.train_img_height / args.downsize))
        n_spixl_w = int(np.floor(args.train_img_width / args.downsize))
    else:
        n_spixl_h = int(np.floor(args.input_img_height / args.downsize))
        n_spixl_w = int(np.floor(args.input_img_width / args.downsize))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor = shift9pos(spix_values)

    torch_spix_idx_tensor = torch.from_numpy(
        np.tile(spix_idx_tensor, (args.batch_size, 1, 1, 1))).type(torch.float).cuda()

    return torch_spix_idx_tensor

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)

def batch2img(img):
    b,_,h,w = img.shape
    tmp = img.permute(0,2,3,1)
    for i in range(b):
        if i ==0:
            tmp_stack = tmp[i,:,:,:]
        else:
            tmp_stack = torch.cat([tmp_stack,tmp[i,:,:,:]],dim=-2)
    return  tmp_stack


def build_LABXY_feat(label_in, XY_feat):

    img_lab = label_in.clone().type(torch.float)

    b, _, curr_img_height, curr_img_width = XY_feat.shape
    scale_img =  F.interpolate(img_lab, size=(curr_img_height,curr_img_width), mode='nearest')
    LABXY_feat = torch.cat([scale_img, XY_feat],dim=1)

    return LABXY_feat


def rgb2Lab_torch(img_in, mean_values = None):
    # self implemented function that convert RGB image to LAB
    # inpu img intense should be [0,1] float b*3*h*w
    assert img_in.min() >= 0 and img_in.max()<=1

    img= (img_in.clone() + mean_values.cuda()).clamp(0, 1)

    mask = img > 0.04045
    img[mask] = torch.pow((img[mask] + 0.055) / 1.055, 2.4)
    img[~mask] /= 12.92

    xyz_from_rgb = torch.tensor([[0.412453, 0.357580, 0.180423],
                             [0.212671, 0.715160, 0.072169],
                             [0.019334, 0.119193, 0.950227]]).cuda()
    rgb = img.permute(0,2,3,1)

    xyz_img = torch.matmul(rgb, xyz_from_rgb.transpose_(0,1))


    xyz_ref_white = torch.tensor([0.95047, 1., 1.08883]).cuda()

    # scale by CIE XYZ tristimulus values of the reference white point
    lab = xyz_img / xyz_ref_white

    # Nonlinear distortion and linear transformation
    mask = lab > 0.008856
    lab[mask] = torch.pow(lab[mask], 1. / 3.)
    lab[~mask] = 7.787 * lab[~mask] + 16. / 116.

    x, y, z = lab[..., 0:1], lab[..., 1:2], lab[..., 2:3]

    # Vector scaling
    L = (116. * y) - 16.
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    return torch.cat([L, a, b], dim=-1).permute(0,3,1,2)


def label2one_hot_torch(labels, C=14):
    # w.r.t http://jacobkimmel.github.io/pytorch_onehot/
    '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.

        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            N x 1 x H x W, where N is batch size.
            Each value is an integer representing correct classification.
        C : integer.
            number of classes in labels.

        Returns
        -------
        target : torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        '''
    b,_, h, w = labels.shape
    one_hot = torch.zeros(b, C, h, w, dtype=torch.long).cuda()
    target = one_hot.scatter_(1, labels.type(torch.long).data, 1) #require long type

    return target.type(torch.float32)

