import os
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import time
import random
from glob import glob
import cv2
import numpy as np

from . import spixel
from . import spixel_loss

import pdb
'''
Original author:Fengting Yang 

'''

def test(img, spx_pred, val_spixelID, save_path=None, im_num=None, downsize=None, viz_only=False):
    _,_,h,w = img.shape
    # assign the spixel map
    spixl_map = spixel.update_spixl_map(val_spixelID, spx_pred)
    # spixl_map = F.interpolate(spixl_map.type(torch.float), size=(4*h,4*w), mode='nearest').type(torch.int)
    # img = F.interpolate(img, size=(4*h,4*w), mode='nearest')

    spixel_viz, spixel_label_map = spixel.get_spixel_image(img.squeeze(), spixl_map.squeeze())

    # save spixel viz
    if not viz_only:
        if not os.path.isdir(os.path.join(save_path)):
            os.makedirs(os.path.join(save_path))
        spixl_save_name = os.path.join(save_path,'{}_sPixel.png'.format(im_num))
        cv2.imwrite(spixl_save_name, (spixel_viz.transpose(1, 2, 0)*255).astype(np.uint8))

    return (spixel_viz.transpose(1, 2, 0)*255).astype(np.uint8)
