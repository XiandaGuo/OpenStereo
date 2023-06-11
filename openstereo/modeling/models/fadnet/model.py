from __future__ import print_function
import torch
import torch.optim as optim
import torch.nn as nn
from base_trainer import BaseTrainer
from modeling.common.modules  import *
from modeling.base_model import BaseModel
from utils import get_attr_from,get_valid_args
from modeling.common.lamb import Lamb

class FADNet(BaseModel):
    """
    A general stereo matching model which fits most methods.

    """
    def __init__(self, *args, **kwargs):
        super(FADNet, self).__init__(*args, **kwargs)
        self.relu = nn.ReLU(inplace=False)
        self.model_trt = None
        self.Trainer =Fadnet_Traner
    
    def build_network(self):
        model_cfg=self.model_cfg
        self.maxdisp=model_cfg['base_config']['max_disp']
        self.backbone=self.build_backbone(model_cfg['backbone_cfg'])
        self.cost_processor=self.build_cost_processor(model_cfg['cost_processor_cfg'])
        self.disp_predictor = self.build_disp_processor(model_cfg['disp_processor_cfg'])
    
    def forward(self, inputs,enabled_tensorrt=False):
        #dispnetc_flows, dispnetres_flows, dispnetres_final_flow=self.fadnet(inputs)
         # parse batch
        ref_img = inputs["ref_img"]
        tgt_img = inputs["tgt_img"]

        # extract image feature
        conv1_l, conv2_l, conv3a_l, conv3a_r  = self.backbone(ref_img, tgt_img)

        out_corr = self.build_corr(conv3a_l, conv3a_r, max_disp=self.maxdisp//8+16)

         # generate first-stage flows

        dispnetc_flows = self.cost_processor(ref_img, tgt_img , conv1_l, conv2_l, conv3a_l, out_corr)
        dispnetc_final_flow = dispnetc_flows[0]

          # warp img1 to img0; magnitude of diff between img0 and warped_img1,
        resampled_img1 = warp_right_to_left(tgt_img, -dispnetc_final_flow)
        diff_img0 = ref_img - resampled_img1
        norm_diff_img0 = channel_length(diff_img0)

        inputs_net2 = torch.cat((ref_img,tgt_img,resampled_img1, dispnetc_final_flow, norm_diff_img0), dim = 1)

        if enabled_tensorrt:
            dispnetres_flows = self.disp_predictor(inputs_net2, dispnetc_final_flow)
        else:
            dispnetres_flows = self.disp_predictor(inputs_net2, dispnetc_flows)

        index = 0
        dispnetres_final_flow = dispnetres_flows[index]

        if self.training:
            output = {
                "training_disp": {
                    "dispnetc_flows": {
                        "disp_ests": dispnetc_flows,
                        "disp_gt": inputs["disp_gt"],
                        "mask": inputs["mask"]
                    },
                    "dispnetres_flows":{
                        "disp_ests": dispnetres_flows,
                        "disp_gt": inputs["disp_gt"],
                        "mask": inputs["mask"]
                    }

                },
                "visual_summary": {}
            }
        else:
            if dispnetres_final_flow.dim()==4:
                dispnetres_final_flow=dispnetres_final_flow.squeeze(1)
            
            if 'disp_gt' in inputs:
                output = {
                    "inference_disp": {   
                        "disp_est": dispnetres_final_flow,
                        "disp_gt": inputs["disp_gt"],
                        "mask": inputs["mask"]
                        
                    },
                    "visual_summary": {}
                }
            else:
                output = {
                    "inference_disp": {   
                        "disp_est": dispnetres_final_flow
                    },
                    "visual_summary": {}
                }
        return output
    
    def build_corr(self,img_left, img_right, max_disp=40, zero_volume=None):
        B, C, H, W = img_left.shape
        if zero_volume is not None:
            tmp_zero_volume = zero_volume #* 0.0
            #print('tmp_zero_volume: ', mean)
            volume = tmp_zero_volume
        else:
            volume = img_left.new_zeros([B, max_disp, H, W])
        for i in range(max_disp):
            if (i > 0) & (i < W):
                volume[:, i, :, i:] = (img_left[:, :, :, i:] * img_right[:, :, :, :W-i]).mean(dim=1)
            else:
                volume[:, i, :, :] = (img_left[:, :, :, :] * img_right[:, :, :, :]).mean(dim=1)

        volume = volume.contiguous()
        return volume

class Fadnet_Traner(BaseTrainer):
    def __init__(
            self,
            model: nn.Module = None,
            trainer_cfg: dict = None,
            data_cfg: dict = None,
            is_dist: bool = True,
            rank: int = None,
            device: torch.device = torch.device('cpu'),
            **kwargs
    ):
        super(Fadnet_Traner, self).__init__(model,trainer_cfg,data_cfg,
            is_dist,rank,device, **kwargs)


    def build_optimizer(self, optimizer_cfg):
        if optimizer_cfg['solver']=='lamb':
            #use lamb optimizer
            self.msg_mgr.log_info(optimizer_cfg)
            optimizer = Lamb(filter(lambda p: p.requires_grad, self.model.parameters()), lr=optimizer_cfg['lr'])
        else:
            self.msg_mgr.log_info(optimizer_cfg)
            optimizer = get_attr_from([optim], optimizer_cfg['solver'])
            valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
            optimizer = optimizer(params=[p for p in self.model.parameters() if p.requires_grad], **valid_arg)
        self.optimizer = optimizer
    
    def train_model(self):
        self.msg_mgr.log_info('Training started.')
        total_round = self.trainer_cfg.get('total_round')
        total_epoch = self.trainer_cfg.get('total_epoch')
        for round in range(total_round):
            while self.current_epoch < total_epoch[round]:
                self.train_epoch()
                if self.current_epoch % self.trainer_cfg['save_every'] == 0:
                    self.save_ckpt()
                if self.current_epoch % self.trainer_cfg['val_every'] == 0:
                    self.val_epoch()
        self.msg_mgr.log_info('Training finished.')