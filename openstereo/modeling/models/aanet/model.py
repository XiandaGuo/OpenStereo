import torch
import torch.nn as nn
import torch.nn.functional as F
from .aanet import aanet
from modeling.base_model import BaseModel
from base_trainer import BaseTrainer
from modeling.common.lamb import Lamb


class AANet(BaseModel):
    def __init__(self, *args, **kwargs):
        super(AANet, self).__init__(*args, **kwargs)
        self.Trainer = AANet_Trainer
    
    def build_network(self):
        self.aanet=aanet(model_cfg=self.model_cfg)
    
    def forward(self, inputs):
        disparity_pyramid=self.aanet(inputs)
    
        if self.training:
                output = {
                    "training_disp": {
                        "disparity_pyramid": {
                            "disp_ests": disparity_pyramid,
                            "disp_gt": inputs["disp_gt"],
                            "mask": inputs["mask"]
                        },
                    },
                    "visual_summary": {}
                }
        else:
                if  disparity_pyramid[-1].dim()==4:
                    disparity_pyramid[-1]=disparity_pyramid[-1].squeeze(1)
                output = {
                    "inference_disp": {   
                        "disp_est": disparity_pyramid[-1],
                        "disp_gt": inputs["disp_gt"],
                        "mask": inputs["mask"]            
                    },
                    "visual_summary": {}
                }
        return output


class AANet_Trainer(BaseTrainer):
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
        super(AANet_Trainer, self).__init__(model,trainer_cfg,data_cfg,
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