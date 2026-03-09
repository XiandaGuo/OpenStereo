# @Time    : 2026/2/6 11:39
# @Author  : Qian Zhou
from stereo.modeling.trainer_template import TrainerTemplate
from .igevpp_stereo import IGEVPPStereo
import torch
from stereo.utils import common_utils

__all__ = {
    'IGEVPP': IGEVPPStereo,
}

class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)


    def build_optimizer_and_scheduler(self):
        if self.cfgs.OPTIMIZATION.OPTIMIZER.NAME == 'Lamb':
            from stereo.utils.lamb import Lamb
            optimizer_cls = Lamb
        else:
            optimizer_cls = getattr(torch.optim, self.cfgs.OPTIMIZATION.OPTIMIZER.NAME)
        valid_arg = common_utils.get_valid_args(optimizer_cls, self.cfgs.OPTIMIZATION.OPTIMIZER, ['name'])
        optimizer = optimizer_cls(params=[p for p in self.model.parameters() if p.requires_grad], **valid_arg)

        self.cfgs.OPTIMIZATION.SCHEDULER.TOTAL_STEPS = self.max_iter + 100
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfgs.OPTIMIZATION.SCHEDULER.NAME)
        valid_arg = common_utils.get_valid_args(scheduler_cls, self.cfgs.OPTIMIZATION.SCHEDULER, ['name', 'on_epoch'])
        scheduler = scheduler_cls(optimizer, **valid_arg)

        return optimizer, scheduler