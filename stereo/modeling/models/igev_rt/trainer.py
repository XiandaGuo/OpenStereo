# @Time    : 2026/2/6 11:39
# @Author  : Qian Zhou
from stereo.modeling.trainer_template import TrainerTemplate
from .igev_rt_stereo import IGEVRTtereo

__all__ = {
    'IGEVRT': IGEVRTtereo,
}

class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)
