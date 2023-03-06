import argparse
import os

import torch
import torch.nn as nn

from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr

parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--cfgs', type=str,
                    default='configs/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test', 'val', 'test_kitti'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
opt = parser.parse_args()


# def set_ddp_env(rank, world_size):
#     torch.cuda.set_device(opt.local_rank)
#     torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
#     torch.backends.cudnn.benchmark = True
#
#
# def cleanup():
#     torch.distributed.destroy_process_group()


def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['name'], cfgs['model_cfg']['model'], engine_cfg['save_name'])
    iteration = engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], int) else 0
    log_iter = engine_cfg['log_iter'] if training else 1
    msg_mgr.init_manager(output_path, opt.log_to_file, log_iter, iteration)
    msg_mgr.log_info(engine_cfg)
    seed = torch.distributed.get_rank()
    init_seeds(seed)


def run_model(cfgs, scope):
    is_train = scope == 'train'
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, scope)
    if is_train and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()
    model = get_ddp_module(model, find_unused_parameters=model_cfg['find_unused_parameters'])

    # if '_set_static_graph' in model_cfg.keys():
    #     if model_cfg['_set_static_graph']:
    #         model._set_static_graph()

    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")
    if scope == 'train':
        model.run_train(model)
    elif scope == 'val':
        res = model.run_val(model)
        msg_mgr.log_info(res)
    elif scope == 'test':
        model.run_test(model)
    elif scope == 'test_kitti':
        model.run_test_kitti(model)
    else:
        raise ValueError("Scope should be one of ['train', 'val', 'test', 'test_kitti'].")


if __name__ == '__main__':
    torch.distributed.init_process_group('nccl', init_method='env://')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of availuable GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))
    cfgs = config_loader(opt.cfgs)
    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)

    assert opt.phase.lower() in ['train', 'val', 'test', 'test_kitti']
    is_train = (opt.phase == 'train')

    initialization(cfgs, is_train)
    run_model(cfgs, opt.phase.lower())
