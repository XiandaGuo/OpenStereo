import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from modeling import models
from utils import config_loader, init_seeds, get_msg_mgr
from utils.common import DDPPassthrough, params_count


def arg_parse():
    parser = argparse.ArgumentParser(description='Main program for OpenStereo.')
    parser.add_argument('--config', type=str, default='configs/psmnet/PSMNet_sceneflow.yaml',
                        help="path of config file")
    parser.add_argument('--scope', default='train', choices=['train', 'val', 'test_kitti'],
                        help="choose train or test scope")
    parser.add_argument('--master_addr', type=str, default='localhost', help="master address")
    parser.add_argument('--master_port', type=str, default='12355', help="master port")
    parser.add_argument('--no_distribute', action='store_true', default=False, help="disable distributed training")
    parser.add_argument('--log_to_file', action='store_true',
                        help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
    parser.add_argument('--device', type=str, default='cuda', help="device to use for non-distributed mode.")
    parser.add_argument('--restore_hint', type=str, default=0, help="restore hint for loading checkpoint.")
    opt = parser.parse_args()
    return opt


def ddp_init(rank, world_size, master_addr, master_port):
    if master_addr is not None:
        os.environ["MASTER_ADDR"] = master_addr
    if master_port is not None:
        os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def initialization(opt, cfgs):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs[f'trainer_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['name'], cfgs['model_cfg']['model'], engine_cfg['save_name'])
    log_iter = engine_cfg['log_iter']
    msg_mgr.init_manager(output_path, opt.log_to_file, log_iter)
    msg_mgr.log_info(engine_cfg)
    seed = 0 if opt.no_distribute else dist.get_rank()
    init_seeds(seed)


def worker(rank, world_size, opt, cfgs):
    is_dist = not opt.no_distribute
    if is_dist:
        ddp_init(rank, world_size, opt.master_addr, opt.master_port)
        torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}') if is_dist else torch.device(opt.device)
    initialization(opt, cfgs)
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    data_cfg = cfgs['data_cfg']
    trainer_cfg = cfgs['trainer_cfg']
    scope = opt.scope
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs)
    Trainer = model.Trainer

    if is_dist and trainer_cfg.get('sync_bn', False):
        msg_mgr.log_info('convert batch norm to sync batch norm')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device)
    if is_dist:
        # for most models, make sure find_unused_parameters is False
        # https://github.com/pytorch/pytorch/issues/43259#issuecomment-706486925
        find_unused_parameters = model_cfg.get('find_unused_parameters', False)
        model = DDPPassthrough(model, device_ids=[rank], output_device=rank,
                               find_unused_parameters=find_unused_parameters)  # DDPmodel
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    model_trainer = Trainer(
        model=model,
        trainer_cfg=trainer_cfg,
        data_cfg=data_cfg,
        is_dist=is_dist,
        rank=rank,
        device=device,
    )

    # restore checkpoint
    restore_hint = opt.restore_hint if str(opt.restore_hint) != "0" else trainer_cfg.get('restore_hint', 0)
    model_trainer.resume_ckpt(restore_hint)

    # run model
    if scope == 'train':
        model_trainer.train_model()
    elif scope == 'val':
        model_trainer.val_epoch()
    elif scope == 'test_kitti':
        # train_loader and val_loader are built in model_trainer
        # build test loader here since it is not used in training
        model_trainer.test_loader = model_trainer.get_data_loader(model_trainer.data_cfg, 'test')
        model_trainer.test_kitti()
    else:
        raise ValueError(f"Unknown scope: {scope}")

    if is_dist:
        dist.destroy_process_group()


if __name__ == '__main__':
    opt = arg_parse()
    cfgs = config_loader(opt.config)
    is_dist = not opt.no_distribute
    if is_dist:
        print("Distributed mode.")
        world_size = torch.cuda.device_count()
        mp.spawn(worker, args=(world_size, opt, cfgs), nprocs=world_size)
    else:
        print("Non-distributed mode.")
        worker(0, None, opt, cfgs)
