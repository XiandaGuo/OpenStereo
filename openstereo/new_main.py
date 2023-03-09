import argparse
import os

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from data.stereo_dataset_batch import StereoBatchDataset
from modeling import models
from trainer import Trainer
from utils import config_loader, init_seeds, get_msg_mgr
from utils.common import DDPPassthrough, params_count


def arg_parse():
    parser = argparse.ArgumentParser(description='Main program for OpenStereo.')
    parser.add_argument('--config', type=str, default='configs/coex/CoExNet_sceneflow_g1.yaml',
                        help="path of config file")
    parser.add_argument('--scope', default='train', choices=['train', 'val', 'test_kitti'],
                        help="choose train or test scope")
    # set distributed training store true
    parser.add_argument('--master_addr', type=str, default='localhost', help="master address")
    parser.add_argument('--master_port', type=str, default='12355', help="master port")
    parser.add_argument('--no_distribute', action='store_true', default=True, help="disable distributed training")
    parser.add_argument('--log_to_file', action='store_true',
                        help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
    parser.add_argument('--device', type=str, default='cuda', help="device to use for non-distributed mode.")

    opt = parser.parse_args()
    return opt


def ddp_init(rank, world_size, master_addr, master_port):
    if master_addr is not None:
        os.environ["MASTER_ADDR"] = master_addr
    if master_port is not None:
        os.environ["MASTER_PORT"] = master_port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def initialization(opt, cfgs, scope):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs[f'{scope}_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['name'], cfgs['model_cfg']['model'], engine_cfg['save_name'])
    iteration = engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], int) else 0
    log_iter = engine_cfg['log_iter']
    msg_mgr.init_manager(output_path, opt.log_to_file, log_iter, iteration)
    msg_mgr.log_info(engine_cfg)
    if torch.distributed.is_initialized():
        seed = torch.distributed.get_rank()
        init_seeds(seed)
    else:
        init_seeds(0)


def get_data_loader(data_cfg, scope, distributed=True):
    dataset = StereoBatchDataset(data_cfg, scope)
    batch_size = data_cfg.get(f'{scope}_batch_size', 1)
    num_workers = data_cfg['num_workers']
    pin_memory = data_cfg['pin_memory']
    shuffle = data_cfg.get(f'{scope}_shuffle', False)
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    sampler = BatchSampler(sampler, batch_size, drop_last=False)
    loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        collate_fn=dataset.collect_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


def dist_worker(rank, world_size, opt, cfgs):
    ddp_init(rank, world_size, opt.master_addr, opt.master_port)
    initialization(opt, cfgs, opt.scope)
    model_cfg = cfgs['model_cfg']
    scope = opt.scope
    Model = getattr(models, model_cfg['model'])
    device = torch.device(f'cuda:{rank}')
    model = Model(cfgs, device, scope)
    if cfgs['train_cfg'].get('sync_bn', False):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['train_cfg'].get('fix_bn', False):
        model.fix_bn()
    model = model.to(device)
    model = DDPPassthrough(model, device_ids=[rank])  # DDPmodel
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")
    data_cfg = cfgs['data_cfg']
    model_trainer = Trainer(
        model=model,
        train_loader=get_data_loader(data_cfg, 'train'),
        val_loader=get_data_loader(data_cfg, 'val'),
        optimizer_cfg=cfgs['optimizer_cfg'],
        scheduler_cfg=cfgs['scheduler_cfg'],
        evaluator_cfg=cfgs['evaluator_cfg'],
        device=torch.device('cuda'),
        rank=rank,
        fp16=True,
        is_dist=True,
    )
    # model_trainer.load_model('results/checkpoints/epoch_2.pth')
    model_trainer.train_model(15)
    # model_trainer.val_epoch()
    cleanup()


def worker(opt, cfgs, device):
    initialization(opt, cfgs, opt.scope)
    model_cfg = cfgs['model_cfg']
    scope = opt.scope
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, device, scope)
    data_cfg = cfgs['data_cfg']
    model = model.to(device)
    # model.fix_bn()
    model_trainer = Trainer(
        model=model,
        train_loader=get_data_loader(data_cfg, 'train', distributed=False),
        val_loader=get_data_loader(data_cfg, 'val', distributed=False),
        optimizer_cfg=cfgs['optimizer_cfg'],
        scheduler_cfg=cfgs['scheduler_cfg'],
        evaluator_cfg=cfgs['evaluator_cfg'],
        device=device,
        fp16=True,
        is_dist=False,
    )
    # model_trainer.load_model('results/checkpoints/epoch_0.pth')
    model_trainer.train_model()
    # model_trainer.val_epoch()


if __name__ == '__main__':
    opt = arg_parse()
    cfgs = config_loader(opt.config)
    is_dist = not opt.no_distribute
    if is_dist:
        world_size = torch.cuda.device_count()
        mp.spawn(dist_worker, args=(world_size, opt, cfgs), nprocs=world_size)
    else:
        print("Non-distributed mode.")
        worker(opt, cfgs, torch.device(opt.device))
