import argparse
import os

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed import init_process_group
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.stereo_dataset_batch import StereoBatchDataset
from modeling import models
from trainer import Trainer
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr


def arg_parse():
    parser = argparse.ArgumentParser(description='Main program for OpenStereo.')
    parser.add_argument('--cfgs', type=str, default='configs/psmnet/PSMNet_sceneflow_g8.yaml',
                        help="path of config file")
    parser.add_argument('--scope', default='train', choices=['train', 'val', 'test_kitti'],
                        help="choose train or test scope")
    # set distributed training store true
    parser.add_argument('--no_distribute', action='store_true', default=False, help="disable distributed training")
    parser.add_argument('--log_to_file', action='store_true',
                        help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
    opt = parser.parse_args()
    return opt


def ddp_init(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
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


def dist_worker(rank, world_size, opt, cfgs):
    ddp_init(rank, world_size)
    initialization(opt, cfgs, opt.scope)
    model_cfg = cfgs['model_cfg']
    scope = opt.scope
    Model = getattr(models, model_cfg['model'])
    device = torch.device(f'cuda:{rank}')
    model = Model(cfgs, device, scope)
    data_cfg = cfgs['data_cfg']
    model_trainer = Trainer(
        model=model,
        train_loader=get_data_loader(data_cfg, 'train'),
        val_loader=get_data_loader(data_cfg, 'val'),
        optimizer_cfg=cfgs['optimizer_cfg'],
        scheduler_cfg=cfgs['scheduler_cfg'],
        is_dist=True,
        device=rank,
        fp16=True,
    )
    # model_trainer.load_model('results/checkpoints/epoch_2.pth')
    model_trainer.train_model()
    cleanup()


def collect_fn(batch):
    return batch[0]


def get_data_loader(data_cfg, scope, distributed=True):
    dataset = StereoBatchDataset(data_cfg, scope)
    if distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
        )
    else:
        sampler = RandomSampler(
            dataset,
        )
    sampler = BatchSampler(
        sampler,
        10,
        drop_last=False,
    )
    loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        collate_fn=collect_fn,
        num_workers=4,
        pin_memory=False,
    )
    return loader


def worker(opt, cfgs, device):
    initialization(opt, cfgs, opt.scope)
    model_cfg = cfgs['model_cfg']
    scope = opt.scope
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, device, scope == 'train')
    model.to(device)
    data_cfg = cfgs['data_cfg']
    train_loader = get_data_loader(data_cfg, 'train', False)
    val_loader = get_data_loader(data_cfg, 'val', False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pbar = tqdm(total=len(train_loader), desc="Training: loss=0.000")

    for i, batch_data in enumerate(train_loader):
        loss = model.train_step(batch_data, optimizer)
        pbar.set_description(f'Training: loss={loss.item():.3f}')
        pbar.update(1)
    for i, batch_data in enumerate(val_loader):
        model.val_step(batch_data)


if __name__ == '__main__':
    opt = arg_parse()
    cfgs = config_loader(opt.cfgs)
    is_dist = not opt.no_distribute
    if is_dist:
        world_size = torch.cuda.device_count()
        mp.spawn(dist_worker, args=(world_size, opt, cfgs), nprocs=world_size)
    else:
        device = torch.device('cpu')
        worker(opt, cfgs, device)
