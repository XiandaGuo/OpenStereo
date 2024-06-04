# @Time    : 2023/10/17 16:18
# @Author  : zhangchenming
import sys
import os
import argparse
import datetime
import torch
import torch.distributed as dist
from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.modeling import build_trainer
from cfgs.data_basic import DATA_PATH_DICT

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for eval')
    parser.add_argument('--eval_data_cfg_file', type=str, default=None)
    # dataloader
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='data loader pin memory')

    parser.add_argument('--save_root_dir', type=str, default='./output', help='save root dir for this experiment')

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    if args.eval_data_cfg_file:
        eval_data_yaml_config = common_utils.config_loader(args.eval_data_cfg_file)
        eval_data_cfgs = EasyDict(eval_data_yaml_config)
        cfgs.DATA_CONFIG = eval_data_cfgs.DATA_CONFIG

    args.exp_group_path = os.path.join(cfgs.DATA_CONFIG.DATA_INFOS[0].DATASET, cfgs.MODEL.NAME)

    for each in cfgs.DATA_CONFIG.DATA_INFOS:
        dataset_name = each.DATASET
        if dataset_name == 'KittiDataset':
            dataset_name = 'KittiDataset15' if 'kitti15' in each.DATA_SPLIT.EVALUATING else 'KittiDataset12'
        each.DATA_PATH = DATA_PATH_DICT[dataset_name]

    args.run_mode = 'eval'
    return args, cfgs


def main():
    args, cfgs = parse_config()
    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
    else:
        local_rank = 0
        global_rank = 0

    # env
    torch.cuda.set_device(local_rank)
    seed = 0 if not args.dist_mode else dist.get_rank()
    common_utils.set_random_seed(seed=seed)

    args.output_dir = str(os.path.join(args.save_root_dir, args.exp_group_path, 'eval'))
    os.makedirs(args.output_dir, exist_ok=True)
    if args.dist_mode:
        dist.barrier()
    # log
    log_file = os.path.join(args.output_dir, 'eval_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=local_rank)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'eval_tensorboard')) if global_rank == 0 else None

    # log args and cfgs
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)

    # model
    trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer)
    trainer.evaluate(current_epoch=0)


if __name__ == '__main__':
    main()
