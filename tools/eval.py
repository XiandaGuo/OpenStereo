# @Time    : 2023/10/17 16:18
# @Author  : zhangchenming
import glob
import sys
import os
import argparse
import datetime
import thop
import torch
import torch.distributed as dist
from easydict import EasyDict
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.datasets import build_dataloader
from stereo.modeling import build_trainer
from stereo.evaluation.eval_utils import eval_one_epoch
from stereo.utils.common_utils import load_params_from_file
from cfgs.data_basic import DATA_PATH_DICT

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for eval')
    parser.add_argument('--eval_data_cfg_file', type=str, default=None)
    # dataloader
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='data loader pin memory')
    # parameters
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    # interval
    parser.add_argument('--logger_iter_interval', type=int, default=1, help='')
    parser.add_argument('--profile', action='store_true', default=False, help='')

    args = parser.parse_args()
    args.output_dir = str(Path(args.pretrained_model).parent.parent)

    if args.cfg_file is None:
        yaml_files = glob.glob(os.path.join(args.output_dir, '*.yaml'), recursive=False)
        args.cfg_file = yaml_files[0]

    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

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

    # log
    log_file = os.path.join(args.output_dir, 'eval_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=local_rank)
    tb_writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'eval_tensorboard')) if global_rank == 0 else None

    # log args and cfgs
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)

    # eval dataset
    data_yaml_config = common_utils.config_loader(args.eval_data_cfg_file)
    data_cfgs = EasyDict(data_yaml_config)
    for each in data_cfgs.DATA_CONFIG.DATA_INFOS:
        dataset_name = each.DATASET
        if dataset_name == 'KittiDataset':
            dataset_name = 'KittiDataset15' if 'kitti15' in each.DATA_SPLIT.EVALUATING else 'KittiDataset12'
        each.DATA_PATH = DATA_PATH_DICT[dataset_name]
    logger.info('')
    logger.info('~~~~~~~~~~~~~~~~~~~~ EVAL DATASET INFO ~~~~~~~~~~~~~~~~~~~~')
    common_utils.log_configs(data_cfgs.DATA_CONFIG, logger=logger)
    eval_set, eval_loader, eval_sampler = build_dataloader(
        data_cfg=data_cfgs.DATA_CONFIG,
        batch_size=data_cfgs.EVALUATOR.BATCH_SIZE_PER_GPU,
        is_dist=args.dist_mode,
        workers=args.workers,
        pin_memory=args.pin_memory,
        mode='evaluating')
    logger.info('Total samples for eval dataset: %d' % (len(eval_set)))

    # model
    trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer)
    model = trainer.model

    # load pretrained model
    if args.pretrained_model is not None:
        if not os.path.isfile(args.pretrained_model):
            raise FileNotFoundError
        logger.info('Loading parameters from checkpoint %s' % args.pretrained_model)
        load_params_from_file(model, args.pretrained_model, device='cuda:%d' % local_rank,
                              dist_mode=args.dist_mode, logger=logger, strict=False)

    # profile
    if args.profile:
        model.eval()
        inputs = {'left': torch.randn(1, 3, 320, 736).cuda(),
                  'right': torch.randn(1, 3, 320, 736).cuda()}
        flops, params = thop.profile(model, inputs=(inputs,))
        logger.info("Number of calculates: %.2fGFlops" % (flops / 1e9))
        logger.info("Number of parameters: %.2fM" % (params / 1e6))

    model.eval()
    eval_one_epoch(
        current_epoch=0,
        local_rank=local_rank,
        is_dist=args.dist_mode,
        logger=logger,
        logger_iter_interval=args.logger_iter_interval,
        tb_writer=tb_writer,
        visualization=True,
        model=model,
        eval_loader=eval_loader,
        evaluator_cfgs=data_cfgs.EVALUATOR,
        use_amp=cfgs.OPTIMIZATION.AMP)


if __name__ == '__main__':
    main()
