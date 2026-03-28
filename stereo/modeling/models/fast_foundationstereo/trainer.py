# @Time    : 2026/3/27 11:39
# @Author  : zhouqian
import time
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from stereo.modeling.trainer_template import TrainerTemplate
from stereo.utils import common_utils
from stereo.utils.common_utils import color_map_tensorboard, write_tensorboard
from .core.foundation_stereo import FastFoundationStereo

__all__ = {
    'FastFoundationStereo': FastFoundationStereo,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)

    def load_fast_foundationstereo(self, model, filename, device='cpu', logger=None):
        # for fast_foundationstereo loading, which can be downloaded from https://github.com/NVlabs/Fast-FoundationStereo?tab=readme-ov-file#weights-and-trade-off
        # by Qian Zhou, note that it should be serializable ckpt

        ckpt_model = torch.load(filename, map_location=device, weights_only=False)
        replaced = []
        skipped = []

        for name, module in ckpt_model.named_children():
            if hasattr(model, name):
                setattr(model, name, module)
                replaced.append(name)
            else:
                skipped.append(name)

        if logger:
            logger.info(f"Loaded serialized FastFoundationStereo model")
            # logger.info(f"Replaced top-level modules: {replaced}")
            if skipped:
                logger.info(f"Skipped top-level modules: {skipped}")
        else:
            print("Loaded serialized FastFoundationStereo model")
            # print(f"Replaced top-level modules: {replaced}")
            if skipped:
                print(f"Skipped top-level modules: {skipped}")
        return model
        
    def build_model(self, model):
        if self.cfgs.MODEL.fast_foundationstereo_path is not None:
            self.logger.info('Loading parameters from checkpoint %s' % self.cfgs.MODEL.fast_foundationstereo_path)
            if not os.path.isfile(self.cfgs.MODEL.fast_foundationstereo_path):
                raise FileNotFoundError
            model = self.load_fast_foundationstereo(model, self.cfgs.MODEL.fast_foundationstereo_path, device='cpu', logger=self.logger)

        
        if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
            model = common_utils.freeze_bn(model)
            self.logger.info('Freeze the batch normalization layers')

        if self.cfgs.OPTIMIZATION.SYNC_BN and self.args.dist_mode:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.logger.info('Convert batch norm to sync batch norm')
        model = model.to(self.local_rank)

        if self.args.dist_mode:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=self.cfgs.MODEL.FIND_UNUSED_PARAMETERS)

        # load pretrained model
        if self.cfgs.MODEL.PRETRAINED_MODEL:
            self.logger.info('Loading parameters from checkpoint %s' % self.cfgs.MODEL.PRETRAINED_MODEL)
            if not os.path.isfile(self.cfgs.MODEL.PRETRAINED_MODEL):
                raise FileNotFoundError
            common_utils.load_params_from_file(
                model, self.cfgs.MODEL.PRETRAINED_MODEL, device='cuda:%d' % self.local_rank,
                dist_mode=self.args.dist_mode, logger=self.logger, strict=False)
        return model

    def train_one_epoch(self, current_epoch, tbar):
        start_epoch = self.last_epoch + 1
        logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL
        total_loss = 0.0
        loss_func = self.model.module.get_loss if self.args.dist_mode else self.model.get_loss

        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):
            total_iter = current_epoch * len(self.train_loader) + i
            if total_iter >= self.max_iter:
                break

            self.optimizer.zero_grad()
            lr = self.optimizer.param_groups[0]['lr']

            start_timer = time.time()
            data = next(train_loader_iter)
            for k, v in data.items():
                data[k] = v.to(self.local_rank) if torch.is_tensor(v) else v
            data_timer = time.time()

            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
                model_pred = self.model(data)
                infer_timer = time.time()
                loss, tb_info = loss_func(model_pred, data)

            # ===== Begin: check if loss is NaN =====
            # 1. 本地检查loss是否为NaN/inf
            is_invalid = torch.isnan(loss) | torch.isinf(loss)
            # 转为0/1张量（0=有效，1=无效），用于进程间通信
            invalid_flag = torch.tensor([1], dtype=torch.int, device=loss.device) if is_invalid else torch.tensor(
                [0], dtype=torch.int, device=loss.device)
            # 2. 全局同步：所有进程交换无效标记, 确保所有进程都知道是否有任何进程的loss无效
            dist.all_reduce(invalid_flag, op=dist.ReduceOp.SUM)
            global_invalid = invalid_flag.item() > 0  # 只要有一个进程无效，全局标记为True
            # 3. 所有进程同步决策
            if global_invalid:
                print('loss have nan/inf, continue~')
                del model_pred, loss  # 手动删除中间变量
                torch.cuda.empty_cache()  # 释放GPU内存
                continue
                # torch.save(data)
                #     # self.save_ckpt(current_epoch=12345)
                #     # break
            # ===== End: check if loss is NaN =====

            # 不要在autocast下调用, calls backward() on scaled loss to create scaled gradients.
            self.scaler.scale(loss).backward()
            # 做梯度剪裁的时候需要先unscale, unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)
            # 梯度剪裁
            if self.clip_gard is not None:
                self.clip_gard(self.model)
            # optimizer's gradients are already unscaled, so scaler.step does not unscale them
            self.scaler.step(self.optimizer)
            # Updates the scale for next iteration.
            self.scaler.update()
            # torch.cuda.empty_cache()

            total_loss += loss.item()

            # warmup_scheduler period>1 和 batch_scheduler 不要同时使用
            with self.warmup_scheduler.dampening():
                if not self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
                    self.scheduler.step()

            trained_time_past_all = tbar.format_dict['elapsed']
            single_iter_second = trained_time_past_all / (total_iter + 1 - start_epoch * len(self.train_loader))
            remaining_second_all = single_iter_second * (self.total_epochs * len(self.train_loader) - total_iter - 1)
            if total_iter % logger_iter_interval == 0:
                message = ('Training Epoch:{:>2d}/{} Iter:{:>4d}/{} '
                           'Loss:{:#.6g}({:#.6g}) LR:{:.4e} '
                           'DataTime:{:.2f} InferTime:{:.2f}ms '
                           'Time cost: {}/{}'
                           ).format(current_epoch, self.total_epochs, i, len(self.train_loader),
                                    loss.item(), total_loss / (i + 1), lr,
                                    data_timer - start_timer, (infer_timer - data_timer) * 1000,
                                    tbar.format_interval(trained_time_past_all),
                                    tbar.format_interval(remaining_second_all))
                self.logger.info(message)

            if self.cfgs.TRAINER.TRAIN_VISUALIZATION:
                tb_info['image/train/image'] = torch.cat([data['left'][0], data['right'][0]], dim=1) / 256
                tb_info['image/train/disp'] = color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])

            tb_info.update({'scalar/train/lr': lr})
            if total_iter % logger_iter_interval == 0 and self.local_rank == 0 and self.tb_writer is not None:
                write_tensorboard(self.tb_writer, tb_info, total_iter)
