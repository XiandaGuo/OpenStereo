# @Time    : 2024/1/20 03:13
# @Author  : zhangchenming
import os
import time
import math
import glob
import torch
import torch.nn as nn
import torch.distributed as dist

from stereo.datasets import build_dataloader
from stereo.utils import common_utils
from stereo.utils.common_utils import color_map_tensorboard, write_tensorboard
from stereo.utils.warmup import LinearWarmup
from stereo.utils.clip_grad import ClipGrad
from stereo.utils.lamb import Lamb

from stereo.evaluation.eval_utils import eval_one_epoch


class TrainerTemplate:
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer, model):
        self.args = args
        self.cfgs = cfgs
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.logger = logger
        self.tb_writer = tb_writer

        self.model = self.build_model(model)

        if self.args.run_mode in ['train', 'eval']:
            self.eval_set, self.eval_loader, self.eval_sampler = self.build_eval_loader()

        if self.args.run_mode == 'train':
            self.train_set, self.train_loader, self.train_sampler = self.build_train_loader()

            if 'TOTAL_STEPS' in cfgs.OPTIMIZATION.SCHEDULER:
                self.total_epochs = math.ceil(cfgs.OPTIMIZATION.SCHEDULER.TOTAL_STEPS / len(self.train_loader))
            else:
                self.total_epochs = cfgs.OPTIMIZATION.NUM_EPOCHS
                cfgs.OPTIMIZATION.SCHEDULER.TOTAL_STEPS = self.total_epochs * len(self.train_loader)

            self.optimizer, self.scheduler = self.build_optimizer_and_scheduler()
            self.scaler = torch.cuda.amp.GradScaler(enabled=cfgs.OPTIMIZATION.AMP)
            self.last_epoch = -1

            if self.cfgs.MODEL.CKPT > -1 and self.args.run_mode == 'train':
                self.resume_ckpt()

            self.warmup_scheduler = self.build_warmup()
            self.clip_gard = self.build_clip_grad()

    def build_train_loader(self):
        train_set, train_loader, train_sampler = build_dataloader(
            data_cfg=self.cfgs.DATA_CONFIG,
            batch_size=self.cfgs.OPTIMIZATION.BATCH_SIZE_PER_GPU,
            is_dist=self.args.dist_mode,
            workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            mode='training')
        self.logger.info('Total samples for train dataset: %d' % (len(train_set)))
        return train_set, train_loader, train_sampler

    def build_eval_loader(self):
        eval_set, eval_loader, eval_sampler = build_dataloader(
            data_cfg=self.cfgs.DATA_CONFIG,
            batch_size=self.cfgs.EVALUATOR.BATCH_SIZE_PER_GPU,
            is_dist=self.args.dist_mode,
            workers=self.args.workers,
            pin_memory=self.args.pin_memory,
            mode='evaluating')
        self.logger.info('Total samples for eval dataset: %d' % (len(eval_set)))
        return eval_set, eval_loader, eval_sampler

    def build_model(self, model):
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

    def build_optimizer_and_scheduler(self):
        if self.cfgs.OPTIMIZATION.OPTIMIZER.NAME == 'Lamb':
            optimizer_cls = Lamb
        else:
            optimizer_cls = getattr(torch.optim, self.cfgs.OPTIMIZATION.OPTIMIZER.NAME)
        valid_arg = common_utils.get_valid_args(optimizer_cls, self.cfgs.OPTIMIZATION.OPTIMIZER,
                                                ['name', 'specific_lr'])
        base_lr = valid_arg['lr']
        specific_lr = self.cfgs.OPTIMIZATION.OPTIMIZER.get('SPECIFIC_LR', {})
        tmp_model = self.model.module if self.args.dist_mode else self.model
        opt_params = []
        for name, module in tmp_model.named_children():
            each = {'params': [p for p in module.parameters() if p.requires_grad],
                    'lr': specific_lr.get(name, base_lr)}
            opt_params.append(each)
        optimizer = optimizer_cls(params=opt_params, **valid_arg)

        scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfgs.OPTIMIZATION.SCHEDULER.NAME)
        valid_arg = common_utils.get_valid_args(scheduler_cls, self.cfgs.OPTIMIZATION.SCHEDULER,
                                                ['name', 'on_epoch'])
        scheduler = scheduler_cls(optimizer, **valid_arg)

        return optimizer, scheduler

    def resume_ckpt(self):
        self.logger.info('Resume from ckpt:%d' % self.cfgs.MODEL.CKPT)
        ckpt_path = str(os.path.join(self.args.ckpt_dir, 'checkpoint_epoch_%d.pth' % self.cfgs.MODEL.CKPT))
        checkpoint = torch.load(ckpt_path, map_location='cuda:%d' % self.local_rank)
        self.last_epoch = checkpoint['epoch']
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        if self.args.dist_mode:
            self.model.module.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint['model_state'])

    def build_warmup(self):
        last_step = (self.last_epoch + 1) * len(self.train_loader) - 1
        if 'WARMUP' in self.cfgs.OPTIMIZATION.SCHEDULER:
            warmup_steps = self.cfgs.OPTIMIZATION.SCHEDULER.WARMUP.get('WARM_STEPS', 1)
            warmup_scheduler = LinearWarmup(
                self.optimizer,
                warmup_period=warmup_steps,
                last_step=last_step)
        else:
            warmup_scheduler = LinearWarmup(
                self.optimizer,
                warmup_period=1,
                last_step=last_step)

        return warmup_scheduler

    def build_clip_grad(self):
        clip_gard = None
        if 'CLIP_GRAD' in self.cfgs.OPTIMIZATION:
            clip_type = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('TYPE', None)
            clip_value = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('CLIP_VALUE', 0.1)
            max_norm = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('MAX_NORM', 35)
            norm_type = self.cfgs.OPTIMIZATION.CLIP_GRAD.get('NORM_TYPE', 2)
            clip_gard = ClipGrad(clip_type, clip_value, max_norm, norm_type)
        return clip_gard

    def train(self, current_epoch, tbar):
        self.model.train()
        if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
            self.model = common_utils.freeze_bn(self.model)
        if self.args.dist_mode:
            self.train_sampler.set_epoch(current_epoch)
        self.train_one_epoch(current_epoch=current_epoch, tbar=tbar)
        if self.args.dist_mode:
            dist.barrier()
        if self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
            self.scheduler.step()
            self.warmup_scheduler.lrs = [group['lr'] for group in self.optimizer.param_groups]

    def evaluate(self, current_epoch):
        if current_epoch % self.cfgs.TRAINER.EVAL_INTERVAL == 0 or current_epoch == self.total_epochs - 1:
            self.model.eval()
            eval_one_epoch(
                current_epoch=current_epoch,
                local_rank=self.local_rank,
                is_dist=self.args.dist_mode,
                logger=self.logger,
                logger_iter_interval=self.cfgs.TRAINER.LOGGER_ITER_INTERVAL,
                tb_writer=self.tb_writer,
                visualization=self.cfgs.TRAINER.EVAL_VISUALIZATION,
                model=self.model,
                eval_loader=self.eval_loader,
                evaluator_cfgs=self.cfgs.EVALUATOR,
                use_amp=self.cfgs.OPTIMIZATION.AMP)
            if self.args.dist_mode:
                dist.barrier()

    def save_ckpt(self, current_epoch):
        if (current_epoch % self.cfgs.TRAINER.CKPT_SAVE_INTERVAL == 0 or current_epoch == self.total_epochs - 1) and self.global_rank == 0:
            ckpt_list = glob.glob(os.path.join(self.args.ckpt_dir, 'checkpoint_epoch_*.pth'))
            ckpt_list.sort(key=os.path.getmtime)
            if len(ckpt_list) >= self.cfgs.TRAINER.MAX_CKPT_SAVE_NUM:
                for cur_file_idx in range(0, len(ckpt_list) - self.args.max_ckpt_save_num + 1):
                    os.remove(ckpt_list[cur_file_idx])
            ckpt_name = os.path.join(self.args.ckpt_dir, 'checkpoint_epoch_%d.pth' % current_epoch)
            common_utils.save_checkpoint(self.model, self.optimizer, self.scheduler, self.scaler,
                                         self.args.dist_mode, current_epoch, filename=ckpt_name)
        if self.args.dist_mode:
            dist.barrier()

    def train_one_epoch(self, current_epoch, tbar):
        start_epoch = self.last_epoch + 1
        logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL
        total_loss = 0.0
        loss_func = self.model.module.get_loss if self.args.dist_mode else self.model.get_loss

        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):
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

            # warmup_scheduler period>1 和 batch_scheduler 不要同时使用
            with self.warmup_scheduler.dampening():
                if not self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
                    self.scheduler.step()

            total_loss += loss.item()
            total_iter = current_epoch * len(self.train_loader) + i
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
