import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.stereo_dataset_batch import StereoBatchDataset
from evaluation.evaluator import OpenStereoEvaluator
from modeling.common import ClipGrad, fix_bn
from utils import NoOp, get_attr_from, get_valid_args, mkdir
from utils.warmup import LinearWarmup


class BaseTrainer:
    def __init__(
            self,
            model: nn.Module = None,
            trainer_cfg: dict = None,
            data_cfg: dict = None,
            is_dist: bool = True,
            rank: int = None,
            device: torch.device = torch.device('cpu'),
            **kwargs
    ):
        self.msg_mgr = model.msg_mgr
        self.model = model
        self.trainer_cfg = trainer_cfg
        self.data_cfg = data_cfg
        self.optimizer_cfg = trainer_cfg['optimizer_cfg']
        self.scheduler_cfg = trainer_cfg['scheduler_cfg']
        self.evaluator_cfg = trainer_cfg['evaluator_cfg']
        self.clip_grade_config = trainer_cfg['clip_grad_cfg']
        self.optimizer = None
        self.evaluator = NoOp()
        self.warmup_scheduler = NoOp()
        self.epoch_scheduler = NoOp()
        self.batch_scheduler = NoOp()
        self.clip_gard = NoOp()
        self.is_dist = is_dist
        self.rank = rank if is_dist else None
        self.device = torch.device('cuda', rank) if is_dist else device
        self.current_epoch = 0
        self.current_iter = 0
        self.save_path = os.path.join(
            'output/', self.data_cfg['name'], self.model.model_name, self.trainer_cfg['save_name']
        )
        self.amp = self.trainer_cfg.get('amp', False)
        if self.amp:
            self.scaler = torch.cuda.amp.GradScaler()
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.build_model()
        self.build_data_loader()
        self.build_optimizer(self.optimizer_cfg)
        self.build_scheduler(self.scheduler_cfg)
        self.build_warmup_scheduler(self.scheduler_cfg)
        self.build_evaluator()
        self.build_clip_grad()

    def build_model(self, *args, **kwargs):
        # apply sync batch norm
        if self.is_dist and self.trainer_cfg.get('sync_bn', False):
            self.msg_mgr.log_info('convert batch norm to sync batch norm')
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        # apply fix batch norm
        if self.trainer_cfg.get('fix_bn', False):
            self.msg_mgr.log_info('fix batch norm')
            self.model = fix_bn(self.model)
        # init parameters
        if self.trainer_cfg.get('init_parameters', False):
            self.msg_mgr.log_info('init parameters')
            self.model.init_parameters()

    def build_data_loader(self):
        self.train_loader = self.get_data_loader(self.data_cfg, 'train')
        self.val_loader = self.get_data_loader(self.data_cfg, 'val')

    def get_data_loader(self, data_cfg, scope):
        dataset = StereoBatchDataset(data_cfg, scope)
        batch_size = data_cfg.get(f'{scope}_batch_size', 1)
        num_workers = data_cfg.get('num_workers', 4)
        pin_memory = data_cfg.get('pin_memory', False)
        shuffle = data_cfg.get(f'shuffle', False) if scope == 'train' else False
        if self.is_dist:
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

    def build_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        optimizer = optimizer(params=[p for p in self.model.parameters() if p.requires_grad], **valid_arg)
        self.optimizer = optimizer

    def build_scheduler(self, scheduler_cfg):
        self.msg_mgr.log_info(scheduler_cfg)
        scheduler = get_attr_from([optim.lr_scheduler], scheduler_cfg['scheduler'])
        valid_arg = get_valid_args(scheduler, scheduler_cfg, ['scheduler', 'warmup', 'on_epoch'])
        scheduler = scheduler(self.optimizer, **valid_arg)
        if scheduler_cfg.get('on_epoch', True):
            self.epoch_scheduler = scheduler
        else:
            self.batch_scheduler = scheduler

    def build_warmup_scheduler(self, scheduler_cfg):
        warmup_cfg = scheduler_cfg.get('warmup', None)
        if warmup_cfg is None:
            return
        self.warmup_scheduler = LinearWarmup(
            self.optimizer,
            warmup_period=warmup_cfg.get('warmup_steps', 1),
            last_step=self.current_iter - 1,
        )

    def build_evaluator(self):
        metrics = self.evaluator_cfg.get('metrics', ['epe', 'd1_all', 'thres_1', 'thres_2', 'thres_3'])
        self.evaluator = OpenStereoEvaluator(metrics, use_np=False)

    def build_clip_grad(self):
        clip_type = self.clip_grade_config.get('type', None)
        if clip_type is None:
            return
        clip_value = self.clip_grade_config.get('clip_value', 0.1)
        max_norm = self.clip_grade_config.get('max_norm', 35)
        norm_type = self.clip_grade_config.get('norm_type', 2)
        self.clip_gard = ClipGrad(clip_type, clip_value, max_norm, norm_type)

    def train_epoch(self):
        self.current_epoch += 1
        total_loss = 0
        self.model.train()
        self.msg_mgr.log_info(
            f"Using {dist.get_world_size() if self.is_dist else 1} Device,"
            f" batches on each device: {len(self.train_loader)},"
            f" batch size: {self.train_loader.sampler.batch_size}"
        )
        if self.is_dist and self.rank == 0 or not self.is_dist:
            pbar = tqdm(total=len(self.train_loader), desc=f'Train epoch {self.current_epoch}')
        else:
            pbar = NoOp()
        # for distributed sampler to shuffle data
        # the first sampler is batch sampler and the second is distributed sampler
        if self.is_dist:
            self.train_loader.sampler.sampler.set_epoch(self.current_epoch)
        for i, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if self.amp:
                with autocast():
                    # training_disp, visual_summary = self.model.forward_step(data, device=self.device)
                    # training_disp, visual_summary = self.model.forward_step(data, device=self.device)
                    # ISSUE:
                    #   1. use forward_step will cause torch failed to find unused parameters
                    #   this will cause the model can not sync properly in distributed training
                    batch_inputs = self.model.prepare_inputs(data, device=self.device)
                    outputs = self.model.forward(batch_inputs)
                    training_disp, visual_summary = outputs['training_disp'], outputs['visual_summary']
                    loss, loss_info = self.model.compute_loss(training_disp, inputs=batch_inputs)
                self.scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)  # optional
                self.clip_gard(self.model)
                self.scaler.step(self.optimizer)
                # Updates the scale for next iteration
                self.scaler.update()
            else:
                # training_disp, visual_summary = self.model.forward_step(data, device=self.device)
                # ISSUE:
                #   1. use forward_step will cause torch failed to find unused parameters
                #   this will cause the model can not sync properly in distributed training
                batch_inputs = self.model.prepare_inputs(data, device=self.device)
                outputs = self.model.forward(batch_inputs)
                training_disp, visual_summary = outputs['training_disp'], outputs['visual_summary']
                loss, loss_info = self.model.compute_loss(training_disp, inputs=batch_inputs)
                loss.backward()
                self.clip_gard(self.model)
                self.optimizer.step()
            self.current_iter += 1
            with self.warmup_scheduler.dampening():
                self.batch_scheduler.step()
            total_loss += loss.item() if not torch.isnan(loss) else 0

            log_iter = self.trainer_cfg.get('log_iter', 10)
            if i % log_iter == 0:
                pbar.update(log_iter) if i != 0 else pbar.update(0)
                pbar.set_postfix({
                    'loss': loss.item(),
                    'epoch_loss': total_loss / (i + 1),
                    'lr': self.optimizer.param_groups[0]['lr']
                })
                loss_info.update(visual_summary)
            self.msg_mgr.train_step(loss_info)
        pbar.close()
        total_loss = torch.tensor(total_loss, device=self.device)
        if self.is_dist:
            dist.barrier()
            # reduce loss from all devices
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss /= dist.get_world_size()
        with self.warmup_scheduler.dampening():
            self.epoch_scheduler.step()
        return total_loss.item() / len(self.train_loader)

    def train_model(self):
        self.msg_mgr.log_info('Training started.')
        total_epoch = self.trainer_cfg.get('total_epoch', 10)
        while self.current_epoch <= total_epoch:
            self.train_epoch()
            if self.current_epoch % self.trainer_cfg['save_every'] == 0:
                self.save_ckpt()
            if self.current_epoch % self.trainer_cfg['val_every'] == 0:
                self.val_epoch()
                # self.resume_ckpt(self.current_epoch)
                # self.val_epoch()
        self.msg_mgr.log_info('Training finished.')
        # self.msg_mgr.log_info('Training finished. Testing final model.')
        # self.resume_ckpt(total_epoch)
        # self.val_epoch()

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()

        # init metrics
        epoch_metrics = {}
        for k in self.evaluator.metrics:
            epoch_metrics[k] = 0

        self.msg_mgr.log_info(
            f"Using {dist.get_world_size() if self.is_dist else 1} Device,"
            f" batches on each device: {len(self.val_loader)},"
            f" batch size: {self.val_loader.sampler.batch_size}"
        )

        if self.is_dist and self.rank == 0 or not self.is_dist:
            pbar = tqdm(total=len(self.val_loader), desc=f'Eval epoch {self.current_epoch}')
        else:
            pbar = NoOp()
        for i, data in enumerate(self.val_loader):
            batch_inputs = self.model.prepare_inputs(data, device=self.device)
            with autocast(enabled=self.amp):
                inference_disp = self.model.forward(batch_inputs)['inference_disp']
            val_data = {
                'disp_est': inference_disp['disp_est'],
                'disp_gt': batch_inputs['disp_gt'],
                'mask': batch_inputs['mask'],
            }
            val_res = self.evaluator(val_data)
            for k, v in val_res.items():
                v = v.item() if isinstance(v, torch.Tensor) else v
                epoch_metrics[k] += v
            log_iter = self.trainer_cfg.get('log_iter', 10)
            if i % log_iter == 0:
                pbar.update(log_iter)
                pbar.set_postfix({
                    'epe': val_res['epe'].item(),
                })
        pbar.close()
        for k in epoch_metrics.keys():
            epoch_metrics[k] = torch.tensor(epoch_metrics[k] / len(self.val_loader)).to(self.device)
        if self.is_dist:
            dist.barrier()
            self.msg_mgr.log_debug("Start reduce metrics.")
            for k in epoch_metrics.keys():
                dist.all_reduce(epoch_metrics[k], op=dist.ReduceOp.SUM)
                epoch_metrics[k] /= dist.get_world_size()
        for k in epoch_metrics.keys():
            epoch_metrics[k] = epoch_metrics[k].item()
        visual_info = {}
        for k, v in epoch_metrics.items():
            visual_info[f'scalar/val/{k}'] = v
        self.msg_mgr.write_to_tensorboard(visual_info, self.current_epoch)
        self.msg_mgr.log_info(f"Epoch {self.current_epoch} metrics: {epoch_metrics}")
        return epoch_metrics

    def save_ckpt(self):
        # Only save model from master process
        if not self.is_dist or self.rank == 0:
            mkdir(os.path.join(self.save_path, "checkpoints/"))
            save_name = self.trainer_cfg['save_name']
            state_dict = {
                'model': self.model.state_dict(),
                'epoch': self.current_epoch,
                'iter': self.current_iter,
            }
            # for amp
            if self.amp:
                state_dict['scaler'] = self.scaler.state_dict()
            if not isinstance(self.optimizer, NoOp):
                state_dict['optimizer'] = self.optimizer.state_dict()
            if not isinstance(self.batch_scheduler, NoOp):
                self.msg_mgr.log_debug('Batch scheduler saved.')
                state_dict['batch_scheduler'] = self.batch_scheduler.state_dict()
            if not isinstance(self.epoch_scheduler, NoOp):
                self.msg_mgr.log_debug('Epoch scheduler saved.')
                state_dict['epoch_scheduler'] = self.epoch_scheduler.state_dict()
            save_name = os.path.join(self.save_path, "checkpoints/", f'{save_name}_epoch_{self.current_epoch:0>3}.pt')
            torch.save(
                state_dict,
                save_name
            )
            self.msg_mgr.log_info(f'Model saved to {save_name}')
        if self.is_dist:
            # for distributed training, wait for all processes to finish saving
            dist.barrier()

    def load_ckpt(self, path):
        if not os.path.exists(path):
            self.msg_mgr.log_warning(f"Checkpoint {path} not found.")
            return
        map_location = {'cuda:0': f'cuda:{self.rank}'} if self.is_dist else self.device
        checkpoint = torch.load(path, map_location=map_location)
        self.current_epoch = checkpoint.get('epoch', 0)
        self.current_iter = checkpoint.get('iter', 0)
        self.msg_mgr.iteration = self.current_iter
        try:
            self.model.load_state_dict(checkpoint['model'])
        except RuntimeError:
            self.msg_mgr.log_warning('Loaded model is not compatible with current model.')
            return
        # for amp
        if self.amp:
            if 'scaler' not in checkpoint:
                self.msg_mgr.log_warning('Loaded model is not compatible with current model.')
            else:
                self.scaler.load_state_dict(checkpoint['scaler'])
        try:
            # load optimizer
            if self.trainer_cfg.get('optimizer_reset', False):
                self.msg_mgr.log_info('Optimizer reset.')
                self.build_optimizer(self.optimizer_cfg)
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            # load scheduler
            if self.trainer_cfg.get('scheduler_reset', False):
                self.msg_mgr.log_info('Scheduler reset.')
                self.build_scheduler(self.scheduler_cfg)
            else:
                if not isinstance(self.batch_scheduler, NoOp):
                    self.batch_scheduler.load_state_dict(checkpoint['batch_scheduler'])
                if not isinstance(self.epoch_scheduler, NoOp):
                    self.epoch_scheduler.load_state_dict(checkpoint['epoch_scheduler'])
            # load warmup scheduler
            if self.trainer_cfg.get('warmup_reset', False):
                self.msg_mgr.log_info('Warmup scheduler reset.')
                self.build_warmup_scheduler(self.scheduler_cfg)
            else:
                self.warmup_scheduler.last_step = self.current_iter - 1

        except KeyError:
            self.msg_mgr.log_warning('Optimizer and scheduler not loaded.')

        if not isinstance(self.warmup_scheduler, NoOp):
            self.warmup_scheduler.last_step = self.current_iter
        self.msg_mgr.log_info(f'Model loaded from {path}')

    def resume_ckpt(self, restore_hint):
        restore_hint = str(restore_hint)
        if restore_hint.isdigit():
            save_name = self.trainer_cfg['save_name']
            save_name = os.path.join(
                self.save_path, "checkpoints/", f'{save_name}_epoch_{restore_hint:0>3}.pt'
            )
        else:
            save_name = restore_hint
        self.load_ckpt(save_name)
