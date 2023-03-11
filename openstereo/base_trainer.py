import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
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
        self.current_epoch = 1
        self.current_iter = 1
        self.seed = 1024
        self.save_path = os.path.join(
            'output/', self.data_cfg['name'], self.model.model_name, self.trainer_cfg['save_name']
        )
        self.fp16 = self.trainer_cfg.get('fp16', False)
        if self.fp16:
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
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        # apply fix batch norm
        if self.trainer_cfg.get('fix_bn', False):
            self.model = fix_bn(self.model)
        # init parameters
        if self.trainer_cfg.get('init_parameters', False):
            self.model.init_parameters()

    def build_data_loader(self):
        self.train_loader = self.get_data_loader(self.data_cfg, 'train')
        self.val_loader = self.get_data_loader(self.data_cfg, 'val')

    def get_data_loader(self, data_cfg, scope):
        data_cfg = data_cfg
        dataset = StereoBatchDataset(data_cfg, scope)
        batch_size = data_cfg.get(f'{scope}_batch_size', 1)
        num_workers = data_cfg.get('num_workers', 4)
        pin_memory = data_cfg.get('pin_memory', False)
        shuffle = data_cfg.get(f'shuffle', False)
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
        optimizer = optimizer(params=self.model.parameters(), **valid_arg)
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
            if self.fp16:
                with autocast():
                    outputs = self.model.forward_step(data, device=self.device)
                    loss, loss_info = self.model.compute_loss(None, outputs)
                self.scaler.scale(loss).backward()
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual
                self.clip_gard(self.model)
                self.scaler.step(self.optimizer)
                # Updates the scale for next iteration
                self.scaler.update()
            else:
                outputs = self.model.forward_step(data, device=self.device)
                loss, loss_info = self.model.compute_loss(None, outputs)
                loss.backward()
                self.clip_gard(self.model)
                self.optimizer.step()
            self.current_iter += 1
            if self.warmup_scheduler is not None:
                with self.warmup_scheduler.dampening():
                    self.batch_scheduler.step()
            else:
                self.batch_scheduler.step()
            total_loss += loss.item() if not torch.isnan(loss) else 0
            pbar.update(1)
            pbar.set_postfix({
                'loss': loss.item(),
                'epoch_loss': total_loss / (i + 1),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        pbar.close()
        total_loss = torch.tensor(total_loss, device=self.device)
        if self.is_dist:
            dist.barrier()
            # reduce loss from all devices
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss /= dist.get_world_size()
        if self.warmup_scheduler is not None:
            with self.warmup_scheduler.dampening():
                self.epoch_scheduler.step()
        else:
            self.epoch_scheduler.step()
        return total_loss.item() / len(self.train_loader)

    def train_model(self, epochs=10):
        for epoch in range(epochs):
            self.train_epoch()
            if self.current_epoch % self.trainer_cfg['save_every'] == 0:
                self.save_model()
            if self.current_epoch % self.trainer_cfg['val_every'] == 0:
                self.val_epoch()
            self.current_epoch += 1
        self.msg_mgr.log_info('Training finished.')

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
            with autocast(enabled=self.fp16):
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
            pbar.update(1)
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
                # reduce from all devices
                dist.all_reduce(epoch_metrics[k], op=dist.ReduceOp.SUM)
                epoch_metrics[k] /= dist.get_world_size()
        for k in epoch_metrics.keys():
            epoch_metrics[k] = epoch_metrics[k].item()
        self.msg_mgr.log_info(f"Epoch {self.current_epoch} metrics: {epoch_metrics}")
        return epoch_metrics

    def load_model(self, path):
        if not os.path.exists(path):
            self.msg_mgr.log_warning(f"Checkpoint {path} not found.")
            return
        map_location = {'cuda:0': f'cuda:{self.rank}'} if self.is_dist else self.device
        checkpoint = torch.load(path, map_location=map_location)
        self.current_epoch = checkpoint.get('epoch', -1) + 1
        self.current_iter = checkpoint.get('iter', -1) + 1
        try:
            self.model.load_state_dict(checkpoint['model'])
        except RuntimeError:
            self.msg_mgr.log_warning('Loaded model is not compatible with current model.')
            return
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if not isinstance(self.batch_scheduler, NoOp):
                self.batch_scheduler.load_state_dict(checkpoint['batch_scheduler'])
            if not isinstance(self.epoch_scheduler, NoOp):
                self.epoch_scheduler.load_state_dict(checkpoint['epoch_scheduler'])
        except KeyError:
            self.msg_mgr.log_warning('Optimizer and scheduler not loaded.')

        if not isinstance(self.warmup_scheduler, NoOp):
            self.warmup_scheduler.last_step = self.current_iter
        self.msg_mgr.log_info(f'Model loaded from {path}')

    def save_model(self):
        # Only save model from master process
        if self.is_dist and self.rank != 0:
            dist.barrier()
            return
        mkdir(os.path.join(self.save_path, "checkpoints/"))
        save_name = self.trainer_cfg['save_name']
        state_dict = {
            'model': self.model.state_dict(),
            'epoch': self.current_epoch,
            'iter': self.current_iter,
        }
        if not isinstance(self.optimizer, NoOp):
            state_dict['optimizer'] = self.optimizer.state_dict()
        if not isinstance(self.batch_scheduler, NoOp):
            self.msg_mgr.log_debug('Batch scheduler saved.')
            state_dict['batch_scheduler'] = self.batch_scheduler.state_dict()
        if not isinstance(self.epoch_scheduler, NoOp):
            self.msg_mgr.log_debug('Epoch scheduler saved.')
            state_dict['epoch_scheduler'] = self.epoch_scheduler.state_dict()
        torch.save(
            state_dict,
            os.path.join(self.save_path, "checkpoints/", f'{save_name}_epoch_{self.current_epoch:0>3}.pt')
        )
        self.msg_mgr.log_info(f'Model saved to {save_name}_epoch_{self.current_epoch:0>3}.pt')
        if self.is_dist:
            # for distributed training, wait for all processes to finish saving
            dist.barrier()
