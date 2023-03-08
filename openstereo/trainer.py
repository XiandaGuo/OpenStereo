import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.evaluator import OpenStereoEvaluator
from utils import NoOp, get_attr_from, get_valid_args


class Trainer:
    def __init__(
            self,
            model: nn.Module = None,
            train_loader: DataLoader = None,
            val_loader: DataLoader = None,
            optimizer_cfg: dict = None,
            scheduler_cfg: dict = None,
            evaluator_cfg: dict = None,
            is_dist: bool = True,
            device: torch.device = torch.device('cpu'),
            fp16: bool = False,
            rank: int = None,
    ):
        self.msg_mgr = model.msg_mgr
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.evaluator_cfg = evaluator_cfg
        self.optimizer = NoOp()
        self.warmup = NoOp()
        self.epoch_scheduler = NoOp()
        self.batch_scheduler = NoOp()
        self.evaluator = NoOp()
        self.is_dist = is_dist
        self.rank = rank if is_dist else None
        self.device = torch.device('cuda', rank) if is_dist else device
        self.fp16 = fp16
        self.seed = 1024
        self.save_dir = './results'
        self.current_epoch = 0
        self.current_iter = 0
        if self.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        self.build_model()
        self.build_optimizer(optimizer_cfg)
        self.build_scheduler(scheduler_cfg)
        self.build_warmup(scheduler_cfg)
        self.build_evaluator()

    def build_model(self, checkpoint=None):
        if checkpoint is not None:
            self.load_model(checkpoint)
        self.model.to(self.device)

    def build_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        optimizer = optimizer(params=self.model.parameters(), **valid_arg)
        self.optimizer = optimizer

    def build_scheduler(self, scheduler_cfg):
        self.msg_mgr.log_info(scheduler_cfg)
        Scheduler = get_attr_from([optim.lr_scheduler], scheduler_cfg['scheduler'])
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler', 'warmup', 'on_epoch'])
        scheduler = Scheduler(self.optimizer, **valid_arg)
        if scheduler_cfg.get('on_epoch', True):
            self.epoch_scheduler = scheduler
        else:
            self.batch_scheduler = scheduler

    def build_warmup(self, scheduler_cfg):
        pass

    def build_evaluator(self):
        metrics = self.evaluator_cfg.get('metrics', ['epe', 'd1_all', 'thres_1', 'thres_2', 'thres_3'])
        self.evaluator = OpenStereoEvaluator(metrics, use_np=False)

    def train_epoch(self):
        total_loss = 0
        self.model.train()
        self.model.msg_mgr.log_info(
            f"Total batch: {len(self.train_loader)},"
            f" batch size: {self.train_loader.sampler.batch_size}"
        )
        if self.is_dist and self.rank == 0 or not self.is_dist:
            pbar = tqdm(total=len(self.train_loader), desc=f'Train epoch {self.current_epoch}')
        else:
            pbar = NoOp()
        for i, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            if self.fp16:
                with autocast():
                    outputs = self.model.forward_step(data)
                    loss = self.model.compute_loss(None, outputs)
                    self.scaler.scale(loss).backward()
                    scale = self.scaler.get_scale()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=35, norm_type=2)
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    self.scaler.step(self.optimizer)
                    # Updates the scale for next iteration
                    self.scaler.update()
                    if scale > self.scaler.get_scale():
                        self.model.msg_mgr.log_debug(
                            "Training step skip. Expected the former scale equals to the present, got {} and {}".format(
                                scale, self.scaler.get_scale()))
                        pbar.update(1)
                        continue
            else:
                outputs = self.model.forward_step(data)
                loss = self.model.compute_loss(None, outputs)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=35, norm_type=2)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
            self.current_iter += 1
            self.batch_scheduler.step()
            self.warmup.step()
            total_loss += loss.item() if not torch.isnan(loss) else 0
            pbar.update(1)
            pbar.set_postfix({
                'loss': loss.item(),
                'epoch_loss': total_loss / (i + 1),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        pbar.close()
        total_loss = torch.tensor(total_loss, device=self.device)
        dist.barrier()
        if self.is_dist:
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
        return total_loss / len(self.train_loader)

    def train_model(self, epochs=10):
        for epoch in range(epochs):
            self.model.train()
            self.train_epoch()
            self.epoch_scheduler.step()
            self.save_model(os.path.join(self.save_dir, "checkpoints", f'epoch_{self.current_epoch}.pth'))
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

        if self.is_dist and self.rank == 0 or not self.is_dist:
            pbar = tqdm(total=len(self.val_loader), desc=f'Eval epoch {self.current_epoch}')
        else:
            pbar = NoOp()
        self.model.msg_mgr.log_info(
            f"Total batch: {len(self.val_loader)},"
            f" batch size: {self.val_loader.sampler.batch_size}"
        )
        for i, data in enumerate(self.val_loader):
            batch_inputs = self.model.prepare_inputs(data)
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
        pbar.close()
        for k in epoch_metrics.keys():
            epoch_metrics[k] = torch.tensor(epoch_metrics[k] / len(self.val_loader)).to(self.device)
        if self.is_dist:
            dist.barrier()
            self.msg_mgr.log_debug("Start reduce metrics.")
            for k in epoch_metrics.keys():
                dist.all_reduce(epoch_metrics[k], op=dist.ReduceOp.AVG)
        for k in epoch_metrics.keys():
            epoch_metrics[k] = epoch_metrics[k].item()
        self.msg_mgr.log_info(f"Epoch {self.current_epoch} metrics: {epoch_metrics}")
        return epoch_metrics

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.current_epoch = checkpoint.get('epoch', 0)
        self.current_iter = checkpoint.get('iter', 0)
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
        self.msg_mgr.log_info(f'Model loaded from {path}')

    def save_model(self, path):
        # Only save model from master process
        if self.is_dist and self.rank != 0:
            return
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
        torch.save(state_dict, path)
        self.msg_mgr.log_info(f'Model saved to {path}.')
