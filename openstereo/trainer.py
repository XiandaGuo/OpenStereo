import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation.evaluator import OpenStereoEvaluator
from utils import NoOp, get_attr_from, get_valid_args


def invTrans(img):
    return T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )(img)


class Trainer:
    def __init__(
            self,
            model: nn.Module = None,
            train_loader: DataLoader = None,
            val_loader: DataLoader = None,
            optimizer_cfg=None,
            scheduler_cfg=None,
            is_dist=True,
            device=None,
            fp16=False,
    ):
        self.msg_mgr = model.msg_mgr
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = NoOp()
        self.warmup = NoOp()
        self.epoch_scheduler = NoOp()
        self.batch_scheduler = NoOp()
        self.is_dist = is_dist
        self.device = torch.device('cuda', device) if is_dist else device
        self.rank = device if is_dist else None
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
        warmup_steps = 1
        warmup_cfg = scheduler_cfg.get('warmup', None)
        if warmup_cfg is not None:
            warmup_steps = warmup_cfg.get('warmup_steps', 1)
        if self.current_iter > warmup_steps:
            warmup_steps = 1

        def get_lr(x):
            x_real = self.current_iter
            return 1 if x_real >= warmup_steps else x_real / warmup_steps

        self.warmup = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: get_lr(x)
        )

    def train_epoch(self):
        total_loss = 0
        self.model.train()
        self.model.msg_mgr.log_info(f"Total batch: {len(self.train_loader)}")
        if not self.is_dist or self.rank == 0:
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
            total_loss += loss.item()
            pbar.update(1)
            pbar.set_postfix({
                'loss': loss.item(),
                'epoch_loss': total_loss / (i + 1),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        pbar.close()
        return total_loss / len(self.train_loader)

    def train_model(self, epochs=10):
        for epoch in range(epochs):
            self.model.train()
            self.train_epoch()
            self.current_epoch += 1
            self.epoch_scheduler.step()
            if self.rank == 0 or not self.is_dist:
                self.save_model(os.path.join(self.save_dir, "checkpoints", f'epoch_{self.current_epoch}.pth'))
            epe, _ = self.val_epoch()
            # print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val EPE: {epe:.4f}')
        print('Training finished.')

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        epoch_epe = 0
        epoch_d1_all = 0
        if not self.is_dist or self.rank == 0:
            pbar = tqdm(total=len(self.val_loader), desc=f'Eval epoch {self.current_epoch}')
        else:
            pbar = NoOp()
        for i, data in enumerate(self.val_loader):
            batch_inputs = self.model.prepare_inputs(data)

            with autocast(enabled=self.fp16):
                inference_disp = self.model.forward(batch_inputs)['inference_disp']
            val_data = {
                'disp_est': inference_disp['disp_est'],
                'disp_gt': batch_inputs['disp_gt'],
                'mask': batch_inputs['mask'],
            }
            val_res = OpenStereoEvaluator(val_data)
            epe = val_res['scalar/val/epe'].item()
            d1_all = val_res['scalar/val/d1_all'].item()

            epoch_epe += epe
            epoch_d1_all += d1_all
            pbar.update(1)
            pbar.set_postfix({
                'epe': epe,
                'epoch_epe': epoch_epe / (i + 1),
                'd1_all': d1_all,
                'epoch_d1_all': epoch_d1_all / (i + 1),
            })
        pbar.close()
        return {
            'epe': epoch_epe / len(self.val_loader),
            'd1_all': epoch_d1_all / len(self.val_loader),
        }

    def predict(self, x):
        self.model.eval()
        x = x.to(self.device)
        with torch.no_grad():
            pred = self.model(x)
            y_pred = (torch.max(torch.exp(pred), 1)[1])
        return y_pred

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint['model'])
        except RuntimeError:
            print('Loaded model is not compatible with current model.')
            return
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.current_iter = checkpoint.get('iter', 0)
        if not isinstance(self.batch_scheduler, NoOp):
            self.batch_scheduler.load_state_dict(checkpoint['batch_scheduler'])
        if not isinstance(self.epoch_scheduler, NoOp):
            self.epoch_scheduler.load_state_dict(checkpoint['epoch_scheduler'])
        print('Model loaded from {}'.format(path))

        # try:
        #     self.optimizer.load_state_dict(checkpoint['optimizer'])
        #     self.current_epoch = checkpoint.get('epoch', 0)
        #     if not isinstance(self.batch_scheduler, NoOp):
        #         self.batch_scheduler.load_state_dict(checkpoint['batch_scheduler'])
        #     if not isinstance(self.epoch_scheduler, NoOp):
        #         self.epoch_scheduler.load_state_dict(checkpoint['epoch_scheduler'])
        # except KeyError:
        #     print('Optimizer and scheduler not loaded.')

    def save_model(self, path):
        state_dict = {
            'model': self.model.state_dict(),
            'epoch': self.current_epoch,
            'iter': self.current_iter,
        }
        if not isinstance(self.optimizer, NoOp):
            state_dict['optimizer'] = self.optimizer.state_dict()
        if not isinstance(self.batch_scheduler, NoOp):
            print('Batch scheduler saved.')
            state_dict['batch_scheduler'] = self.batch_scheduler.state_dict()
        if not isinstance(self.epoch_scheduler, NoOp):
            print('Epoch scheduler saved.')
            state_dict['epoch_scheduler'] = self.epoch_scheduler.state_dict()

        torch.save(state_dict, path)
        print(f'Model saved to {path}.')
