import time
import torch
import torch.optim as optim

from stereo.modeling.trainer_template import TrainerTemplate
from stereo.utils import common_utils
from stereo.utils.common_utils import color_map_tensorboard, write_tensorboard

from .monster import MonSter

__all__ = {
    "MonSter": MonSter,
}

class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)


    def build_optimizer_and_scheduler(self):
        # self.model 可能是 DDP
        model = self.model.module if hasattr(self.model, "module") else self.model
        assert hasattr(model, "feat_decoder"), "model must have attribute feat_decoder"

        feat_params = list(model.feat_decoder.parameters())
        feat_param_ids = set(map(id, feat_params))
        rest_params = [p for p in model.parameters() if p.requires_grad and id(p) not in feat_param_ids]

        base_lr = float(self.cfgs.OPTIMIZATION.OPTIMIZER.LR)
        param_groups = [{"params": feat_params, "lr": base_lr / 2.0},
            {"params": rest_params, "lr": base_lr},]

        optimizer_name = self.cfgs.OPTIMIZATION.OPTIMIZER.NAME
        if optimizer_name == "Lamb":
            from stereo.utils.lamb import Lamb
            optimizer_cls = Lamb
        else:
            optimizer_cls = getattr(optim, optimizer_name)

        valid_arg = common_utils.get_valid_args(optimizer_cls, self.cfgs.OPTIMIZATION.OPTIMIZER, ['name'])
        optimizer = optimizer_cls(param_groups, **valid_arg)

        self.cfgs.OPTIMIZATION.SCHEDULER.TOTAL_STEPS = self.max_iter + 100
        scheduler_cls = getattr(optim.lr_scheduler, self.cfgs.OPTIMIZATION.SCHEDULER.NAME)
        sched_arg = common_utils.get_valid_args(scheduler_cls, self.cfgs.OPTIMIZATION.SCHEDULER, free_keys=['name', 'on_epoch'])
        scheduler = scheduler_cls(optimizer, **sched_arg)

        return optimizer, scheduler

    def train_one_epoch(self, current_epoch, tbar):
        start_epoch = self.last_epoch + 1
        logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL
        total_loss = 0.0

        loss_func = self.model.module.get_loss if self.args.dist_mode else self.model.get_loss

        amp_enabled = bool(self.cfgs.OPTIMIZATION.AMP)
        amp_dtype_cfg = str(self.cfgs.OPTIMIZATION.get("AMP_DTYPE", "fp16")).lower()
        if amp_dtype_cfg in ["bf16", "bfloat16"]:
            amp_dtype = torch.bfloat16
            use_scaler = False
        else:
            amp_dtype = torch.float16
            use_scaler = True

        scaler = self.scaler if use_scaler else None

        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):
            total_iter = current_epoch * len(self.train_loader) + i
            if total_iter >= self.max_iter:
                break

            self.optimizer.zero_grad()
            lr = self.optimizer.param_groups[0]["lr"]

            start_timer = time.time()
            data = next(train_loader_iter)
            for k, v in data.items():
                data[k] = v.to(self.local_rank) if torch.is_tensor(v) else v
            data_timer = time.time()

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                model_pred = self.model(data)
                infer_timer = time.time()
                loss, tb_info = loss_func(model_pred, data)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                if self.clip_gard is not None:
                    self.clip_gard(self.model)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                if self.clip_gard is not None:
                    self.clip_gard(self.model)
                self.optimizer.step()

            with self.warmup_scheduler.dampening():
                if not self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
                    self.scheduler.step()

            total_loss += loss.item()

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

