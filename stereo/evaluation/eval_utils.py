# @Time    : 2023/8/29 18:25
# @Author  : zhangchenming
import time
import torch
import torch.distributed as dist
from functools import partial
# from .metric import epe_metric, d1_metric, threshold_metric
from .metric_per_image import epe_metric, d1_metric, threshold_metric
from stereo.utils.common_utils import color_map_tensorboard, write_tensorboard

metric_func_dict = {
    'epe': epe_metric,
    'd1_all': d1_metric,
    'thres_1': partial(threshold_metric, threshold=1),
    'thres_2': partial(threshold_metric, threshold=2),
    'thres_3': partial(threshold_metric, threshold=3),
}

@torch.no_grad()
def eval_one_epoch(current_epoch, local_rank, is_dist,
                    logger, logger_iter_interval, tb_writer, visualization,
                    model, eval_loader, evaluator_cfgs, use_amp):

    epoch_metrics = {}
    for k in evaluator_cfgs.METRIC:
        epoch_metrics[k] = {'indexes': [], 'values': []}

    for i, data in enumerate(eval_loader):
        for k, v in data.items():
            data[k] = v.to(local_rank) if torch.is_tensor(v) else v

        with torch.cuda.amp.autocast(enabled=use_amp):
            infer_start = time.time()
            model_pred = model(data)
            infer_time = time.time() - infer_start

        disp_pred = model_pred['disp_pred']
        disp_gt = data["disp"]
        mask = (disp_gt < evaluator_cfgs.MAX_DISP) & (disp_gt > 0)
        if 'occ_mask' in data and evaluator_cfgs.get('APPLY_OCC_MASK', False):
            mask = mask & (data['occ_mask'] == 255.0)

        for m in evaluator_cfgs.METRIC:
            if m not in metric_func_dict:
                raise ValueError("Unknown metric: {}".format(m))
            metric_func = metric_func_dict[m]
            res = metric_func(disp_pred.squeeze(1), disp_gt, mask)
            epoch_metrics[m]['indexes'].extend(data['index'].tolist())
            epoch_metrics[m]['values'].extend(res.tolist())

        if i % logger_iter_interval == 0:
            message = ('Evaluating Epoch:{:>2d} Iter:{:>4d}/{} InferTime: {:.2f}ms'
                       ).format(current_epoch, i, len(eval_loader), infer_time * 1000)
            logger.info(message)

            if visualization and tb_writer is not None:
                tb_info = {
                    'image/eval/image': torch.cat([data['left'][0], data['right'][0]], dim=1) / 256,
                    'image/eval/disp': color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])
                }
                write_tensorboard(tb_writer, tb_info, current_epoch * len(eval_loader) + i)

    # gather from all gpus
    if is_dist:
        dist.barrier()
        logger.info("Start reduce metrics.")
        for k in epoch_metrics.keys():
            indexes = torch.tensor(epoch_metrics[k]["indexes"]).to(local_rank)
            values = torch.tensor(epoch_metrics[k]["values"]).to(local_rank)
            gathered_indexes = [torch.zeros_like(indexes) for _ in range(dist.get_world_size())]
            gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_indexes, indexes)
            dist.all_gather(gathered_values, values)
            unique_dict = {}
            for key, value in zip(torch.cat(gathered_indexes, dim=0).tolist(), torch.cat(gathered_values, dim=0).tolist()):
                if key not in unique_dict:
                    unique_dict[key] = value
            epoch_metrics[k]["indexes"] = list(unique_dict.keys())
            epoch_metrics[k]["values"] = list(unique_dict.values())

    results = {}
    for k in epoch_metrics.keys():
        results[k] = torch.tensor(epoch_metrics[k]["values"]).mean()

    if local_rank == 0 and tb_writer is not None:
        tb_info = {}
        for k, v in results.items():
            tb_info[f'scalar/val/{k}'] = v.item()

        write_tensorboard(tb_writer, tb_info, current_epoch)

    logger.info(f"Epoch {current_epoch} metrics: {results}")


