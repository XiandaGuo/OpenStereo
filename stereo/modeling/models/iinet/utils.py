import numpy as np
import torchvision.utils as vutils
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import torch, random
import torch.nn.functional as F
import torch.nn as nn

# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cuda"))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))



def upsample(x):
    """
    Upsample input tensor by a factor of 2
    """
    return nn.functional.interpolate(
                                x,
                                scale_factor=2,
                                mode="bilinear",
                                align_corners=False,
                            )

def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            normalizer = mpl.colors.Normalize(vmin=img[0].min(), vmax=img[0].max())
            mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')
            cimg = (mapper.to_rgba(img[0])[:,:,:3] *255).astype(np.uint8).transpose(2,0,1)
            cimg = torch.from_numpy(cimg)
            return cimg
        else:
            cimg = torch.from_numpy(img[:1])
            return vutils.make_grid(cimg, padding=0, nrow=1, normalize=True, scale_each=True)


    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


def savepreprocess(img, ifRgb=False):
    if not ifRgb:
        normalizer = mpl.colors.Normalize(vmin=img[0].min(), vmax=img[0].max())
        mapper = cm.ScalarMappable(norm=normalizer, cmap='jet')
        cimg = (mapper.to_rgba(img[0])[:, :, :3] * 255).astype(np.uint8)
        return cimg
    else:
        channel_max = np.max(img[0], axis=(1, 2), keepdims=True)
        channel_min = np.min(img[0], axis=(1, 2), keepdims=True)
        cimg = (img[0] - channel_min) / (channel_max - channel_min)
        return cimg.transpose(1, 2, 0)


class DictAverageMeter(object):
    def __init__(self, wsize=100):
        self.wsize = wsize
        self.sum_data = {}
        self.avg_data = {}
        self.count = {}
        return

    def update(self, new_input):
        if len(self.sum_data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.sum_data[k] = [v]
                self.avg_data[k] = v
                self.count[k] = 1
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                if np.isnan(v):
                    v = 0.0
                if v < 1e-8:
                    continue
                self.sum_data[k].append(self.sum_data[k][-1] + v)
                self.count[k] += 1
                self.avg_data[k] = self.sum_data[k][-1] / self.count[k]
                if self.count[k] > self.wsize:
                    del (self.sum_data[k][0])

    def wmean(self, key):
        if self.count[key] >= self.wsize:
            return (self.sum_data[key][-1] - self.sum_data[key][0]) / self.wsize
        else:
            return self.avg_data[key]

    def clear(self):
        self.sum_data = {}
        self.avg_data = {}
        self.count = {}



# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def D1_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

@make_nograd_func
@compute_metrics_for_each_image
def Thres_metric(D_est, D_gt, mask, thres):
    assert isinstance(thres, (int, float))
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())

# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def EPE_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    return F.l1_loss(D_est, D_gt, size_average=True)

import torch.distributed as dist
def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def reduce_scalar_outputs(scalar_outputs):
    world_size = get_world_size()
    if world_size < 2:
        return scalar_outputs
    with torch.no_grad():
        names = []
        scalars = []
        for k in sorted(scalar_outputs.keys()):
            names.append(k)
            scalars.append(scalar_outputs[k])
        scalars = torch.stack(scalars, dim=0)
        dist.reduce(scalars, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            scalars /= world_size
        reduced_scalars = {k: v for k, v in zip(names, scalars)}

    return reduced_scalars

def adjust_learning_rate(optimizer, epoch, base_lr, lrepochs):
    splits = lrepochs.split(':')
    assert len(splits) == 2

    # parse the epochs to downscale the learning rate (before :)
    downscale_epochs = [int(eid_str) for eid_str in splits[0].split(',')]
    # parse downscale rate (after :)
    downscale_rates = [float(eid_str) for eid_str in splits[1].split(',')]
    print("downscale epochs: {}, downscale rate: {}".format(downscale_epochs, downscale_rates))

    lr = base_lr
    for eid, downscale_rate in zip(downscale_epochs, downscale_rates):
        if epoch >= eid:
            lr /= downscale_rate
        else:
            break
    print("setting learning rate to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

