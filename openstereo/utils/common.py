import copy
import inspect
import logging
import os
import random
from collections import OrderedDict, namedtuple

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP


class NoOp:
    """A no-op class for the case when we don't want to do anything."""

    def __call__(self, *args, **kwargs):
        pass

    def __getattr__(self, *args, **kwargs):
        def no_op(*args, **kwargs): pass

        return no_op

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass

    def __enter__(self):
        pass

    def dampening(self):
        return self


class Odict(OrderedDict):
    def append(self, odict):
        dst_keys = self.keys()
        for k, v in odict.items():
            if not is_list(v):
                v = [v]
            if k in dst_keys:
                if is_list(self[k]):
                    self[k] += v
                else:
                    self[k] = [self[k]] + v
            else:
                self[k] = v


def Ntuple(description, keys, values):
    if not is_list_or_tuple(keys):
        keys = [keys]
        values = [values]
    Tuple = namedtuple(description, keys)
    return Tuple._make(values)


def get_valid_args(obj, input_args, free_keys=[]):
    if inspect.isfunction(obj):
        expected_keys = inspect.getfullargspec(obj)[0]
    elif inspect.isclass(obj):
        expected_keys = inspect.getfullargspec(obj.__init__)[0]
    else:
        raise ValueError('Just support function and class object!')
    unexpect_keys = list()
    expected_args = {}
    for k, v in input_args.items():
        if k in expected_keys:
            expected_args[k] = v
        elif k in free_keys:
            pass
        else:
            unexpect_keys.append(k)
    if unexpect_keys != []:
        logging.info("Find Unexpected Args(%s) in the Configuration of - %s -" %
                     (', '.join(unexpect_keys), obj.__name__))
    return expected_args


def get_attr_from(sources, name):
    try:
        return getattr(sources[0], name)
    except:
        return get_attr_from(sources[1:], name) if len(sources) > 1 else getattr(sources[0], name)


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))


def is_bool(x):
    return isinstance(x, bool)


def is_str(x):
    return isinstance(x, str)


def is_list(x):
    return isinstance(x, list) or isinstance(x, nn.ModuleList)


def is_dict(x):
    return isinstance(x, dict) or isinstance(x, OrderedDict) or isinstance(x, Odict)


def is_tensor(x):
    return isinstance(x, torch.Tensor)


def is_array(x):
    return isinstance(x, np.ndarray)


def ts2np(x):
    return x.cpu().data.numpy()


def ts2var(x, **kwargs):
    return autograd.Variable(x, **kwargs).cuda()


def np2var(x, **kwargs):
    return ts2var(torch.from_numpy(x), **kwargs)


def list2var(x, **kwargs):
    return np2var(np.array(x), **kwargs)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def MergeCfgsDict(src, dst):
    for k, v in src.items():
        if (k not in dst.keys()) or (type(v) != type(dict())):
            dst[k] = v
        else:
            if is_dict(src[k]) and is_dict(dst[k]):
                MergeCfgsDict(src[k], dst[k])
            else:
                dst[k] = v


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    # with open("./configs/default.yaml", 'r') as stream:
    #     dst_cfgs = yaml.safe_load(stream)
    # MergeCfgsDict(src_cfgs, dst_cfgs)
    return src_cfgs


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True



def handler(signum, frame):
    logging.info('Ctrl+c/z pressed')
    os.system(
        "kill $(ps aux | grep main.py | grep -v grep | awk '{print $2}') ")
    logging.info('process group flush!')


def ddp_all_gather(features, dim=0, requires_grad=True):
    '''
        inputs: [n, ...]
    '''

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    feature_list = [torch.ones_like(features) for _ in range(world_size)]
    torch.distributed.all_gather(feature_list, features.contiguous())

    if requires_grad:
        feature_list[rank] = features
    feature = torch.cat(feature_list, dim=dim)
    return feature


# https://github.com/pytorch/pytorch/issues/16885
class DDPPassthrough(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def get_ddp_module(module, **kwargs):
    if len(list(module.parameters())) == 0:
        # for the case that loss module has not parameters.
        return module
    device = torch.cuda.current_device()
    module = DDPPassthrough(module, device_ids=[device], output_device=device, **kwargs)
    return module


def params_count(net):
    n_parameters = sum(p.numel() for p in net.parameters())
    return 'Parameters Count: {:.5f}M'.format(n_parameters / 1e6)


def convert_state_dict(ori_state_dict, is_dist=True):
    new_state_dict = OrderedDict()
    if is_dist:
        if not next(iter(ori_state_dict)).startswith('module'):
            for k, v in ori_state_dict.items():
                new_state_dict[f'module.{k}'] = v
        else:
            new_state_dict = ori_state_dict
    else:
        if not next(iter(ori_state_dict)).startswith('module'):
            new_state_dict = ori_state_dict
        else:
            for k, v in ori_state_dict.items():
                k = k[7:]
                new_state_dict[k] = v

    return new_state_dict


def color_map_tensorboard(disp_gt, pred, disp_max=192):
    import matplotlib.pyplot as plt
    import numpy as np
    cm = plt.get_cmap('plasma')

    disp_gt = disp_gt.detach().data.cpu().numpy()
    pred = pred.detach().data.cpu().numpy()
    error_map = np.abs(pred - disp_gt)

    disp_gt = np.clip(disp_gt, a_min=0, a_max=disp_max)
    pred = np.clip(pred, a_min=0, a_max=disp_max)

    gt_tmp = 255.0 * disp_gt / disp_max
    pred_tmp = 255.0 * pred / disp_max
    error_map_tmp = 255.0 * error_map / np.max(error_map)

    gt_tmp = cm(gt_tmp.astype('uint8'))
    pred_tmp = cm(pred_tmp.astype('uint8'))
    error_map_tmp = cm(error_map_tmp.astype('uint8'))

    gt_tmp = np.transpose(gt_tmp[:, :, :3], (2, 0, 1))
    pred_tmp = np.transpose(pred_tmp[:, :, :3], (2, 0, 1))
    error_map_tmp = np.transpose(error_map_tmp[:, :, :3], (2, 0, 1))

    color_disp_c = np.concatenate((gt_tmp, pred_tmp, error_map_tmp), axis=1)
    color_disp_c = torch.from_numpy(color_disp_c)

    return color_disp_c
