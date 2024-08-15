# @Time    : 2023/8/28 22:28
# @Author  : zhangchenming
import os
import yaml
import random
import shutil
import numpy as np
import torch
import logging
import inspect
from easydict import EasyDict
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt
import cv2


def config_loader(path):
    with open(path, 'r') as stream:
        src_cfgs = yaml.safe_load(stream)
    return src_cfgs


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_valid_args(obj, input_args, free_keys=None):
    if free_keys is None:
        free_keys = []
    if inspect.isfunction(obj):
        expected_keys = inspect.getfullargspec(obj)[0]
    elif inspect.isclass(obj):
        expected_keys = inspect.getfullargspec(obj.__init__)[0]
    else:
        raise ValueError('Just support function and class object!')
    unexpect_keys = list()
    expected_args = {}
    for k, v in input_args.items():
        k = k.lower()
        if k in expected_keys:
            expected_args[k] = v
        elif k in free_keys:
            pass
        else:
            unexpect_keys.append(k)
    if unexpect_keys:
        print("Find Unexpected Args(%s) in the Configuration of - %s -" % (', '.join(unexpect_keys), obj.__name__))
    return expected_args


def backup_source_code(backup_dir):
    # 子文件夹下的同名也会被忽略
    ignore_hidden = shutil.ignore_patterns(
        ".idea", ".git*", "*pycache*",
        "cfgs", "data", "output")

    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

    shutil.copytree('.', backup_dir, ignore=ignore_hidden)
    # os.system("chmod -R g+w {}".format(backup_dir))


def log_configs(cfgs, pre='cfgs', logger=None):
    for key, val in cfgs.items():
        if isinstance(cfgs[key], EasyDict):
            logger.info('----------- %s -----------' % key)
            log_configs(cfgs[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def save_checkpoint(model, optimizer, scheduler, scaler, is_dist, epoch, filename='checkpoint'):
    if is_dist:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    optim_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    scaler_state = scaler.state_dict()

    state = {'epoch': epoch,
             'model_state': model_state,
             'optimizer_state': optim_state,
             'scheduler_state': scheduler_state,
             'scaler_state': scaler_state}
    torch.save(state, filename)


def freeze_bn(module):
    """Freeze the batch normalization layers."""
    for m in module.modules():
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    return module


# def convert_state_dict(ori_state_dict, is_dist=True):
#     new_state_dict = OrderedDict()
#     if is_dist:
#         if not next(iter(ori_state_dict)).startswith('module'):
#             for k, v in ori_state_dict.items():
#                 new_state_dict[f'module.{k}'] = v
#         else:
#             new_state_dict = ori_state_dict
#     else:
#         if not next(iter(ori_state_dict)).startswith('module'):
#             new_state_dict = ori_state_dict
#         else:
#             for k, v in ori_state_dict.items():
#                 k = k.replace('module.', '')
#                 new_state_dict[k] = v
#
#     return new_state_dict


def load_params_from_file(model, filename, device, dist_mode, logger, strict=True):
    checkpoint = torch.load(filename, map_location=device)
    pretrained_state_dict = checkpoint['model_state']
    tmp_model = model.module if dist_mode else model
    state_dict = tmp_model.state_dict()

    unused_state_dict = {}
    update_state_dict = {}
    unupdate_state_dict = {}
    for key, val in pretrained_state_dict.items():
        if key in state_dict and state_dict[key].shape == val.shape:
            update_state_dict[key] = val
        else:
            unused_state_dict[key] = val
    for key in state_dict:
        if key not in update_state_dict:
            unupdate_state_dict[key] = state_dict[key]

    if strict:
        tmp_model.load_state_dict(update_state_dict)
    else:
        state_dict.update(update_state_dict)
        tmp_model.load_state_dict(state_dict)

    message = 'Unused weight: '
    for key, val in unused_state_dict.items():
        message += str(key) + ':' + str(val.shape) + ', '
    if logger:
        logger.info(message)
    else:
        print(message)

    message = 'Not updated weight: '
    for key, val in unupdate_state_dict.items():
        message += str(key) + ':' + str(val.shape) + ', '
    if logger:
        logger.info(message)
    else:
        print(message)


def color_map_tensorboard(disp_gt, pred, disp_max=192):
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


def write_tensorboard(tb_writer, tb_info, step):
    for k, v in tb_info.items():
        module_name = k.split('/')[0]
        writer_module = getattr(tb_writer, 'add_' + module_name)
        board_name = k.replace(module_name + "/", '')
        v = v.detach() if torch.is_tensor(v) else v
        if module_name == 'image' and v.dim() == 2:
            writer_module(board_name, v, step, dataformats='HW')
        else:
            writer_module(board_name, v, step)


def draw_depth_2_image(disp, image, baseline=0.54, focallength=1.003556e+3):
    # baseline 单位米
    disp = disp.detach().data.cpu().numpy()  # [h, w]
    image = image.detach().data.cpu().numpy()  # [3, h, w]
    image = np.transpose(image, [1, 2, 0])  # [h, w, 3]
    image = np.ascontiguousarray(image, dtype=np.uint8)

    point_size = 2
    point_colors = [(255, 255, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    thickness = -1

    depths = baseline * focallength / disp
    depths[np.isinf(depths)] = 0.0
    depths[np.isnan(depths)] = 0.

    print('depth', depths.shape)
    print('image', image.shape)
    assert depths.shape[:2] == image.shape[:2]

    for i in range(depths.shape[0]):
        for j in range(depths.shape[1]):
            if depths[i, j] > 40:
                point_color = point_colors[-1]
            elif depths[i, j] > 20:
                point_color = point_colors[-2]
            elif depths[i, j] > 15:
                point_color = point_colors[-3]
            elif depths[i, j] > 8:
                point_color = point_colors[-4]
            elif depths[i, j] > 0.5:
                point_color = point_colors[-5]
            else:
                continue
            cv2.circle(image, (j, i), point_size, point_color, thickness)

    image = np.transpose(image, [2, 0, 1])
    image = np.ascontiguousarray(image, dtype=np.float32)
    return torch.from_numpy(image)

def get_pos_fullres(fx, w, h):
    x_range = (np.linspace(0, w - 1, w) + 0.5 - w // 2) / fx
    y_range = (np.linspace(0, h - 1, h) + 0.5 - h // 2) / fx
    x, y = np.meshgrid(x_range, y_range)
    z = np.ones_like(x)
    pos_grid = np.stack([x, y, z], axis=0).astype(np.float32)
    return pos_grid