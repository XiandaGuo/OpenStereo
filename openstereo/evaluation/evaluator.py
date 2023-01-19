from functools import partial

import numpy as np


def d1_metric(disp_est, disp_gt, mask):
    """
    Compute the D1 metric for disparity estimation.
    The metric is defined as:
        Percentage of stereo disparity outliers in first frame.
        Outliers are defined as pixels with disparity error > 3 pixels.
    Args:
        disp_est: estimated disparity map
        disp_gt: ground truth disparity map
        mask: mask of valid pixels and areas, eg: bg, fg and all areas
    Returns:
        float: D1 metric value
    """
    disp_est, disp_gt = disp_est[mask], disp_gt[mask]
    E = np.abs(disp_gt - disp_est)
    err_mask = (E > 3) & (E / np.abs(disp_gt) > 0.05)
    return np.mean(err_mask.astype(float)) * 100


def threshold_metric(disp_est, disp_gt, mask, threshold):
    """
    Compute the threshold metric for disparity estimation.
    The metric is defined as:
        Percentage of erroneous pixels in specified error threshold.
    Args:
        disp_est: estimated disparity map
        disp_gt: ground truth disparity map
        mask: mask of valid pixels and areas, eg: all pixels or non-occluded areas
        threshold: error threshold in pixels
    Returns:
        float: threshold metric value
    """
    disp_est, disp_gt = disp_est[mask], disp_gt[mask]
    E = np.abs(disp_gt - disp_est)
    err_mask = E > threshold
    return np.mean(err_mask.astype(float)) * 100


def epe_metric(disp_est, disp_gt, mask):
    """
    Compute the EPE metric for disparity estimation.
    Also known as the average error metric or L1 error.
    Args:
        disp_est: estimated disparity map
        disp_gt: ground truth disparity map
        mask: mask of valid pixels
    Returns:
        float: EPE metric value
    """
    disp_est, disp_gt = disp_est[mask], disp_gt[mask]
    E = np.abs(disp_gt - disp_est)
    return np.mean(E)


def evaluate_kitti_2012(data, conf=None):
    """compute the error metrics for KITTI 2012 dataset"""

    threshold = conf['threshold'] if conf is not None else 3

    disp_est = data['disp_est']
    disp_gt_occ = data['disp_gt_occ']
    disp_gt_noc = data['disp_gt_noc']

    out_noc = threshold_metric(disp_est, disp_gt_noc, disp_gt_noc > 0, threshold)
    avg_noc = epe_metric(disp_est, disp_gt_noc, disp_gt_noc > 0)

    out_occ = threshold_metric(disp_est, disp_gt_occ, disp_gt_occ > 0, threshold)
    avg_occ = epe_metric(disp_est, disp_gt_occ, disp_gt_occ > 0)

    metric = {
        'out_noc': out_noc,
        'avg_noc': avg_noc,
        'out_occ': out_occ,
        'avg_occ': avg_occ,
    }

    return metric


def evaluate_kitti_2015(data, conf=None):
    """compute the error metrics for KITTI 2015 dataset"""

    disp_est = data['disp_est']
    disp_gt_occ = data['disp_gt_occ']
    d1_all = d1_metric(disp_est, disp_gt_occ, disp_gt_occ > 0)

    # Todo：add the d1_bg, d1_fg metrics

    metric = {
        'd1_all': d1_all,
        # 'd1_fg': d1_fg,
        # 'd1_bg': d1_bg,
    }
    return metric


def evaluate_sceneflow(data, conf=None):
    """compute the error metrics for SceneFlow dataset"""
    disp_est = data['disp_est']
    disp_gt = data['disp_gt']
    epe = epe_metric(disp_est, disp_gt, disp_gt > 0)
    metric = {
        'epe': epe,
    }
    return metric


def evaluate_openstereo(data, conf=None):
    """compute the error metrics for OpenStereo"""
    disp_est = data['disp_est']
    disp_gt = data['disp_gt']

    epe = epe_metric(disp_est, disp_gt, disp_gt > 0)
    d1_all = d1_metric(disp_est, disp_gt, disp_gt > 0)

    metric = {
        'epe': epe,
        'd1_all': d1_all,
    }
    return metric


METRICS = {
    'epe': epe_metric,
    'd1_all': d1_metric,
    'thres_1': partial(threshold_metric, threshold=1),
    'thres_2': partial(threshold_metric, threshold=2),
    'thres_3': partial(threshold_metric, threshold=3),
    'kitti_2012': evaluate_kitti_2012,
    'kitti_2015': evaluate_kitti_2015,
    'sceneflow': evaluate_sceneflow,
}


def OpenStereoEvaluator(data, metric=None):
    """compute the error metrics for SceneFlow dataset"""
    if metric is None:
        metric = ['epe', 'd1_all']
    disp_est = data['disp_est']
    disp_gt = data['disp']
    res = {}
    for m in metric:
        if m not in METRICS:
            raise ValueError("Unknown metric: {}".format(m))
        else:
            metric_func = METRICS[m]
            res[m] = metric_func(disp_est, disp_gt, disp_gt > 0)
    print(res)
    return res
