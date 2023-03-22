from functools import partial

from evaluation.metric import *


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


METRICS_NP = {
    'epe': epe_metric_np,
    'd1_all': d1_metric_np,
    'thres_1': partial(threshold_metric_np, threshold=1),
    'thres_2': partial(threshold_metric_np, threshold=2),
    'thres_3': partial(threshold_metric_np, threshold=3),
    # 'kitti_2012': evaluate_kitti_2012,
    # 'kitti_2015': evaluate_kitti_2015,
    # 'sceneflow': evaluate_sceneflow,
}

METRICS = {
    'epe': epe_metric,
    'd1_all': d1_metric,
    'thres_1': partial(threshold_metric, threshold=1),
    'thres_2': partial(threshold_metric, threshold=2),
    'thres_3': partial(threshold_metric, threshold=3),
    # 'kitti_2012': evaluate_kitti_2012,
    # 'kitti_2015': evaluate_kitti_2015,
    # 'sceneflow': evaluate_sceneflow,
}


def OpenStereoEvaluator(data, metric=None, use_np=False):
    """compute the error metrics for SceneFlow dataset"""
    if metric is None:
        metric = ['epe', 'd1_all']
    disp_est = data['disp_est']
    disp_gt = data['disp_gt']
    mask = data['mask']
    res = {}
    for m in metric:
        if m not in METRICS:
            raise ValueError("Unknown metric: {}".format(m))
        else:
            metric_func = METRICS[m] if not use_np else METRICS_NP[m]
            res[f"scalar/val/{m}"] = metric_func(disp_est, disp_gt, mask)
    return res


class OpenStereoEvaluator:
    def __init__(self, metrics=None, use_np=False):
        if metrics is None:
            metrics = ['epe', 'd1_all']
        self.metrics = metrics
        self.use_np = use_np

    def __call__(self, data):
        disp_est = data['disp_est']
        disp_gt = data['disp_gt']
        mask = data['mask']
        res = {}
        for m in self.metrics:
            if m not in METRICS:
                raise ValueError("Unknown metric: {}".format(m))
            else:
                metric_func = METRICS[m] if not self.use_np else METRICS_NP[m]
                res[m] = metric_func(disp_est, disp_gt, mask)
        return res
