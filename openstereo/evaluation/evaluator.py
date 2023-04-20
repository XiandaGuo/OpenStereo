from functools import partial

from evaluation.metric import *

METRICS_NP = {
    # EPE metric (Average Endpoint Error)
    'epe': epe_metric_np,
    # D1 metric (Percentage of erroneous pixels with disparity error > 3 pixels and relative error > 0.05)
    'd1_all': d1_metric_np,
    # Threshold metrics (Percentage of erroneous pixels with disparity error > threshold)
    'thres_1': partial(threshold_metric_np, threshold=1),
    'thres_2': partial(threshold_metric_np, threshold=2),
    'thres_3': partial(threshold_metric_np, threshold=3),

}

METRICS = {
    # EPE metric (Average Endpoint Error)
    'epe': epe_metric,
    # D1 metric (Percentage of erroneous pixels with disparity error > 3 pixels and relative error > 0.05)
    'd1_all': d1_metric,
    # Threshold metrics (Percentage of erroneous pixels with disparity error > threshold)
    'thres_1': partial(threshold_metric, threshold=1),
    'thres_2': partial(threshold_metric, threshold=2),
    'thres_3': partial(threshold_metric, threshold=3),
}


class OpenStereoEvaluator:
    def __init__(self, metrics=None, use_np=False):
        # Set default metrics if none are given
        if metrics is None:
            metrics = ['epe', 'd1_all']
        self.metrics = metrics
        self.use_np = use_np

    def __call__(self, data):
        # Extract input data
        disp_est = data['disp_est']
        disp_gt = data['disp_gt']
        mask = data['mask']
        res = {}

        # Loop through the specified metrics and compute results
        for m in self.metrics:
            # Check if the metric is valid
            if m not in METRICS:
                raise ValueError("Unknown metric: {}".format(m))
            else:
                # Get the appropriate metric function based on use_np
                metric_func = METRICS[m] if not self.use_np else METRICS_NP[m]

                # Compute the metric and store the result in the dictionary
                res[m] = metric_func(disp_est, disp_gt, mask)
        return res
