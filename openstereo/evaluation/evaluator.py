from functools import partial

from evaluation.metric import *

METRICS = {
    # EPE metric (Average Endpoint Error)
    'epe': epe_metric_per_image,
    # D1 metric (Percentage of erroneous pixels with disparity error > 3 pixels and relative error > 0.05)
    'd1_all': d1_metric_per_image,
    # Threshold metrics (Percentage of erroneous pixels with disparity error > threshold)
    'bad_1': partial(bad_metric_per_image, threshold=1),
    'bad_2': partial(bad_metric_per_image, threshold=2),
    'bad_3': partial(bad_metric_per_image, threshold=3),
}


class OpenStereoEvaluator:
    def __init__(self, metrics=None):
        # Set default metrics if none are given
        if metrics is None:
            metrics = ['epe', 'd1_all']
        self.metrics = metrics

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
                metric_func = METRICS[m]

                # Compute the metric and store the result in the dictionary
                res[m] = metric_func(disp_est, disp_gt, mask)
        return res
