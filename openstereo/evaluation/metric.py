import numpy as np
import torch


def d1_metric_np(disp_est, disp_gt, mask):
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
    if mask.sum() == 0:
        return 0
    disp_est, disp_gt = disp_est[mask], disp_gt[mask]
    E = np.abs(disp_gt - disp_est)
    err_mask = (E > 3) & (E / np.abs(disp_gt) > 0.05)
    return np.mean(err_mask.astype(float)) * 100


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
    if mask.sum() == 0:
        return 0
    disp_est, disp_gt = disp_est[mask], disp_gt[mask]
    E = torch.abs(disp_gt - disp_est)
    err_mask = (E > 3) & (E / torch.abs(disp_gt) > 0.05)
    return torch.mean(err_mask.float()) * 100


def threshold_metric_np(disp_est, disp_gt, mask, threshold):
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
    if mask.sum() == 0:
        return 0
    disp_est, disp_gt = disp_est[mask], disp_gt[mask]
    E = np.abs(disp_gt - disp_est)
    err_mask = E > threshold
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
    if mask.sum() == 0:
        return 0
    disp_est, disp_gt = disp_est[mask], disp_gt[mask]
    E = torch.abs(disp_gt - disp_est)
    err_mask = E > threshold
    return torch.mean(err_mask.float()) * 100


def epe_metric_np(disp_est, disp_gt, mask):
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
    if mask.sum() == 0:
        return 0
    disp_est, disp_gt = disp_est[mask], disp_gt[mask]
    E = np.abs(disp_gt - disp_est)
    return np.mean(E)


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
    if mask.sum() == 0:
        return 0
    disp_est, disp_gt = disp_est[mask], disp_gt[mask]
    E = torch.abs(disp_gt - disp_est)
    return torch.mean(E)
