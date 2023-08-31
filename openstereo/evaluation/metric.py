import numpy as np
import torch


def d1_metric_np(disp_est, disp_gt, mask):
    """
    Compute the D1 metric for disparity estimation.
    The metric is defined as the percentage of stereo disparity outliers in the first frame.
    Outliers are defined as pixels with disparity error > 3 pixels and relative error > 0.05.
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
        return np.mean(0.0)

    # Apply the mask to the estimated and ground truth disparity maps
    disp_est, disp_gt = disp_est[mask], disp_gt[mask]

    # Calculate the absolute error between estimated and ground truth disparities
    E = np.abs(disp_gt - disp_est)

    # Create an error mask where the error is greater than 3 pixels and the relative error is greater than 0.05
    err_mask = (E > 3) & (E / np.abs(disp_gt) > 0.05)

    # Calculate the percentage of errors and return the result
    return np.mean(err_mask.astype(float)) * 100


def d1_metric(disp_est, disp_gt, mask):
    """
    Compute the D1 metric for disparity estimation.
    The metric is defined as the percentage of stereo disparity outliers in the first frame.
    Outliers are defined as pixels with disparity error > 3 pixels.
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
        return torch.tensor(0.0).to(disp_est.device)

    # Apply the mask to the estimated and ground truth disparity maps
    disp_est, disp_gt = disp_est[mask], disp_gt[mask]

    # Calculate the absolute error between estimated and ground truth disparities
    E = torch.abs(disp_gt - disp_est)

    # Create an error mask where the error is greater than 3 pixels and the relative error is greater than 0.05
    err_mask = (E > 3) & (E / torch.abs(disp_gt) > 0.05)

    # Calculate the percentage of errors and return the result
    return torch.mean(err_mask.float()) * 100


def bad_metric_np(disp_est, disp_gt, mask, threshold):
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
        return np.mean(0.0)

    # Apply the mask to the estimated and ground truth disparity maps
    disp_est, disp_gt = disp_est[mask], disp_gt[mask]

    # Calculate the absolute error between estimated and ground truth disparities
    E = np.abs(disp_gt - disp_est)

    # Create an error mask where the error is greater than the specified threshold
    err_mask = E > threshold

    # Calculate the percentage of errors and return the result
    return np.mean(err_mask.astype(float)) * 100


def bad_metric(disp_est, disp_gt, mask, threshold):
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
        return torch.tensor(0.0).to(disp_est.device)

    # Apply the mask to the estimated and ground truth disparity maps
    disp_est, disp_gt = disp_est[mask], disp_gt[mask]

    # Calculate the absolute error between estimated and ground truth disparities
    E = torch.abs(disp_gt - disp_est)

    # Create an error mask where the error is greater than the specified threshold
    err_mask = E > threshold

    # Calculate the percentage of errors and return the result
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
        return np.mean(0.0)

    # Apply the mask to the estimated and ground truth disparity maps
    disp_est, disp_gt = disp_est[mask], disp_gt[mask]

    # Calculate the absolute error between estimated and ground truth disparities
    E = np.abs(disp_gt - disp_est)

    # Calculate the average error and return the result
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
        return torch.tensor(0.0).to(disp_est.device)

    # Apply the mask to the estimated and ground truth disparity maps
    disp_est, disp_gt = disp_est[mask], disp_gt[mask]

    # Calculate the absolute error between estimated and ground truth disparities
    E = torch.abs(disp_gt - disp_est)

    # Calculate the average error and return the result
    return torch.mean(E)
