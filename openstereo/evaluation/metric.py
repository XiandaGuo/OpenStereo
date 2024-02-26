import torch


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


def d1_metric_per_image(disp_est, disp_gt, mask):
    """
    Compute the D1 metric for disparity estimation for each image in a batch.
    The metric is defined as the percentage of stereo disparity outliers in the first frame.
    Outliers are defined as pixels with disparity error > 3 pixels and relative error > 0.05.
    Args:
        disp_est: estimated disparity map
        disp_gt: ground truth disparity map
        mask: mask of valid pixels
    Returns:
        Tensor: D1 metric value for each image
    """
    # Calculate the absolute error between estimated and ground truth disparities
    E = torch.abs(disp_gt - disp_est)

    # Create an error mask where the error is greater than 3 pixels and the relative error is greater than 0.05
    err_mask = (E > 3) & (E / torch.abs(disp_gt) > 0.05)

    # Apply the mask to the error mask
    err_mask = err_mask & mask

    # Calculate the number of errors and the number of valid pixels for each image
    num_errors = err_mask.sum(dim=[1, 2])
    num_valid_pixels = mask.sum(dim=[1, 2])

    # Calculate the percentage of errors for each image
    d1_per_image = num_errors.float() / num_valid_pixels.float() * 100

    # Handle the case where the number of valid pixels is 0
    d1_per_image = torch.where(num_valid_pixels > 0, d1_per_image, torch.zeros_like(d1_per_image))

    return d1_per_image


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


def bad_metric_per_image(disp_est, disp_gt, mask, threshold):
    """
    Compute the threshold metric for disparity estimation for each image in a batch.
    The metric is defined as:
        Percentage of erroneous pixels in specified error threshold.
    Args:
        disp_est: estimated disparity map
        disp_gt: ground truth disparity map
        mask: mask of valid pixels and areas, eg: all pixels or non-occluded areas
        threshold: error threshold in pixels
    Returns:
        Tensor: threshold metric value for each image
    """
    # Calculate the absolute error between estimated and ground truth disparities
    E = torch.abs(disp_gt - disp_est)

    # Create an error mask where the error is greater than the specified threshold
    err_mask = E > threshold

    # Apply the mask to the error mask
    err_mask = err_mask & mask

    # Calculate the number of errors and the number of valid pixels for each image
    num_errors = err_mask.sum(dim=[1, 2])
    num_valid_pixels = mask.sum(dim=[1, 2])

    # Calculate the percentage of errors for each image
    bad_per_image = num_errors.float() / num_valid_pixels.float() * 100

    # Handle the case where the number of valid pixels is 0
    bad_per_image = torch.where(num_valid_pixels > 0, bad_per_image, torch.zeros_like(bad_per_image))

    return bad_per_image


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


def epe_metric_per_image(disp_est, disp_gt, mask):
    """
    Compute the EPE metric for disparity estimation for each image in a batch.
    Also known as the average error metric or L1 error.
    Args:
        disp_est: estimated disparity map
        disp_gt: ground truth disparity map
        mask: mask of valid pixels
    Returns:
        Tensor: EPE metric value for each image
    """
    # Calculate the absolute error between estimated and ground truth disparities
    E = torch.abs(disp_gt - disp_est)

    # Apply the mask to the error map
    E_masked = torch.where(mask, E, torch.zeros_like(E))

    # Calculate the sum of the errors and the number of valid pixels for each image
    E_sum = E_masked.sum(dim=[1, 2])
    num_valid_pixels = mask.sum(dim=[1, 2])

    # Calculate the average error for each image
    epe_per_image = E_sum / num_valid_pixels

    # Handle the case where the number of valid pixels is 0
    epe_per_image = torch.where(num_valid_pixels > 0, epe_per_image, torch.zeros_like(epe_per_image))

    return epe_per_image
