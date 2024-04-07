import torch


def d1_metric(disp_pred, disp_gt, mask):
    E = torch.abs(disp_gt - disp_pred)
    err_mask = (E > 3) & (E / torch.abs(disp_gt) > 0.05)

    err_mask = err_mask & mask
    num_errors = err_mask.sum(dim=[1, 2])
    num_valid_pixels = mask.sum(dim=[1, 2])

    d1_per_image = num_errors.float() / num_valid_pixels.float() * 100
    d1_per_image = torch.where(num_valid_pixels > 0, d1_per_image, torch.zeros_like(d1_per_image))

    return d1_per_image


def threshold_metric(disp_pred, disp_gt, mask, threshold):
    E = torch.abs(disp_gt - disp_pred)
    err_mask = E > threshold

    err_mask = err_mask & mask
    num_errors = err_mask.sum(dim=[1, 2])
    num_valid_pixels = mask.sum(dim=[1, 2])

    bad_per_image = num_errors.float() / num_valid_pixels.float() * 100
    bad_per_image = torch.where(num_valid_pixels > 0, bad_per_image, torch.zeros_like(bad_per_image))

    return bad_per_image


def epe_metric(disp_pred, disp_gt, mask):
    E = torch.abs(disp_gt - disp_pred)
    E_masked = torch.where(mask, E, torch.zeros_like(E))

    E_sum = E_masked.sum(dim=[1, 2])
    num_valid_pixels = mask.sum(dim=[1, 2])
    epe_per_image = E_sum / num_valid_pixels
    epe_per_image = torch.where(num_valid_pixels > 0, epe_per_image, torch.zeros_like(epe_per_image))

    return epe_per_image
