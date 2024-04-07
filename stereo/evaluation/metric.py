import torch


def d1_metric(disp_pred, disp_gt, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0).to(disp_pred.device)
    disp_pred, disp_gt = disp_pred[mask], disp_gt[mask]
    E = torch.abs(disp_gt - disp_pred)
    err_mask = (E > 3) & (E / torch.abs(disp_gt) > 0.05)
    return torch.mean(err_mask.float()) * 100


def threshold_metric(disp_pred, disp_gt, mask, threshold):
    if mask.sum() == 0:
        return torch.tensor(0.0).to(disp_pred.device)
    disp_pred, disp_gt = disp_pred[mask], disp_gt[mask]
    E = torch.abs(disp_gt - disp_pred)
    err_mask = E > threshold
    return torch.mean(err_mask.float()) * 100


def epe_metric(disp_pred, disp_gt, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0).to(disp_pred.device)
    disp_pred, disp_gt = disp_pred[mask], disp_gt[mask]
    E = torch.abs(disp_gt - disp_pred)
    return torch.mean(E)
