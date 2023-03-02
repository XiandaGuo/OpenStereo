import torch.nn.functional as F
import torch


def stereo_psmnet_loss(inputs, target, mask, dlossw=None):
    total_loss = torch.tensor(0.0, dtype=target.dtype, device=target.device, requires_grad=False)
    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        disp0, disp1, disp2, disp3 = stage_inputs["pred0"], stage_inputs["pred1"], stage_inputs["pred2"], stage_inputs[
            "pred3"]

        loss = 0.5 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + \
               0.5 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + \
               0.7 * F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean') + \
               1.0 * F.smooth_l1_loss(disp3[mask], target[mask], reduction='mean')

        if dlossw is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += dlossw[stage_idx] * loss
        else:
            total_loss += loss

    return total_loss
