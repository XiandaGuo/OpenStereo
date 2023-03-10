import torch


class ClipGrad:
    def __init__(self, clip_type="None", clip_value=0.1, max_norm=35, norm_type=2):
        self.clip_type = clip_type
        self.clip_value = clip_value
        self.max_norm = max_norm
        self.norm_type = norm_type

    def __call__(self, model):
        if self.clip_type == 'value':
            torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
        elif self.clip_type == 'norm':
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm, self.norm_type)
        else:
            raise ValueError(f"Unknown clip type {self.clip_type}.")
