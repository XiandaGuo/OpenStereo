from .base import LinearWarmup, ExponentialWarmup


class UntunedLinearWarmup(LinearWarmup):
    """Untuned linear warmup schedule for Adam.

    This warmup scheme is described in
    `On the adequacy of untuned warmup for adaptive optimization
    <https://arxiv.org/abs/1910.04209>`_.

    Arguments:
        optimizer (Optimizer): an Adam optimizer
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, last_step=-1):
        def warmup_period_fn(beta2):
            return int(2.0 / (1.0-beta2))
        warmup_period = [warmup_period_fn(x['betas'][1]) for x in optimizer.param_groups]
        super(UntunedLinearWarmup, self).__init__(optimizer, warmup_period, last_step)


class UntunedExponentialWarmup(ExponentialWarmup):
    """Untuned exponetial warmup schedule for Adam.

    This warmup scheme is described in
    `On the adequacy of untuned warmup for adaptive optimization
    <https://arxiv.org/abs/1910.04209>`_.

    Arguments:
        optimizer (Optimizer): an Adam optimizer
        last_step (int): The index of last step. (Default: -1)
    """

    def __init__(self, optimizer, last_step=-1):
        def warmup_period_fn(beta2):
            return int(1.0 / (1.0-beta2))
        warmup_period = [warmup_period_fn(x['betas'][1]) for x in optimizer.param_groups]
        super(UntunedExponentialWarmup, self).__init__(optimizer, warmup_period, last_step)
