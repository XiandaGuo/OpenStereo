from base_trainer import BaseTrainer
from modeling.base_model import BaseModel


class PSMNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        To use your own trainer, you need to set self.Trainer to your trainer class.
        """
        self.Trainer = PSMNetTrainer

    def init_parameters(self):
        pass

    # backbone modeling/backbone/PSMNet.py
    # cost processor modeling/cost_processor/PSMNet.py
    # disp processor modeling/disp_processor/PSMNet.py


class PSMNetTrainer(BaseTrainer):
    """
    You can define your own trainer class here.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return 'PSMNetTrainer'
