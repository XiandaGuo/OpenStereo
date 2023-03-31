from base_trainer import BaseTrainer
from modeling.base_model import BaseModel


class PSMNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Trainer = PSMNetTrainer

    def init_parameters(self):
        pass


class PSMNetTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return 'PSMNetTrainer'
