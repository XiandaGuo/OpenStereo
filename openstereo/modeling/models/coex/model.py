from modeling.base_model import BaseModel


class CoExNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # backbone modeling/backbone/CoEx.py
    # cost processor modeling/cost_processor/CoEx.py
    # disp processor modeling/disp_processor/CoEx.py
