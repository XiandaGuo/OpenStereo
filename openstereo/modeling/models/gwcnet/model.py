from modeling.base_model import BaseModel


class GwcNet(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # backbone modeling/backbone/gwcnet.py
    # cost processor modeling/cost_processor/gwcnet.py
    # disp processor modeling/disp_processor/gwcnet.py
