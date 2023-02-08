from modeling.base_model import BaseModel
from .DSMNet import DSMNet as Net1x
from .DSMNet2x2 import DSMNet as Net2x


class DSMNet1x(BaseModel):
    def __init__(self, *args, **kwargs):
        self.maxdisp = 192
        super().__init__(*args, **kwargs)

    def build_network(self, model_cfg):
        """Build the network."""
        self.net = Net1x(
            maxdisp=self.maxdisp,
        )

    def forward(self, inputs):
        """Forward the network."""
        ref_img = inputs["ref_img"]
        tgt_img = inputs["tgt_img"]
        res = self.net(ref_img, tgt_img)

        if self.training:
            [disp0, disp1, disp2] = res
            output = {
                "training_disp": {
                    "disp0": {
                        "disp_ests": disp0,
                        "disp_gt": inputs['disp_gt'],
                        "mask": inputs['mask']
                    },
                    "disp1": {
                        "disp_ests": disp1,
                        "disp_gt": inputs['disp_gt'],
                        "mask": inputs['mask']
                    },
                    "disp2": {
                        "disp_ests": disp2,
                        "disp_gt": inputs['disp_gt'],
                        "mask": inputs['mask']
                    },
                },
                "visual_summary": {}
            }
        else:
            output = {
                "inference_disp": {
                    "disp_est": res
                },
                "visual_summary": {}
            }

        return output


class DSMNet2x(DSMNet1x):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_network(self, model_cfg):
        """Build the network."""
        self.net = Net2x(
            maxdisp=self.maxdisp,
        )
