from transformers import PreTrainedModel

from .backbone import DF_Arena_500M
from .configuration_antispoofing import DF_Arena_500M_Config


class DF_Arena_500M_Antispoofing(PreTrainedModel):
    config_class = DF_Arena_500M_Config

    def __init__(self, config: DF_Arena_500M_Config):
        super().__init__(config)
        self.backbone = DF_Arena_500M()
        self.post_init()

    def forward(self, input_values, attention_mask=None):
        return {"logits": self.backbone(input_values)}
