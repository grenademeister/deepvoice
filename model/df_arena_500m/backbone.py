import torch
import torch.nn as nn
from pathlib import Path

from transformers import Wav2Vec2Config, Wav2Vec2Model

from .conformer import FinalConformer


class DF_Arena_500M(nn.Module):
    def __init__(self):
        super().__init__()
        config = Path(__file__).parent / "facebook" / "wav2vec2-xls-r-300m"
        self.ssl_model = Wav2Vec2Model(Wav2Vec2Config.from_pretrained(config, local_files_only=True))
        self.ssl_model.config.output_hidden_states = True
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.conformer = FinalConformer(
            emb_size=1024, heads=4, ffmult=4, exp_fac=2, kernel_size=31, n_encoders=4
        )
        self.attn_scores = nn.Linear(1024, 1, bias=False)

    def get_attenF1Dpooling(self, x):
        logits = self.attn_scores(x)
        weights = torch.softmax(logits, dim=1)
        return torch.sum(weights * x, dim=1, keepdim=True)

    def get_attenF1D(self, layerResult):
        poollayerResult = []
        fullf = []
        for layer in layerResult:
            layery = self.get_attenF1Dpooling(layer)
            poollayerResult.append(layery)
            fullf.append(layer.unsqueeze(1))
        layery = torch.cat(poollayerResult, dim=1)
        fullfeature = torch.cat(fullf, dim=1)
        return layery, fullfeature

    def forward(self, x):
        out_ssl = self.ssl_model(x.unsqueeze(0) if x.dim() == 1 else x)
        y0, fullfeature = self.get_attenF1D(out_ssl.hidden_states)
        y0 = self.fc0(y0)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)
        fullfeature = fullfeature.unsqueeze(dim=1)
        fullfeature = self.first_bn(fullfeature)
        fullfeature = self.selu(fullfeature)
        output, _ = self.conformer(fullfeature.squeeze(1))
        return output
