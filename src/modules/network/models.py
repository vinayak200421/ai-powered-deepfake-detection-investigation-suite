import torch.nn as nn
from .xception import xception

class TransferModel(nn.Module):
    def __init__(self, modelchoice='xception', num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        self.model = xception(num_classes=num_out_classes, pretrained=False)

    def forward(self, x):
        return self.model(x)
