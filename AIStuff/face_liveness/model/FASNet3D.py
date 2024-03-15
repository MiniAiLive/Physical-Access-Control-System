import torch
import torch.nn.functional as F
from torch import nn
from .FASNetB import FASNetB


class MNet(nn.Module):
    """ define the model called MNet """
    def __init__(self, num_classes=2):
        super(MNet, self).__init__()
        self.fasnetb = FASNetB()
        self.fasnetb.class_ret = nn.Sequential(nn.Linear(self.fasnetb.class_ret.in_features, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        """ forward model with 3d data """
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.fasnetb(x_3d[:, t, :, :, :])
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x
