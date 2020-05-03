import torch
from torch import nn
from torch import functional as F
from hive_vision import HiveNetVision


class HiveNet(nn.Module):

    def __init__(self, frame_height, frame_width, kernel_size, stride, num_of_clusters):
        super(HiveNet, self).__init__()
        self.vision = HiveNetVision(frame_height, frame_width, kernel_size, stride, outputs=32)
        self.hidden1 = nn.Linear(32+num_of_clusters, 64)
        self.output = nn.Linear(64, 2*num_of_clusters)

    def forward(self, map_input, thresholds):
        x = self.vision(map_input)
        x = torch.cat((x, thresholds), dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.output(x))
        return x





