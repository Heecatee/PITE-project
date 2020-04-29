import torch
from torch import nn
from torch import functional as F


class HiveNetVision(nn.Module):
    def __init__(self, frame_height, frame_width, kernel_size, stride,  outputs):
        super(HiveNetVision, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=kernel_size, stride=stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        conv_w = conv2d_size_out(conv2d_size_out(frame_width))
        conv_h = conv2d_size_out(conv2d_size_out(frame_height))
        linear_input_size = conv_w * conv_h * 16
        self.output = nn.Linear(linear_input_size, outputs)

    def forward(self, map_input):
        x = F.relu(self.bn1(self.conv1(map_input)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_flatten = torch.flatten(x)
        x = self.output(x_flatten)
        return x
