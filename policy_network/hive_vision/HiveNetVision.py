import torch
from torch import nn
from torch import functional as F
import torchvision.transforms as T
from PIL import Image


class HiveNetVision(nn.Module):

    def __init__(self, frame_height, frame_width, kernel_size, stride, outputs, starting_frame):
        super(HiveNetVision, self).__init__()
        self.process_image_input = T.Compose([T.Grayscale(),
                                              T.Resize(40, interpolation=Image.CUBIC),
                                              T.ToTensor()])
        self.map_history = torch.Tensor([starting_frame, starting_frame, starting_frame])
        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=kernel_size, stride=stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(frame_width))
        convh = conv2d_size_out(conv2d_size_out(frame_height))
        linear_input_size = convw * convh * 16
        self.output = nn.Linear(linear_input_size, outputs)

    def forward(self, map_image):
        x = self.process_image_input(map_image)
        self.map_history = torch.cat((self.map_history[1:], x), dim=0)
        x = self.map_history
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x_flatten = torch.flatten(x)
        x = self.output(x_flatten)
        return x

