import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
from PIL import Image


class HiveNetVision(nn.Module):

    def __init__(self, kernel_size, stride, outputs, starting_frame,
                 hidden_layer_dims=(16, 32),
                 frames_per_input=3,
                 image_compressed_size=60):
        super(HiveNetVision, self).__init__()
        self.process_image_input = T.Compose([T.Grayscale(),
                                              T.Resize(image_compressed_size, interpolation=Image.CUBIC)])
        self.map_history = [starting_frame, starting_frame, starting_frame]
        hidden_layer1_size, hidden_layer2_size = hidden_layer_dims
        self.conv1 = nn.Conv2d(in_channels=frames_per_input, out_channels=hidden_layer1_size,
                               kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(hidden_layer1_size)
        self.conv2 = nn.Conv2d(in_channels=hidden_layer1_size, out_channels=hidden_layer2_size,
                               kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(hidden_layer2_size)

        def conv2d_size(size, kernel_size=kernel_size, stride=stride):
            return (size - kernel_size) // stride + 1

        conv_width = conv2d_size(conv2d_size(image_compressed_size))
        conv_height = conv2d_size(conv2d_size(image_compressed_size))
        linear_input_size = conv_width * conv_height * hidden_layer2_size
        self.output = nn.Linear(linear_input_size, outputs)

    def forward(self, map_image):
        x = self.process_image_input(map_image)
        self.map_history.pop(0)
        self.map_history.append(x)
        x = torch.Tensor(self.map_history)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x)
        x = self.output(x)
        return x

