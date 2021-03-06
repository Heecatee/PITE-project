import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class HiveNetVision(nn.Module):

    def __init__(self, kernel_size, stride, outputs,
                 hidden_layer_dims=(16, 32),
                 frames_per_input=3,
                 image_compressed_size=(90, 60)):

        super(HiveNetVision, self).__init__()

        self.process_image_input = T.Compose([T.Grayscale(),
                                              T.Resize(
                                                  image_compressed_size, interpolation=Image.CUBIC),
                                              T.ToTensor()])
        self.map_history = None
        self.frames_per_input = frames_per_input
        self.map_history_shape = (
            1, self.frames_per_input, image_compressed_size[0], image_compressed_size[1])

        hidden_layer1_size, hidden_layer2_size = hidden_layer_dims

        self.conv1 = nn.Conv2d(in_channels=self.frames_per_input, out_channels=hidden_layer1_size,
                               kernel_size=kernel_size, stride=stride).to(device)
        self.bn1 = nn.BatchNorm2d(hidden_layer1_size).to(device)
        self.conv2 = nn.Conv2d(in_channels=hidden_layer1_size, out_channels=hidden_layer2_size,
                               kernel_size=kernel_size, stride=stride).to(device)
        self.bn2 = nn.BatchNorm2d(hidden_layer2_size).to(device)

        def conv2d_size(size, kernel_size=kernel_size, stride=stride):
            return (size - kernel_size) // stride + 1

        conv_width = conv2d_size(conv2d_size(image_compressed_size[0]))
        conv_height = conv2d_size(conv2d_size(image_compressed_size[1]))
        linear_input_size = conv_width * conv_height * hidden_layer2_size
        self.output = nn.Linear(linear_input_size, outputs).to(device)

    def forward(self, map_image):
        map_image = Image.frombytes(
            mode='RGB', size=(1280, 540), data=map_image)
        x = self.process_image_input(map_image).to(device)

        if self.map_history is None:
            self.map_history = []
            for _ in range(self.frames_per_input):
                self.map_history.append(x)
        else:
            self.map_history.pop(0)
            self.map_history.append(x)

        x = torch.stack(self.map_history, dim=1).to(device)
        x = F.relu(self.bn1(self.conv1(x))).to(device)
        x = F.relu(self.bn2(self.conv2(x))).to(device)
        x = torch.flatten(x).to(device)
        x = self.output(x).to(device)
        return x
