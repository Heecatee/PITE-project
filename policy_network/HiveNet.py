import torch
from torch import nn
from torch import functional as F
from hive_vision import HiveNetVision


class HiveNet(nn.Module):

    def __init__(self, frame_height, frame_width, kernel_size, stride, num_of_clusters):
        super(HiveNet, self).__init__()
        self.vision = HiveNetVision(frame_height, frame_width, kernel_size, stride, outputs=32)
        self.policy_hidden1 = nn.Linear(32+num_of_clusters, 64)
        self.policy_output = nn.Linear(64, 2*num_of_clusters)

        self.value_hidden1 = nn.Linear(32 + num_of_clusters, 64)
        self.value_output = nn.Linear(64, 2 * num_of_clusters)

    def forward(self, map_input, thresholds):
        x = self.vision(map_input)
        x = torch.cat((x, thresholds), dim=1)

        actor_x = F.relu(self.policy_hidden1(x))
        action_probabilities = F.relu(self.policy_output(actor_x))

        critic_x = F.relu(self.value_hidden1(x))
        values = F.relu(self.value_output(critic_x))
        return action_probabilities, values





