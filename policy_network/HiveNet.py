import torch
from torch import nn
from torch.nn import functional as F
from hive_vision import HiveNetVision


class HiveNet(nn.Module):
    def __init__(self, kernel_size, stride, num_of_thresholds,
                 hidden_layer_size=64,
                 vision_net_output=32,
                 actions_per_threshold=2):

        super(HiveNet, self).__init__()
        self.vision = HiveNetVision(kernel_size, stride, outputs=vision_net_output)
        self.policy_hidden1 = nn.Linear(in_features=vision_net_output + num_of_thresholds,
                                        out_features=hidden_layer_size)
        possible_actions_size = actions_per_threshold*num_of_thresholds
        self.policy_output = nn.Linear(in_features=hidden_layer_size,
                                       out_features=possible_actions_size)

        self.value_hidden1 = nn.Linear(in_features=vision_net_output + num_of_thresholds,
                                       out_features=hidden_layer_size)
        self.value_output = nn.Linear(in_features=hidden_layer_size,
                                      out_features=possible_actions_size)

    def forward(self, map_input, thresholds):
        x = self.vision(map_input)
        x = torch.cat((x, thresholds), dim=1)

        actor_x = F.relu(self.policy_hidden1(x))
        actor_x = F.relu(self.policy_output(actor_x))
        action_probabilities = F.softmax(actor_x)

        critic_x = F.relu(self.value_hidden1(x))
        critic_x = F.relu(self.value_output(critic_x))
        values = F.tanh(critic_x)
        return action_probabilities, values





