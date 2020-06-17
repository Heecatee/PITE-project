import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from .hive_vision.HiveNetVision import HiveNetVision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HiveNet(nn.Module):
    def __init__(self, kernel_size, stride, num_of_thresholds,
                 hidden_layer_size=64,
                 vision_net_output=32,
                 actions_per_threshold=2,
                 time_steps_stored=2):

        super(HiveNet, self).__init__()
        self.vision = HiveNetVision(
            kernel_size, stride, outputs=vision_net_output).to(device)
        self.num_of_thresholds = num_of_thresholds
        self.policy_hidden1 = nn.Linear(in_features=vision_net_output + time_steps_stored * self.num_of_thresholds,
                                        out_features=hidden_layer_size).to(device)
        self.thresholds_history = None
        self.time_steps_stored = time_steps_stored
        possible_actions_size = actions_per_threshold * self.num_of_thresholds
        self.policy_output = nn.Linear(in_features=hidden_layer_size,
                                       out_features=possible_actions_size).to(device)

        self.value_hidden1 = nn.Linear(in_features=vision_net_output + time_steps_stored * self.num_of_thresholds,
                                       out_features=hidden_layer_size).to(device)
        self.value_output = nn.Linear(in_features=hidden_layer_size,
                                      out_features=1).to(device)

    def update_thresholds_history(self, new_thresholds):
        if self.thresholds_history is None:
            self.thresholds_history = []
            for _ in range(self.time_steps_stored):
                self.thresholds_history.append(new_thresholds)
        else:
            self.thresholds_history.pop(0)
            self.thresholds_history.append(new_thresholds)
        return self.thresholds_history

    def pick_action(self, map_input, thresholds, collector):
        new_thresholds = torch.Tensor(thresholds).to(device)
        self.update_thresholds_history(new_thresholds)

        x = self.vision(map_input).to(device)
        x = torch.cat([x, *self.thresholds_history]).to(device)

        actor_x = F.relu(self.policy_hidden1(x)).to(device)
        actor_x = F.relu(self.policy_output(actor_x)).to(device)
        action_probabilities = F.softmax(actor_x).to(device)
        distribution = Categorical(action_probabilities).to(device)
        action = distribution.sample().to(device)

        collector.states.append(x)
        collector.actions.append(action)
        collector.action_logarithms.append(
            distribution.log_prob(action))

        def bit_representation(action, num_bits):
            return np.unpackbits(np.uint8(action))[-num_bits:]

        return bit_representation(action.item(), num_bits=self.num_of_thresholds)

    def evaluate(self, state, action):
        actor_x = F.relu(self.policy_hidden1(state)).to(device)
        actor_x = F.relu(self.policy_output(actor_x)).to(device)
        probabilities = F.softmax(actor_x).to(device)
        distribution = Categorical(probabilities).to(device)
        logarithm_probabilities = distribution.log_prob(action).to(device)
        entropy = distribution.entropy().to(device)

        critic_x = F.relu(self.value_hidden1(state)).to(device)
        critic_x = F.relu(self.value_output(critic_x)).to(device)
        Qvalue = F.tanh(critic_x).to(device)

        return logarithm_probabilities, torch.squeeze(Qvalue), entropy
