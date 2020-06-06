import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from .hive_vision.HiveNetVision import HiveNetVision

class HiveNet(nn.Module):
  def __init__(self, kernel_size, stride, num_of_thresholds,
               hidden_layer_size=64,
               vision_net_output=32,
               actions_per_threshold=2):

    super(HiveNet, self).__init__()
    self.vision = HiveNetVision(kernel_size, stride, outputs=vision_net_output)
    self.policy_hidden1 = nn.Linear(in_features=vision_net_output + num_of_thresholds,
                                    out_features=hidden_layer_size)
    self.num_of_thresholds = num_of_thresholds
    possible_actions_size = actions_per_threshold * self.num_of_thresholds
    self.policy_output = nn.Linear(in_features=hidden_layer_size,
                                   out_features=possible_actions_size)

    self.value_hidden1 = nn.Linear(in_features=vision_net_output + self.num_of_thresholds,
                                   out_features=hidden_layer_size)
    self.value_output = nn.Linear(in_features=hidden_layer_size,
                                  out_features=1)

  def pick_action(self, map_input, thresholds, collector):
    x = self.vision(map_input)
    x = torch.cat((x, torch.Tensor(thresholds)))

    actor_x = F.relu(self.policy_hidden1(x))
    actor_x = F.relu(self.policy_output(actor_x))
    action_probabilities = F.softmax(actor_x)
    distribution = Categorical(action_probabilities)
    action = distribution.sample()

    collector.states.append(x)
    collector.actions.append(action)
    collector.action_logarithms.append(
        distribution.log_prob(action))
    return np.unpackbits(np.uint8(action.item()))[-self.num_of_thresholds:]

  def evaluate(self, state, action):
    actor_x = F.relu(self.policy_hidden1(state))
    actor_x = F.relu(self.policy_output(actor_x))
    probabilities = F.softmax(actor_x)
    distribution = Categorical(probabilities)
    logarithm_probabilities = distribution.log_prob(action)
    entropy = distribution.entropy()

    critic_x = F.relu(self.value_hidden1(state))
    critic_x = F.relu(self.value_output(critic_x))
    Qvalue = F.tanh(critic_x)

    return logarithm_probabilities, torch.squeeze(Qvalue), entropy
