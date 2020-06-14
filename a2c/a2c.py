import torch
import copy

try:
    from .utils.data_collector import DataCollector
except ImportError:
    from utils.data_collector import DataCollector


class A2CTrainer:
    def __init__(self, net, out_num, environment, batch_size,
                 gamma, beta_entropy, learning_rate, clip_size):

        self.net = net
        self.new_net = copy.deepcopy(net)
        self.batch_size = batch_size
        self.beta_entropy = beta_entropy
        self.learning_rate = learning_rate
        self.clip_size = clip_size
        self.optimizer = torch.optim.Adam(
            self.new_net.parameters(), lr=self.learning_rate)
        self.data = DataCollector(net, out_num, environment, gamma)

    def calculate_actor_loss(self, ratio, advantage):
        opt1 = ratio * advantage
        opt2 = torch.clamp(ratio, 1 - self.clip_size,
                           1 + self.clip_size) * advantage
        return (-torch.min(opt1, opt2)).mean()

    def calculate_critic_loss(self, advantage):
        return 0.5 * advantage.pow(2).mean()

    def train(self):
        self.data.clear_previous_batch_data()
        self.data.collect_data_for(batch_size=self.batch_size)
        self.data.stack_data()

        action_logarithms, Qval, entropy = self.new_net.evaluate(
            self.data.states, self.data.actions)

        ratio = torch.exp(action_logarithms -
                          self.data.action_logarithms.detach())
        advantage = self.data.Qval - Qval.detach()
        actor_loss = self.calculate_actor_loss(ratio, advantage)
        critic_loss = self.calculate_critic_loss(advantage)

        loss = actor_loss + critic_loss + self.beta_entropy * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.net.load_state_dict(self.new_net.state_dict())
        return sum(self.data.rewards), self.net
