import torch


class DataCollector:
    def __init__(self, net, out_num, environment, gamma):
        self.net = net
        self.out_num = out_num
        self.env = environment
        self.gamma = gamma
        self.rewards = []
        self.action_logarithms = []
        self.states = []
        self.render = False
        self.actions = []
        self.Qval = 0

    def clear_previous_batch_data(self):
        self.np_Qvals = []
        self.rewards = []
        self.action_logarithms = []
        self.states = []
        self.actions = []
        self.Qval = 0

    def calculate_qvals(self):
        Qval = 0
        Qvals = []
        for reward in reversed(self.rewards):
            Qval = reward + self.gamma * Qval
            Qvals.insert(0, Qval)
        return torch.tensor(Qvals)

    def collect_data_for(self, batch_size):
        current_state = self.env.reset()
        for simulation_step in range(batch_size):

            if self.render:
                self.env.render()

            action = self.net.pick_action(
                current_state['picture'], current_state['thresholds'], self)
            observation, reward, done, _ = self.env.step(action)
            self.rewards.append(reward)
            current_state = observation
            if done or simulation_step == batch_size - 1:
                self.Qval = self.calculate_qvals()
                break

    def stack_data(self):
        self.states = torch.stack(self.states)
        self.actions = torch.stack(self.actions)
        self.action_logarithms = torch.stack(self.action_logarithms)
