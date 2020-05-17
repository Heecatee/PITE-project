import gym
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


# ONLY NETWORK HERE  ==> IT'LL BE REPLACED WITH OUR NETWORK


class Net(nn.Module):
    def __init__(self, in_num, out_num):
        super(Net, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(in_num, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, out_num),
            nn.Softmax(dim=1))

        self.value = nn.Sequential(
            nn.Linear(in_num, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1))

    def forward(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        action_prob = self.policy(x.float())
        value = self.value(x.float())
        return action_prob, value
# NETWORK PART END


class A2CTrainer:
    def __init__(self, net, out_num, environment_name, batch_size,
                 gamma, beta_entropy, learning_rate, clip_size):

        self.net = net
        self.out_num = out_num
        self.env = gym.make(environment_name)
        self.batch_size = batch_size
        self.gamma = gamma
        self.beta_entropy = beta_entropy
        self.learning_rate = learning_rate
        self.clip_size = clip_size
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.learning_rate)
        self.estim_values = []
        self.action_logarithms = []
        self.rewards = []
        self.np_entropy_loss = 0
        self.np_Qvals = None
        self.render = False

    def choose_action(self, probabilities):
        return np.random.choice(self.out_num, p=np.squeeze(probabilities))

    def calculate_entropy(self, action_chance):
        return -np.sum(np.mean(action_chance) * np.log(action_chance))

    def calculate_logarithm_probs(self, probabilities, chosen_action):
        return torch.log(probabilities.squeeze(axis=0)[chosen_action])

    def clear_previous_batch_data(self):
        self.rewards = []
        self.np_estim_values = []
        self.tensor_action_logarithms = []
        self.np_entropy_loss = 0

    def save_batch_data(self, reward, np_estim_value,
                        tensor_logarithm_probs, np_entropy):
        self.rewards.append(reward)
        self.np_estim_values.append(np_estim_value)
        self.tensor_action_logarithms.append(tensor_logarithm_probs)
        self.np_entropy_loss += np_entropy

    def calculate_qvals(self, np_Qval):
        self.np_Qvals = np.zeros_like(self.np_estim_values)
        for it in reversed(range(len(self.rewards))):
            np_Qval = self.rewards[it] + self.gamma * np_Qval
            self.np_Qvals[it] = np_Qval

    def calculate_actor_loss(self, advantage):
        single_tensor_action_logarithms = torch.stack(
            self.tensor_action_logarithms)
        return (-single_tensor_action_logarithms * advantage).mean()

    def calculate_critic_loss(self, advantage):
        return 0.5 * advantage.pow(2).mean()

    def train(self):
        self.clear_previous_batch_data()
        current_state = self.env.reset()

        for simulation_step in range(self.batch_size):

            if self.render:
                self.env.render()

            action_chance, estim_value = self.net.forward(current_state)

            np_action_chance = action_chance.detach().numpy()

            chosen_action = self.choose_action(np_action_chance)

            observation, reward, done, _ = self.env.step(chosen_action)

            np_estim_value = estim_value.detach().numpy()[0, 0]
            np_entropy = self.calculate_entropy(np_action_chance)
            tensor_logarithm_probabilities = self.calculate_logarithm_probs(
                action_chance, chosen_action)

            self.save_batch_data(reward,
                                 np_estim_value,
                                 tensor_logarithm_probabilities,
                                 np_entropy)

            current_state = observation
            if done or simulation_step == self.batch_size - 1:
                _, Qval = self.net.forward(observation)
                np_Qval = Qval.detach().numpy()[0, 0]
                break

        self.calculate_qvals(np_Qval)

        tensor_estim_values = torch.FloatTensor(self.np_estim_values)
        tensor_Qvals = torch.FloatTensor(self.np_Qvals)

        advantage = tensor_Qvals - tensor_estim_values
        actor_loss = self.calculate_actor_loss(advantage)
        critic_loss = self.calculate_critic_loss(advantage)

        loss = actor_loss + critic_loss + self.beta_entropy * self.np_entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return sum(self.rewards), self.net


if __name__ == '__main__':
    CART_POLE_IN_NUM = 4
    CART_POLE_OUT_NUM = 2
    NUM_OF_EPISODES = 1500
    RENDER_INTERVAL = 100
    trainer = A2CTrainer(net=Net(CART_POLE_IN_NUM, CART_POLE_OUT_NUM),
                         out_num=CART_POLE_OUT_NUM,
                         environment_name='CartPole-v1',
                         batch_size=600,
                         gamma=0.99,
                         beta_entropy=0.001,
                         learning_rate=0.001,
                         clip_size=0.2)
    best_result = 0
    in_which_episode = 0
    for episode in range(NUM_OF_EPISODES):
        if episode % RENDER_INTERVAL == 0:
            trainer.render = True
        else:
            trainer.render = False

        curr_result, _ = trainer.train()
        if curr_result > best_result:
            best_result = curr_result
            in_which_episode = episode
        print(
            f'{episode}. {curr_result}\tBest: {best_result} in episode {in_which_episode}')
