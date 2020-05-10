import gym
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

CART_POLE_IN_NUM = 4
CART_POLE_OUT_NUM = 2

# ONLY NETWORK HERE  ==> IT'LL BE REPLACED WITH OUR NETWORK


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(CART_POLE_IN_NUM, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, CART_POLE_OUT_NUM),
            nn.Softmax(dim=1))

        self.value = nn.Sequential(
            nn.Linear(CART_POLE_IN_NUM, 50),
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
    def __init__(self, environment_name, batch_size, gamma, beta_entropy, learning_rate, clip_size):
        self.net = Net()
        self.env = gym.make(environment_name)
        self.batch_size = batch_size
        self.gamma = gamma
        self.beta_entropy = beta_entropy
        self.learning_rate = learning_rate
        self.clip_size = clip_size
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.learning_rate)

    def train(self, batch_number):
        RENDER_INTERVAL = 100
        for batch in range(batch_number):
            values = []
            action_logs = []
            rewards = []
            entropy_loss = 0
            current_state = self.env.reset()
            for simulation_step in range(self.batch_size):
                if episode % RENDER_INTERVAL == 0:
                    self.env.render()
                # Aktor zwraca nam decyzję po części losową (prawdopodobieństwa)
                # Krytyk zwraca q-value dla danego stanu
                # czyli jak do tej pory przewidywaliśmy, że może być dobrz/źle
                action_prob, value = self.net.forward(current_state)
                value = value.detach().numpy()[0, 0]
                probs = action_prob.detach().numpy()
                chosen_one = np.random.choice(
                    CART_POLE_OUT_NUM, p=np.squeeze(probs))
                log_prob = torch.log(action_prob.squeeze(0)[chosen_one])

                entropy = -np.sum(np.mean(probs) * np.log(probs))

                observation, reward, done, _ = self.env.step(chosen_one)

                # Zapisujemy zmienne z kroku do batcha
                rewards.append(reward)
                values.append(value)
                action_logs.append(log_prob)
                entropy_loss += entropy

                # Aktualizujemy stan
                current_state = observation
                if done or simulation_step == self.batch_size - 1:
                    Qval, _ = self.net.forward(observation)
                    Qval = Qval.detach().numpy()[0, 0]
                    break

            Qvals = np.zeros_like(values)
            for it in reversed(range(len(rewards))):
                Qval = rewards[it] + self.gamma * Qval
                Qvals[it] = Qval
            values = torch.FloatTensor(values)
            Qvals = torch.FloatTensor(Qvals)
            action_logs = torch.stack(action_logs)
            advantage = Qvals - values
            actor_loss = (-action_logs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            loss = actor_loss + critic_loss + 0.001 * entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return sum(rewards)


if __name__ == '__main__':
    trainer = A2CTrainer('CartPole-v1',
                         batch_size=600,
                         gamma=0.99,
                         beta_entropy=0.001,
                         learning_rate=0.001,
                         clip_size=0.2)
    best = 0
    for episode in range(1500):
        curr = trainer.train(1)
        if curr > best:
            best = curr
        print(f'{episode}. {curr}\tBest: {best}')
