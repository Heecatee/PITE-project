import gym
from gym import spaces, logger
from gym.utils import seeding
from .simulation.simulation import SwarmBallSimulation

class SwarmBall(gym.Env):
    def __init__(self, acc_factor=0.25, thresh_amount=3, **kwargs):
        self.sim = SwarmBallSimulation(**kwargs)
        self.thresholds = [[] for _ in range(thresh_amount)]
        self.thresh_vel = [0,0,0]
        self.acc_factor = acc_factor

    def reward(self):
        pass

    def step(self, action):
        self.thresholds.pop(0)
        self.thresholds.append(self.sim.threshold_positions())
        self.thresh_vel = [self.thresh_vel[i] + action[i]*self.acc_factor for i in range(len(self.thresh_vel))]
        for i in range(len(action)):
            self.sim.update_thresholds_position(i, self.sim.threshold_positions()[i] + self.thresh_vel[i])
        self.sim.step()
        observations = {'picture': self.sim.space_near_goal_object(0), 'thresholds': self.thresholds}
        return observations, self.reward(), self.sim._enemy_position >= self.sim._goal_object.body.position[0] , {'message': 'You look great today cutiepie!'}

    def reset(self):
        self.sim.reset()

    def render(self):
        self.sim.redraw()

    def close(self):
        """
            Zamknięcie środowiska.
        """
        pass