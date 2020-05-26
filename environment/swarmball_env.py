import gym
from gym import spaces, logger
from gym.utils import seeding
from .simulation.simulation import SwarmBallSimulation

class SwarmBall(gym.Env):
    def __init__(self):
        self.sim = SwarmBallSimulation()
        self.thresholds = [[],[],[]]

    def seed(self, seed=None):
        """
            To przerażające nasiono!!!
        """
        pass

    def reward(self):
        pass

    def step(self, action):
        self.thresholds.pop(0)
        self.thresholds.append(self.sim.threshold_positions())
        for i in range(len(action)):
            print(i, action[i])
            self.sim.update_thresholds_position(i, action[i])
        self.sim.step()
        observations = {'picture': self.sim.space_near_goal_object, 'past_thresholds': self.thresholds}
        return observations, self.reward(), False, {}

    def reset(self):
        self.sim.reset()
        self.goal_prev_pos = self.sim._goal_object.body.position[0]

    def render(self):
        self.sim.redraw()

    def close(self):
        """
            Zamknięcie środowiska.
        """
        pass