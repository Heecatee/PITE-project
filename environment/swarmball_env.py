import gym
from gym import spaces, logger
from gym.utils import seeding
from .simulation.simulation import SwarmBallSimulation

class SwarmBall(gym.Env):
    def __init__(self):
        self.sim = SwarmBallSimulation()

    def seed(self, seed=None):
        """
            To przerażające nasiono!!!
        """
        pass

    def step(self, action):
        self.sim.step()

    def reset(self):
        self.sim.reset()

    def render(self):
        self.sim.redraw()

    def close(self):
        """
            Zamknięcie środowiska.
        """
        pass