import gym
from gym import spaces, logger
from gym.utils import seeding

try:
    from .simulation.simulation import SwarmBallSimulation
except ImportError:
    from simulation.simulation import SwarmBallSimulation

class SwarmBall(gym.Env):
    def __init__(self, acc_factor=0.25, number_of_clusters=3, **kwargs):
        self.sim = SwarmBallSimulation(number_of_clusters, **kwargs)
        self.cluster_count = number_of_clusters
        self.thresh_vel = [0 for _ in range(number_of_clusters)]
        self.acc_factor = acc_factor

    def reward(self):
        points = self.sim._goal_object.body.position[0] - self.goal_prev_pos
        self.goal_prev_pos = self.sim._goal_object.body.position[0]
        return points

    def step(self, action):
        self.thresh_vel = [self.thresh_vel[i] + (2*action[i]-1)*self.acc_factor for i in range(self.cluster_count)]
        for i in range(self.cluster_count):
            self.sim.update_thresholds_position(i, self.sim.threshold_positions()[i] + self.thresh_vel[i])
        self.sim.step()
        observations = {'picture': self.sim.space_near_goal_object(), 'thresholds': self.sim.threshold_positions()}
        return observations, self.reward(), self.sim._enemy_position >= self.sim._goal_object.body.position[0] , {'message': 'You look great today cutiepie!'}

    def reset(self):
        self.thresh_vel = [0 for _ in range(self.cluster_count)]
        self.sim.reset()
        self.goal_prev_pos = self.sim._goal_object.body.position[0]
        self.initial_goal_position = self.sim._goal_object.body.position[0]
        return {'picture': self.sim.space_near_goal_object(), 'thresholds': self.sim.threshold_positions()}

    def render(self):
        self.sim.redraw()

    def close(self):
        """
            Zamknięcie środowiska.
        """
        pass
