import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

try:
    from .simulation.simulation import SwarmBallSimulation
except ImportError:
    from simulation.simulation import SwarmBallSimulation


class SwarmBall(gym.Env):
    def __init__(self, acc_factor=0.25, number_of_clusters=3, v_max=10, shape=(720, 540), **kwargs):
        self.sim = SwarmBallSimulation(number_of_clusters, **kwargs)
        self.cluster_count = number_of_clusters
        self.thresh_vel = np.zeros(number_of_clusters)
        self.v_max = v_max
        self.acc_factor = acc_factor
        self.shape = shape

    def reward(self):
        points = self.sim._goal_object.body.position[0] - self.goal_prev_pos
        self.goal_prev_pos = self.sim._goal_object.body.position[0]
        return points

    def step(self, action):
        self.thresh_vel = self.thresh_vel + (2*np.array(action)-1) * self.acc_factor
        self.thresh_vel = np.clip(self.thresh_vel, -self.v_max, self.v_max)
        for i in range(self.cluster_count):
            self.sim.update_thresholds_position(
                i, self.sim.threshold_positions()[i] + self.thresh_vel[i])
        self.sim.step()
        observations = {'picture': self.sim.space_near_goal_object(self.shape), 'thresholds': np.array(self.sim.threshold_positions()) - self.sim._goal_object.body.position[0]}
        return observations, self.reward(), self.sim._enemy_position >= self.sim._goal_object.body.position[0] , {'message': 'You look great today cutiepie!'}

    def reset(self):
        self.thresh_vel = [0 for _ in range(self.cluster_count)]
        self.sim.reset()
        self.goal_prev_pos = self.sim._goal_object.body.position[0]
        self.initial_goal_position = self.sim._goal_object.body.position[0]
        return {'picture': self.sim.space_near_goal_object(self.shape), 'thresholds': np.array(self.sim.threshold_positions()) - self.sim._goal_object.body.position[0]}

    def render(self):
        self.sim.redraw()

    def close(self):
        """
            Zamknięcie środowiska.
        """
        pass
