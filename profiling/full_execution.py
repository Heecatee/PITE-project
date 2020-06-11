import sys
sys.path.append('..')

from cProfile import run
from environment.swarmball_env import SwarmBall
from random import choice as ch
import logging

env = SwarmBall()
env.sim.debug = False
logging.info(env.reset())
while env.sim._simulation_is_running:
    env.sim._process_events()
    obs = env.step([ch([-1,1]), ch([-1,1]), ch([-1,1])])
    env.render()
    print(obs[0]['thresholds'], obs[1:])