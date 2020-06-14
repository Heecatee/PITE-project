import sys
sys.path.append('..')

import cProfile
from environment.swarmball_env import SwarmBall
from random import choice as ch
import logging

env = SwarmBall()
env.sim.debug = False
logging.info(env.reset())
with cProfile.Profile(subcalls=False) as pr:
    while env.sim._simulation_is_running:
        env.sim._process_events()
        obs = env.step([ch([-1,1]), ch([-1,1]), ch([-1,1])])
        print(obs[0]['thresholds'], obs[1:])

pr.print_stats(sort='cumtime')