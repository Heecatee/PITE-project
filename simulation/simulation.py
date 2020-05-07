import collections
from random import random

import pygame
from pygame.locals import *
from pygame.color import *

import pymunk
import pymunk.autogeometry
import pymunk.pygame_util

import utils.simulation_utils as utils
import utils.simulation_pymunk_utils as pymunk_utils
import utils.simulation_pygame_utils as pygame_utils
from utils.simulation_pymunk_utils import SCREEN_HEIGHT, SCREEN_WIDTH


class SwarmBallSimulation(object):
    def __init__(self):
        # main simulation parameters
        self.number_of_clusters = 3
        self.number_of_agents_per_cluster = 10
        self.goal_object_frame_dim = [100, 100]
        self.enemy_speed = 0.5
        self.debug_mode = False

        self._thresholds = []
        self._map_segments = [
            ((0, 100), (100, 100)),
            ((100, 100), (200, 200)),
            ((200, 200), (1280, 300))
        ]
        self._enemy_position = 0

        # simulation objects
        self._clusters = []
        self._goal_object = None
        self._giant_fry = None
        self._map = None

        # simulation flow parameters
        self._simulation_is_running = True
        self._ticks_to_next_ball = 10

        # pymunk constants
        self._space = pymunk.Space()
        self._space.gravity = [0.0, -900]
        self._dt = 1 / 60.0
        self._physics_steps_per_frame = 1

        # pygame constants
        self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self._clock = pygame.time.Clock()
        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        pygame.init()

    def update_thresholds(self, *accelerations):
        if len(accelerations) == len(self._thresholds):
            # TODO: update velocity of all thresholds
            pass

    # all simulation parameters must be set before launching this method
    def init_simulation(self):
        self._init_static_scenery()
        self._init_simulation_objects()
        self._run()

    @property
    def space_near_goal_object(self):
        pygame.image.tostring(self._screen, "RGB")
        return None

    def _run(self):
        while self._simulation_is_running:
            # Few steps per frame to keep simulation smooth
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)
            self._process_events()
            self._update_simulation_objects()
            self._redraw()

    def _init_static_scenery(self):
        self._map = pymunk_utils.create_map_segments(self._map_segments, self._space)
        self._space.add(self._map)

    def _init_simulation_objects(self):
        self._clusters = pymunk_utils.create_clusters(self.number_of_clusters, self.number_of_agents_per_cluster)
        self._goal_object = pymunk_utils.create_goal_object(600)

        objects = [(self._goal_object.body, self._goal_object)]
        for cluster in self._clusters:
            [objects.append((agent.body, agent)) for agent in cluster.agents]

        self._space.add(objects)

    def _update_simulation_objects(self):
        for cluster in self._clusters:
            # TODO: update threshold position
            for agent in cluster.agents:
                # TODO: move to other method -> update_agents
                agent.body.angular_velocity = utils.get_agent_velocity(
                    cluster.threshold.position,
                    agent.body.position.x
                )
        self._enemy_position += self.enemy_speed

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self._simulation_is_running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                self._simulation_is_running = False
            elif event.type == KEYDOWN and event.key == K_p:
                pygame.image.save(self._screen, "swarm_ball_simulation.png")

    def _redraw(self):
        self._screen.fill(THECOLORS["white"])
        if self.debug_mode:
            self._space.debug_draw(self._draw_options)
            pygame_utils.draw_thresholds(self._screen, self._clusters)
            pygame_utils.draw_line(self._screen, self._enemy_position)
        self._clock.tick(50)
        pygame.display.flip()


if __name__ == '__main__':
    swarmBallSimulation = SwarmBallSimulation()
    swarmBallSimulation.debug_mode = True
    swarmBallSimulation.init_simulation()
