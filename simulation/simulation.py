import pygame
from pygame.locals import *
from pygame.color import *

import pymunk
import pymunk.pygame_util

import simulation.utils.simulation_utils as utils
import simulation.utils.simulation_pymunk_utils as pymunk_utils
import simulation.utils.simulation_pygame_utils as pygame_utils
from simulation.utils.simulation_pymunk_utils import SCREEN_HEIGHT, SCREEN_WIDTH


class SwarmBallSimulation(object):
    def __init__(self, thresh_pos=[200, 500, 800], bots_per_thresh=10, obj_dim=[100, 100], 
    debug_mode=False, gravity=[0.0, -900], dt=1/60, steps_per_frame=1):
        # main simulation parameters
        self.thresholds_positions_x = thresh_pos
        self.number_of_agents_per_thresholds = bots_per_thresh
        self.goal_object_frame_dim = obj_dim
        self.debug_mode = debug_mode

        # simulation objects
        self._clusters = []
        self._goal_object = None
        self._giant_fry = None

        # simulation flow parameters
        self._simulation_is_running = True
        self._ticks_to_next_ball = 10

        # pymunk constants
        self._space = pymunk.Space()
        self._space.gravity = gravity
        self._dt = dt
        self._physics_steps_per_frame = steps_per_frame

        # pygame constants
        self._screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self._clock = pygame.time.Clock()
        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        pygame.init()
        self._init_static_scenery()
        self._init_simulation_objects()

    def step(self):
        # Few steps per frame to keep simulation smooth
        for x in range(self._physics_steps_per_frame):
            self._space.step(self._dt)
        self._process_events()
        self._update_simulation_objects()

    def run(self):
        while self._simulation_is_running:
            self.step()
            self._redraw()
            

    def request_space_near_goal_object(self):
        # it will be done :D no worries
        pass

    def _init_static_scenery(self):
        static_body = self._space.static_body
        static_line = pymunk.Segment(static_body, (0, 360.0), (1280, 360.0), 0.0)
        static_line.elasticity = pymunk_utils.ELASTICITY
        static_line.friction = pymunk_utils.FRICTION
        self._space.add(static_line)

    def _init_simulation_objects(self):
        self._clusters = pymunk_utils.create_clusters(
            self.thresholds_positions_x,
            self.number_of_agents_per_thresholds
        )
        self._goal_object = pymunk_utils.create_goal_object(600)

        objects = [(self._goal_object.body, self._goal_object)]
        for cluster in self._clusters:
            [objects.append((agent.body, agent)) for agent in cluster.agents]

        self._space.add(objects)

    def _update_simulation_objects(self):
        for cluster in self._clusters:
            for agent in cluster.agents:
                agent.body.angular_velocity = utils.get_agent_velocity(
                    cluster.threshold_position,
                    agent.body.position.x
                )

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
            pass
        self._clock.tick(50)
        pygame.display.flip()


if __name__ == '__main__':
    swarmBallSimulation = SwarmBallSimulation()
    swarmBallSimulation.debug_mode = True
    swarmBallSimulation.run()
