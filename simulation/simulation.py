import pygame
from pygame.color import THECOLORS
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_p

import pymunk
import pymunk.autogeometry
import pymunk.pygame_util

import utils.simulation_utils as utils
import utils.simulation_pymunk_utils as pymunk_utils
import utils.simulation_pygame_utils as pygame_utils

from utils.generate_map import Difficulty


class SwarmBallSimulation(object):
    def __init__(self,
                 number_of_clusters=3,
                 number_of_bots_per_cluster=10,
                 enemy_speed=0.5,
                 difficulty=Difficulty.EASY,
                 map_segment_size=(300, 100),
                 initial_object_position=(600, 500),
                 screen_size=(1280, 540),
                 dt=1 / 60.0,
                 ticks_per_step=1,
                 ticks_per_render_frame=50,
                 gravity=(0.0, -900),
                 map_bottom_y_threshold=-300,
                 map_width=3
                 ):
        self._space = None
        self.gravity = gravity
        self._map = []
        self.map_width = map_width
        self.map_bottom_y_threshold = map_bottom_y_threshold
        self.initial_object_position = initial_object_position
        self._map_segment_size = map_segment_size
        map_initial_x_offset = (self.initial_object_position[0]-self._map_segment_size[0])
        self._map_beginning = (-map_initial_x_offset, 0.0)
        self._map_middle_right_boundary = (map_initial_x_offset, 0.0)
        self.ticks_per_step = ticks_per_step
        self.ticks_per_render_frame = ticks_per_render_frame
        self.difficulty = difficulty
        self._giant_fry = None
        self._simulation_is_running = True
        self.number_of_clusters = number_of_clusters
        self.number_of_bots_per_cluster = number_of_bots_per_cluster
        self.enemy_speed = enemy_speed
        self._clusters = None
        self._goal_object = None
        self._thresholds = []
        self._giant_fry_position = -map_initial_x_offset
        self._dt = dt

        # pygame constants
        self.screen_size = screen_size
        self._screen = pygame.display.set_mode(self.screen_size)
        self._clock = pygame.time.Clock()
        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        pygame.init()

    def update_thresholds(self, *accelerations):
        if len(accelerations) == len(self._thresholds):
            # TODO: update velocity of all thresholds
            pass

    def reset(self):
        self._space = pymunk.Space()
        self._space.gravity = self.gravity
        self._init_static_scenery()
        self._init_simulation_objects()

    def space_near_goal_object(self, window_size):
        pygame.image.tostring(self._screen, "RGB")
        return None

    def step(self):
        for _ in range(self.ticks_per_step):
            self._space.step(self._dt)
        self._update_map()
        self._update_simulation_objects()

    def run(self):
        while self._simulation_is_running:
            self.step()
            self._process_events()
            self.redraw()

    def _init_static_scenery(self):
        number_of_starting_platforms = 3
        for _ in range(number_of_starting_platforms):
            map_segment, segment_end_point = pymunk_utils.create_map_segment(difficulty=self.difficulty,
                                                                             space=self._space,
                                                                             starting_point=self._map_beginning,
                                                                             segment_size=self._map_segment_size,
                                                                             map_width=self.map_width)
            self._map.append(map_segment)
            self._map_middle_right_boundary = self._map_beginning
            self._map_beginning = segment_end_point

        for map_segment in self._map:
            self._space.add(map_segment)

    def _init_simulation_objects(self):
        self._clusters = pymunk_utils.create_clusters(self.number_of_clusters,
                                                      self.screen_size,
                                                      self.number_of_bots_per_cluster)
        self._goal_object = pymunk_utils.create_goal_object(self.initial_object_position)

        objects = [(self._goal_object.body, self._goal_object)]
        for cluster in self._clusters:
            [objects.append((bot.body, bot)) for bot in cluster.bots]

        self._space.add(objects)

    def _update_simulation_objects(self):
        self._update_bots()
        self._giant_fry_position += self.enemy_speed

    def _update_bots(self):
        for cluster in self._clusters:
            for bot in cluster.bots:
                if bot.body.position[1] < self.map_bottom_y_threshold:
                    self._space.remove(bot)
                    cluster.bots.remove(bot)
                else:
                    bot.body.angular_velocity = utils.get_bot_velocity(
                        cluster.threshold.position,
                        bot.body.position.x
                    )

    def _update_map(self):
        if self._goal_object.body.position[0] > self._map_middle_right_boundary[0]:
            map_segment, segment_end_point = pymunk_utils.create_map_segment(difficulty=self.difficulty,
                                                                             space=self._space,
                                                                             starting_point=self._map_beginning,
                                                                             segment_size=self._map_segment_size,
                                                                             map_width=self.map_width)
            self._space.remove(self._map[0])
            self._map.pop(0)
            self._map.append(map_segment)
            self._map_middle_right_boundary = self._map_beginning
            self._map_beginning = segment_end_point
            self._space.add(self._map[-1])

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self._simulation_is_running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                self._simulation_is_running = False
            elif event.type == KEYDOWN and event.key == K_p:
                pygame.image.save(self._screen, "swarm_ball_simulation.png")

    def redraw(self, debug=False):
        self._screen.fill(THECOLORS["white"])
        if debug is True:
            self._space.debug_draw(self._draw_options)
            pygame_utils.draw_thresholds(self._screen, self._clusters, self.screen_size)
        else:
            offset = (self.initial_object_position[0] - self._goal_object.body.position[0], 0)
            pygame_utils.draw_clusters(self._screen, self._clusters, offset)
            pygame_utils.draw_enemy(self._screen, self._giant_fry_position, offset, self.screen_size)
            for map_segment in self._map:
                pygame_utils.draw_map(self._screen, map_segment, offset, self.map_width)
            pygame_utils.draw_goal_object(self._screen, self._goal_object,  self.initial_object_position)
        self._clock.tick(self.ticks_per_render_frame)
        pygame.display.flip()


if __name__ == '__main__':
    swarmBallSimulation = SwarmBallSimulation()
    swarmBallSimulation.reset()
    swarmBallSimulation.run()

