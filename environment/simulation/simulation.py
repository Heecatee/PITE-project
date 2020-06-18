import math

import pygame
from pygame.color import THECOLORS
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, K_p

import pymunk
import pymunk.autogeometry
import pymunk.pygame_util

try:
    from .utils import simulation_utils as utils
    from .utils import simulation_pymunk_utils as pymunk_utils
    from .utils import simulation_pygame_utils as pygame_utils
    from .utils.generate_map import Difficulty
except ImportError:
    import utils.simulation_utils as utils
    import utils.simulation_pymunk_utils as pymunk_utils
    import utils.simulation_pygame_utils as pygame_utils
    from utils.generate_map import Difficulty


class SwarmBallSimulation(object):
    def __init__(self,
                 number_of_clusters=3,
                 number_of_bots_per_cluster=10,
                 enemy_acceleration=0.005,
                 difficulty=None,
                 map_segment_size=(600, 600),
                 initial_object_height=10,
                 screen_size=(1800, 840),
                 ticks_per_step=1,
                 ticks_per_render_frame=50,
                 gravity=(0.0, -900),
                 map_bottom_y_threshold=-300,
                 map_width=5
                 ):
        # external simulation properties
        self.debug = False
        self.number_of_clusters = number_of_clusters
        self.number_of_bots_per_cluster = number_of_bots_per_cluster
        self.enemy_acceleration = enemy_acceleration
        self.initial_object_position = (0.0, initial_object_height)
        self.ticks_per_step = ticks_per_step
        self.ticks_per_render_frame = ticks_per_render_frame
        self.difficulty = difficulty
        self.map_segment_size = map_segment_size
        self.map_width = map_width
        self.map_bottom_y_threshold = map_bottom_y_threshold
        self.gravity = gravity
        self.screen_size = screen_size

        # internal simulation properties
        self._simulation_is_running = True
        self._dt = 1 / 80.0
        self._segment_count = 0
        self._current_map_end = (-1.5 * map_segment_size[0], 0.0)
        self._map_middle_right_boundary = (0.5*map_segment_size[0], 0.0)
        self._enemy_position = -map_segment_size[0]
        self._enemy_speed = 0

        # internal simulation objects
        self._map = []
        self._map_sprite = None
        self._space = None
        self._enemy = None
        self._clusters = None
        self._goal_object = None

        # pygame constants
        self._screen = pygame.display.set_mode(self.screen_size)
        self._clock = pygame.time.Clock()
        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        pygame.init()

    # input
    def update_thresholds_position(self, index, position):
        self._clusters[index].threshold.position = position

    # output
    def threshold_positions(self):
        return [cluster.threshold.position for cluster in self._clusters]

    # output
    def space_near_goal_object(self):
        self._update_screen()
        return pygame.image.tostring(pygame.transform.scale(self._screen.copy(), (720, 280)), "RGB")

    def reset(self):
        if self._space is not None:
            self._space.remove(self._space._get_shapes())
        self._space = pymunk.Space()
        self._space.gravity = self.gravity

        self._map = []
        self._segment_count = 0
        self._current_map_end = (-1.5 * self.map_segment_size[0], 0.0)
        self._enemy_position = -1.5 * self.map_segment_size[0]
        self._enemy_speed = 0


        self._init_simulation_objects()
        self._init_static_scenery()

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
            self._segment_count += 1
            map_segment, segment_end_point = pymunk_utils.create_map_segment(difficulty=self.difficulty,
                                                                             space=self._space,
                                                                             starting_point=self._current_map_end,
                                                                             segment_size=self.map_segment_size,
                                                                             map_width=self.map_width,
                                                                             segment_count=self._segment_count)
            self._map.append(map_segment)
            self._map_middle_right_boundary = self._current_map_end
            self._current_map_end = segment_end_point

        for map_segment in self._map:
            self._space.add(map_segment)
        self._update_map_sprite()

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
        self._enemy_speed += math.log1p(self.enemy_acceleration)
        self._enemy_position += self._enemy_speed

    def _update_bots(self):
        for cluster in self._clusters:
            cluster.threshold.position = cluster.threshold.position + 1
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
            self._segment_count += 1
            map_segment, segment_end_point = pymunk_utils.create_map_segment(difficulty=self.difficulty,
                                                                             space=self._space,
                                                                             starting_point=self._current_map_end,
                                                                             segment_size=self.map_segment_size,
                                                                             map_width=self.map_width,
                                                                             segment_count=self._segment_count)
            self._space.remove(self._map[0])
            self._map.pop(0)
            self._map.append(map_segment)
            self._map_middle_right_boundary = self._current_map_end
            self._current_map_end = segment_end_point
            self._space.add(self._map[-1])
            self._update_map_sprite()

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self._simulation_is_running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                self._simulation_is_running = False
            elif event.type == KEYDOWN and event.key == K_p:
                pygame.image.save(self._screen, "swarm_ball_simulation.png")

    def _update_map_sprite(self):
        self._screen.fill(THECOLORS["white"])
        self._map_offset = (self._current_map_end[0] - 3 * self.map_segment_size[0],
                            self.screen_size[1] // 2 - self._goal_object.body.position[1])
        for map_segment in self._map:
            pygame_utils.draw_map(self._screen, map_segment, self.map_width,
                                  map_offset=self._map_offset)
        self._map_sprite = self._screen.copy()

    def _update_screen(self):
        self._screen.fill(THECOLORS["white"])
        offset = (self.screen_size[0] / 2 - self._goal_object.body.position[0],
                  -self.screen_size[1] // 2 + self._goal_object.body.position[1])
        self._screen.blit(self._map_sprite, (offset[0]+self._map_offset[0], offset[1]+self._map_offset[1]))
        if self.debug:
            pygame_utils.draw_thresholds(self._screen, self._clusters, offset, self.screen_size)
        pygame_utils.draw_clusters(self._screen, self._clusters, offset)
        pygame_utils.draw_goal_object(self._screen, self._goal_object, self.screen_size)

    def redraw(self, clock=False):
        self._update_screen()
        if clock is True:
            self._clock.tick(self.ticks_per_render_frame)
        pygame_utils.draw_enemy(self._screen, self._enemy_position, offset, self.screen_size)
        pygame.display.flip()


if __name__ == '__main__':
    swarmBallSimulation = SwarmBallSimulation()
    swarmBallSimulation.debug = True
    swarmBallSimulation.reset()
    swarmBallSimulation.run()
    print(swarmBallSimulation.threshold_positions())
