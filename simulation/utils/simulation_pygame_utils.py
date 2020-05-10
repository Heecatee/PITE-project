import pygame
import pymunk

from pygame.color import THECOLORS

GOAL_OBJECT_COLOR = THECOLORS["blue"]
MAP_COLOR = THECOLORS["black"]


def draw_thresholds(screen, clusters, screen_size):
    for cluster in clusters:
        points = [(cluster.threshold.position, 0), (cluster.threshold.position, screen_size[0])]
        pygame.draw.lines(screen, cluster.color, False, points)


def draw_enemy(screen, position, offset, screen_size):
    points = [(position+offset[0], 0), (position+offset[0], screen_size[0])]
    pygame.draw.lines(screen, THECOLORS["red"], False, points)


def draw_clusters(screen, clusters, offset):
    for cluster in clusters:
        for bot in cluster.bots:
            position = pymunk.pygame_util.to_pygame(bot.body.position, screen)
            pygame.draw.circle(screen, bot.color, (position[0]+int(offset[0]), position[1]), int(bot.radius))


def draw_map(screen, map_segment, offset, map_width):
    for fragment in map_segment:
        p1_translated = pymunk.pygame_util.to_pygame(fragment.a, screen)
        p2_translated = pymunk.pygame_util.to_pygame(fragment.b, screen)
        pygame.draw.line(screen, MAP_COLOR, p1_translated+pymunk.Vec2d(offset), p2_translated+pymunk.Vec2d(offset), 2*map_width)


def draw_goal_object(screen, goal_object, position):
    body = goal_object.body
    points = [point.rotated(body.angle) + pymunk.Vec2d(position[0], body.position[1]) for point in goal_object.get_vertices()]
    points.append(points[0])
    ps = [pymunk.pygame_util.to_pygame(point, screen) for point in points]
    pygame.draw.polygon(screen, GOAL_OBJECT_COLOR, ps)
