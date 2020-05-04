import pygame
import pymunk

from pygame.color import THECOLORS

from utils.simulation_pymunk_utils import SCREEN_HEIGHT, Color

GOAL_OBJECT_COLOR = Color(51, 153, 255)
MAP_COLOR = Color(54, 54, 54)


def draw_thresholds(screen, clusters):
    for cluster in clusters:
        points = [(cluster.threshold.position, 0), (cluster.threshold.position, SCREEN_HEIGHT)]
        # pygame is stupid -> it doesn't allow named parameters
        pygame.draw.lines(screen, cluster.color, False, points)


def draw_enemy(screen, position):
    points = [(position, 0), (position, SCREEN_HEIGHT)]
    pygame.draw.lines(screen, THECOLORS["red"], False, points)


def draw_clusters(screen, clusters):
    for cluster in clusters:
        for agent in cluster.agents:
            position = pymunk.pygame_util.to_pygame(agent.body.position, screen)
            pygame.draw.circle(screen, agent.color, position, int(agent.radius))


def draw_map(screen, map_segments):
    for (p1, p2) in map_segments:
        p1_translated = pymunk.pygame_util.to_pygame(p1, screen)
        p2_translated = pymunk.pygame_util.to_pygame(p2, screen)
        pygame.draw.aaline(screen, MAP_COLOR, p1_translated, p2_translated)


def draw_goal_object(screen, goal_object):
    body = goal_object.body
    points = [point.rotated(body.angle) + body.position for point in goal_object.get_vertices()]
    points.append(points[0])
    ps = [pymunk.pygame_util.to_pygame(point, screen) for point in points]
    pygame.draw.polygon(screen, GOAL_OBJECT_COLOR, ps)
