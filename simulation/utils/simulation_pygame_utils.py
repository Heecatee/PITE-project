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


def draw_enemy(screen, position, offset):
    points = [(position+offset[0], 0), (position+offset[0], SCREEN_HEIGHT)]
    pygame.draw.lines(screen, THECOLORS["red"], False, points)


def draw_clusters(screen, clusters, offset):
    for cluster in clusters:
        for agent in cluster.agents:
            position = pymunk.pygame_util.to_pygame(agent.body.position, screen)
            pygame.draw.circle(screen, agent.color, (position[0]+int(offset[0]), position[1]), int(agent.radius))


def draw_map(screen, map_segment, offset):
    for fragment in map_segment:
        p1_translated = pymunk.pygame_util.to_pygame(fragment.a, screen)
        p2_translated = pymunk.pygame_util.to_pygame(fragment.b, screen)
        pygame.draw.line(screen, MAP_COLOR, p1_translated+pymunk.Vec2d(offset), p2_translated+pymunk.Vec2d(offset), 4)


def draw_goal_object(screen, goal_object, position):
    body = goal_object.body
    points = [point.rotated(body.angle) + pymunk.Vec2d(position, body.position[1]) for point in goal_object.get_vertices()]
    points.append(points[0])
    ps = [pymunk.pygame_util.to_pygame(point, screen) for point in points]
    pygame.draw.polygon(screen, GOAL_OBJECT_COLOR, ps)
