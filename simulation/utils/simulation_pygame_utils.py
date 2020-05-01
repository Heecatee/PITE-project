import pygame
from pygame.color import THECOLORS

from utils.simulation_pymunk_utils import SCREEN_HEIGHT


def draw_thresholds(screen, clusters):
    for cluster in clusters:
        points = [(cluster.threshold.position, 0), (cluster.threshold.position, SCREEN_HEIGHT)]
        # pygame is stupid -> it doesn't allow named parameters
        pygame.draw.lines(screen, cluster.color, False, points)


def draw_line(screen, position):
    points = [(position, 0), (position, SCREEN_HEIGHT)]
    pygame.draw.lines(screen, THECOLORS["red"], False, points)
