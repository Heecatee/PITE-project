import pygame

from simulation.utils.simulation_pymunk_utils import SCREEN_HEIGHT


def draw_thresholds(screen, clusters):
    for cluster in clusters:
        points = [(cluster.threshold_position, 0), (cluster.threshold_position, SCREEN_HEIGHT)]
        # pygame is stupid -> it doesn't allow named parameters
        pygame.draw.lines(screen, cluster.color, False, points)
