import collections
import random

import numpy
import pygame
import pymunk

Color = collections.namedtuple('RGB', 'r g b')
Cluster = collections.namedtuple('AgentsPerThreshold', 'threshold_position agents color')

ELASTICITY = 0
FRICTION = 2
COLLISION_TYPE_NONE = 0
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720


def create_clusters(thresholds_positions_x, number_of_agents_per_threshold):
    clusters = []
    for pos in thresholds_positions_x:
        color = list(numpy.random.random(size=3) * 256)
        cluster = Cluster(pos, [], color)
        for _ in range(number_of_agents_per_threshold):
            cluster.agents.append(create_agent(pos, color))
        clusters.append(cluster)
    return clusters


def create_agent(position_x, color, max_distance_from_threshold=100):
    min_pos = position_x - max_distance_from_threshold
    max_pos = position_x + max_distance_from_threshold

    radius = 12
    mass = 10
    body = pymunk.Body(mass, moment=pymunk.moment_for_circle(mass, inner_radius=0, outer_radius=radius))
    body.position = random.randint(min_pos, max_pos), 400
    shape = pymunk.Circle(body, radius)
    shape.color = color
    shape.elasticity = ELASTICITY
    shape.friction = FRICTION

    return shape


def create_threshold(position_x, color):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    shape = pymunk.Segment(body, (position_x, 0), (position_x, SCREEN_HEIGHT), 1)
    shape.elasticity = ELASTICITY
    shape.friction = FRICTION
    shape.collision_type = COLLISION_TYPE_NONE
    shape.color = color
    return shape


def create_goal_object(position_x):
    mass = 10
    size = (40, 40)
    inertia = pymunk.moment_for_box(mass, size)
    body = pymunk.Body(mass, inertia)
    body.position = position_x, 380
    shape = pymunk.Poly.create_box(body, size)
    shape.elasticity = ELASTICITY
    shape.friction = FRICTION
    return shape