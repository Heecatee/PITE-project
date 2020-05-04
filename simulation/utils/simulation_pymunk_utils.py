import collections
import random

import numpy
import pymunk

import simulation_utils as utils

Color = collections.namedtuple('RGB', 'r g b')

ELASTICITY = 0
FRICTION = 2
AGENTS_RADIUS = 12
AGENTS_MASS = 10
GOAL_OBJECT_MASS = 10

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720


def create_clusters(number_of_clusters, number_of_agents_per_threshold):
    clusters = []
    for _ in range(number_of_clusters):
        color = list(numpy.random.random(size=3) * 256)
        threshold = utils.Threshold(position=random.randint(0, SCREEN_WIDTH), velocity=0)
        cluster = utils.Cluster(color, threshold, agents=[])

        for _ in range(number_of_agents_per_threshold):
            cluster.agents.append(create_agent(threshold.position, color))

        clusters.append(cluster)

    return clusters


def create_agent(position_x, color, max_distance_from_threshold=100):
    min_pos = position_x - max_distance_from_threshold
    max_pos = position_x + max_distance_from_threshold

    radius = AGENTS_RADIUS
    mass = AGENTS_MASS
    body = pymunk.Body(mass, moment=pymunk.moment_for_circle(mass, inner_radius=0, outer_radius=radius))
    body.position = random.randint(min_pos, max_pos), 400
    shape = pymunk.Circle(body, radius)
    shape.color = color
    shape.elasticity = ELASTICITY
    shape.friction = FRICTION

    return shape


def create_goal_object(position_x):
    mass = GOAL_OBJECT_MASS
    size = (30, 30)
    inertia = pymunk.moment_for_box(mass, size)
    body = pymunk.Body(mass, inertia)
    body.position = position_x, 380
    shape = pymunk.Poly.create_box(body, size)
    shape.elasticity = ELASTICITY
    shape.friction = FRICTION
    return shape


def create_map_segments(map_segments, space):
    segments = []
    for (a, b) in map_segments:
        segment = pymunk.Segment(space.static_body, a, b, 1)
        segment.elasticity = ELASTICITY
        segment.friction = FRICTION
        segments.append(segment)
    return segments
