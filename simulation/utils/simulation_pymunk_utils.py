import collections
import random

import numpy
import pymunk

import utils.simulation_utils as utils
import utils.generate_map as gen

Color = collections.namedtuple('RGB', 'r g b')

ELASTICITY = 0.3
FRICTION = 10
AGENTS_RADIUS = 4
AGENTS_MASS = 30
GOAL_OBJECT_MASS = 3

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 540


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
    shape.friction = FRICTION-9.99
    return shape


def create_map_segment(difficulty, space, starting_point, segment_size):
    map_fragments, _ = gen.generate_map(diff_level=difficulty, x_offset=starting_point[0],
                                        starting_y=starting_point[1], resolution=segment_size)
    fragment_start = map_fragments[0]
    map_segment = []
    for fragment_end in map_fragments[1:]:
        fragment = pymunk.Segment(space.static_body, fragment_start, fragment_end, 2)
        fragment_start = fragment_end
        fragment.elasticity = ELASTICITY
        fragment.friction = FRICTION
        map_segment.append(fragment)
    segment_end_point = fragment_end
    return map_segment, segment_end_point

