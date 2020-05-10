import random
import numpy
import pymunk

import utils.simulation_utils as utils
import utils.generate_map as gen


ELASTICITY = 0.3
FRICTION = 10
BOTS_RADIUS = 4
BOTS_MASS = 30
GOAL_OBJECT_SIZE = (20, 20)
GOAL_OBJECT_MASS = 3
GOAL_OBJECT_FRICTION = 0.01


def create_clusters(number_of_clusters, screen_size, number_of_bots_per_threshold):
    clusters = []
    for _ in range(number_of_clusters):
        color = list(numpy.random.random(size=3) * 256)
        threshold = utils.Threshold(position=random.randint(0, screen_size[0]), velocity=0)
        cluster = utils.Cluster(color, threshold, bots=[])

        for _ in range(number_of_bots_per_threshold):
            cluster.bots.append(create_bot(threshold.position, color, screen_size))

        clusters.append(cluster)

    return clusters


def create_bot(position_x, color, screen_size, max_distance_from_threshold=100):
    min_pos = position_x - max_distance_from_threshold
    max_pos = position_x + max_distance_from_threshold

    radius = BOTS_RADIUS
    mass = BOTS_MASS
    body = pymunk.Body(mass, moment=pymunk.moment_for_circle(mass, inner_radius=0, outer_radius=radius))
    body.position = (random.randint(min_pos, max_pos), screen_size[1])
    shape = pymunk.Circle(body, radius)
    shape.color = color
    shape.elasticity = ELASTICITY
    shape.friction = FRICTION

    return shape


def create_goal_object(position):
    mass = GOAL_OBJECT_MASS
    size = GOAL_OBJECT_SIZE
    inertia = pymunk.moment_for_box(mass, size)
    body = pymunk.Body(mass, inertia)
    body.position = position
    shape = pymunk.Poly.create_box(body, size)
    shape.elasticity = ELASTICITY
    shape.friction = GOAL_OBJECT_FRICTION
    return shape


def create_map_segment(difficulty, space, starting_point, segment_size, map_width):
    map_fragments = gen.generate_map(diff_level=difficulty, x_offset=starting_point[0],
                                     y_offset=starting_point[1], resolution=segment_size)
    fragment_start = map_fragments[0]
    map_segment = []
    for fragment_end in map_fragments[1:]:
        fragment = pymunk.Segment(space.static_body, fragment_start, fragment_end, map_width)
        fragment_start = fragment_end
        fragment.elasticity = ELASTICITY
        fragment.friction = FRICTION
        map_segment.append(fragment)
    segment_end_point = fragment_end
    return map_segment, segment_end_point

