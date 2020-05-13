import random
import math
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
from scipy.interpolate import splprep, splev
from collections import namedtuple

Y_CEILING_LIMIT = 20
LOOP_ELIMINATION_ACCURACY = 3
TAKING_STEP_BACK_PROBABILITY = 0.3


Point = namedtuple('Point', 'x y')

class Map:
    def __init__(self, starting_point, x_offset, resolution=(1280, 720), seed=None):
        self.points_before_interpolation = []
        self.map_points = np.array([])
        self.segments = np.array([])
        self.x_offset = x_offset
        self.y_offset = starting_point.y
        self.append_point_before_interpolation(starting_point)
        self.resolution = resolution
        self.seed = seed

    def append_point_before_interpolation(self, point):
        self.points_before_interpolation.append(point)

    def __getitem__(self, key):
        """Get point of the map: e.g. game_map[7] ---> (x7,y7)"""
        if not len(self.map_points):
            return self.points_before_interpolation[key]
        else:
            return self.map_points[key]

    def get_data_as_points(self):
        """Get map in format [(x0,y0), (x1,y1), (x2,y2), ...]"""
        return self.map_points

    def get_data_as_segments(self):
        """Get map in format [((x1,y1), (x1,y1)), ((x1,y1), (x2,y2)), ((x2,y2), (x3,y3)), ...]"""
        return self.segments

    def get_seed(self):
        """Get the seed of the map"""
        return self.seed

    def get_X_list(self):
        """Get list of x coordinates"""
        if not len(self.map_points):
            xs = [point.x for point in self.points_before_interpolation]
        else:
            xs = self.map_points[:, 0]
        return xs

    def get_Y_list(self):
        """Get list of y coordinates"""
        if not len(self.map_points):
            ys = [point.y for point in self.points_before_interpolation]
        else:
            ys = self.map_points[:, 1]
        return ys

    def __len__(self):
        """ Length of map (points, not segments) """
        if not len(self.map_points):
            return len(self.points_before_interpolation)
        else:
            return len(self.map_points)

    def __delitem__(self, index):
        if not len(self.map_points):
            del self.points_before_interpolation[index]
        else:
            del self.map_points[index]

    def are_points_in_clockwise_order(self, A, B, C):
        return (C.y - A.y) * (B.x - A.x) < (B.y - A.y) * (C.x - A.x)

    def are_lines_intersecting(self, index_of_first_line_starting_point, index_of_second_line_starting_point):
        """Function that checks if two lines intersect"""
        i = index_of_first_line_starting_point
        j = index_of_second_line_starting_point

        if i == j:
            return False

        A = self.points_before_interpolation[i]
        B = self.points_before_interpolation[i + 1]
        C = self.points_before_interpolation[j]
        D = self.points_before_interpolation[j + 1]
        order_1 = self.are_points_in_clockwise_order(A, C, D)
        order_2 = self.are_points_in_clockwise_order(B, C, D)
        order_3 = self.are_points_in_clockwise_order(A, B, C)
        order_4 = self.are_points_in_clockwise_order(A, B, D)

        return order_1 != order_2 and order_3 != order_4

    def delete_map_loops(self, step, angle, points_radius=50):
        """Function that gets rid of most of the loops in map so as to make it a little less crazy"""
        i = 0
        size = len(self.points_before_interpolation)
        while i < size - 1:
            left_limit = i - points_radius if i >= points_radius else 0
            right_limit = i + points_radius if i + points_radius < size else size - 2
            for j in range(left_limit + 1, right_limit):
                j = size - 2 if j >= size - 1 else j
                if self.are_lines_intersecting(i, j):
                    del self.points_before_interpolation[i + 1]
                    point = generate_next_point(self.points_before_interpolation[size - 2], step, angle, self.resolution[1])
                    self.append_point_before_interpolation(point)
            i += 1

    def interpolate(self):
        tck, u = splprep([self.get_X_list(), self.get_Y_list()], s=0)
        x_array = np.linspace(start=0, stop=1, num=self.resolution[0])
        x_array, y_array = splev(x_array, tck, der=0)
        x_array += self.x_offset
        self.map_points = np.vstack((x_array, y_array)).T
        self.segments = list(zip(self.map_points[:-1], self.map_points[1:]))

    def save_to_file(self, filename='test_map.png', fill=False):
        plt.clf()
        plt.xlim(self.x_offset, self.resolution[0] + self.x_offset)
        plt.ylim(0.0, self.resolution[1])
        if fill:
            plt.fill_between(self.get_X_list(), self.get_Y_list())
        plt.plot(self.get_X_list(), self.get_Y_list())
        plt.savefig(filename, dpi=100)

    def show_map(self, fill=False):
        plt.clf()
        plt.xlim(self.x_offset, self.resolution[0] + self.x_offset)
        plt.ylim(0.0, self.resolution[1])
        if fill:
            plt.fill_between(self.get_X_list(), self.get_Y_list())
        plt.plot(self.get_X_list(), self.get_Y_list())
        plt.show()


class Difficulty(Enum):
    """Enum for Difficulty level"""
    PATHETIC = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    REALLY_HARD = 4
    WTF = 5


def generate_next_point(prev_point, step, alpha, y_resolution):
    """Function returning next random point for the generalized map"""
    y_range = 2 * step * math.tan(alpha / 2)

    y_min = prev_point.y - y_range / 2
    y_min = y_min if y_min >= 0 else 0
    y_max = prev_point.y + y_range / 2
    y_max = y_max if y_max < y_resolution - Y_CEILING_LIMIT else y_resolution - Y_CEILING_LIMIT

    p = random.random()
    if p >= TAKING_STEP_BACK_PROBABILITY:
        x = prev_point.x + step
    else:
        x = prev_point.x - step if prev_point.x > step else prev_point.x + step

    y = random.uniform(y_min, y_max)

    next_point = Point(x, y)

    return next_point


def prepare_map_before_interpolation(resolution, step, angle_range, x_offset, y_offset, seed):
    """Function that prepares real map - interpolates random points generated by function next_point
        returns interpolated and smoothed map in X and Y arrays
    """
    x_res = resolution[0]
    y_res = resolution[1]

    y_offset = random.randrange(y_res / 4, y_res * 3 / 5) if not y_offset else y_offset
    point = Point(0.0, y_offset)
    game_map = Map(point, x_offset, resolution, seed)

    while point.x < x_res:
        point = generate_next_point(point, step, angle_range, y_res)
        game_map.append_point_before_interpolation(point)

    for _ in range(LOOP_ELIMINATION_ACCURACY):
        game_map.delete_map_loops(step, angle_range)

    return game_map


def get_level_parameters(diff_level, x_res):
    """Function that prepares parameters according to difficulty level of map
        returns X and Y ready to go
    """
    step = None
    angle_range = None

    if diff_level == Difficulty.PATHETIC:
        step = 2
        angle_range = 0

    elif diff_level == Difficulty.EASY:
        step = x_res / 10
        angle_range = math.pi / 4

    elif diff_level == Difficulty.MEDIUM:
        step = x_res / 20
        angle_range = math.pi * 3 / 4

    elif diff_level == Difficulty.HARD:
        step = x_res / 100
        angle_range = 15 * math.pi / 18

    elif diff_level == Difficulty.REALLY_HARD:
        step = x_res / 120
        angle_range = 16 * math.pi / 18

    elif diff_level == Difficulty.WTF:
        step = x_res / 150
        angle_range = 16 * math.pi / 18

    return step, angle_range


def generate_map(seed=None, diff_level=Difficulty.PATHETIC, x_offset=0, y_offset=None, resolution=(1280, 720)):
    """Function to generate map
    parameters - map seed generated before, difficulty level from Difficulty Enum, starting y position, resolution
    returns data in format [(x1,y1), (x2,y2), .....] as points coordinates
    """
    if seed:
        random.setstate(seed)
    else:
        seed = random.getstate()

    step, angle_range = get_level_parameters(diff_level, resolution[0])
    game_map = prepare_map_before_interpolation(resolution, step, angle_range, x_offset, y_offset, seed)
    game_map.interpolate()

    return game_map
