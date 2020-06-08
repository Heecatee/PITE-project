import random
import math
import matplotlib.pyplot as plt
from enum import IntEnum
import numpy as np
from scipy.interpolate import splprep, splev
from collections import namedtuple

IS_BEHIND_PREVIOUS_POINT = False
IS_ABOVE_PREVIOUS_POINT = False

Y_CEILING_LIMIT = 20
LOOP_ELIMINATION_ACCURACY = 1
TAKING_STEP_BACK_PROBABILITY = 0.3
INTERPOLATION_K = 3

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
        """Get map in format [((x1,y1), (x1,y1)), ((x1,y1), (x2,y2)), ...]"""
        return self.segments

    def get_seed(self):
        """Get the seed of the map"""
        return self.seed

    def get_X_list(self):
        """Get list of x coordinates"""
        if self.map_points.size == 0:
            xs = [point.x for point in self.points_before_interpolation]
        else:
            xs = self.map_points[:, 0]
        return xs

    def get_Y_list(self):
        """Get list of y coordinates"""
        if self.map_points.size == 0:
            ys = [point.y for point in self.points_before_interpolation]
        else:
            ys = self.map_points[:, 1]
        return ys

    def __len__(self):
        """ Length of map (points, not segments) """
        if self.map_points.size == 0:
            return len(self.points_before_interpolation)
        else:
            return len(self.map_points)

    def __delitem__(self, index):
        if self.map_points.size == 0:
            del self.points_before_interpolation[index]
        else:
            del self.map_points[index]

    def are_points_in_clockwise_order(self, A, B, C):
        return (C.y - A.y) * (B.x - A.x) < (B.y - A.y) * (C.x - A.x)

    def are_lines_intersecting(self, index_of_first_line_starting_point, index_of_second_line_starting_point):
        """Function that checks if two lines ((X[i],Y[i]), (X[i+1], Y[i+1])), ((X[j],Y[j]), (X[j+1],Y[j+1])) intersect"""
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

    def delete_map_loops(self, step, points_radius=50, filling_points_angle=math.pi/4):
        """Function that gets rid of most of the loops in map so as to make it a little less crazy"""
        size = len(self.points_before_interpolation)
        indexes_of_points_to_delete = []

        for i in range(1, size - 1):
            if len(indexes_of_points_to_delete) >= size - (INTERPOLATION_K + 1):
                break
            left_limit = i - points_radius if i >= points_radius else 0
            right_limit = i + points_radius if i + points_radius < size else size - 2
            for j in range(left_limit + 1, right_limit):
                if self.are_lines_intersecting(i, j):
                    indexes_of_points_to_delete.append(i)
                    break

        indexes_of_points_to_delete.sort(reverse=True)
        for index in indexes_of_points_to_delete:
            del self.points_before_interpolation[index]

        diff = self.resolution[0] - self.points_before_interpolation[-1].x
        points_to_add = []

        while diff > 0:
            step2 = step if diff > step else diff
            prev = self.points_before_interpolation[-1] if not len(points_to_add) else points_to_add[-1]
            point = generate_next_point(
                prev_point=prev,
                step=step2,
                alpha=filling_points_angle,
                y_resolution=self.resolution[1])
            points_to_add.append(point)
            diff -= step2

        for point in points_to_add:
            self.append_point_before_interpolation(point)

    def interpolate(self):
        tck, u = splprep([self.get_X_list(), self.get_Y_list()], s=0, k=INTERPOLATION_K)
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


class Difficulty(IntEnum):
    """Enum for Difficulty level"""
    PATHETIC = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    REALLY_HARD = 5
    WTF = 6


def generate_next_point(prev_point, step, alpha, y_resolution):
    """Function returning next random point for the generalized map"""
    global IS_ABOVE_PREVIOUS_POINT
    global IS_BEHIND_PREVIOUS_POINT
    y_range = 2 * step * math.tan(alpha / 2)

    y_min = prev_point.y - y_range / 2
    y_min = y_min if y_min >= 0 else 0
    y_max = prev_point.y + y_range / 2
    y_max = y_max if y_max < y_resolution - Y_CEILING_LIMIT else y_resolution - Y_CEILING_LIMIT

    if IS_BEHIND_PREVIOUS_POINT:
        if IS_ABOVE_PREVIOUS_POINT:
            y_min = prev_point.y + 5
        else:
            y_max = prev_point.y - 5


    p = random.random()
    if p >= TAKING_STEP_BACK_PROBABILITY or IS_BEHIND_PREVIOUS_POINT:
        x = prev_point.x + step
        IS_BEHIND_PREVIOUS_POINT = False
    else:
        x = prev_point.x - step if prev_point.x > step else prev_point.x + step
        IS_BEHIND_PREVIOUS_POINT = True

    y = random.uniform(y_min, y_max)
    IS_ABOVE_PREVIOUS_POINT = True if y > prev_point.y else False

    next_point = Point(x, y)

    return next_point


def prepare_map_before_interpolation(resolution, step, angle_range, x_offset, y_offset, seed):
    """Function that prepares real map - interpolates random points generated by function next_point
        returns interpolated and smoothed map in X and Y arrays
    """
    x_res = resolution[0]
    y_res = resolution[1]

    y_offset = random.randrange(y_res / 4, y_res * 3 / 5) if y_offset == None else y_offset
    point = Point(0.0, y_offset)
    game_map = Map(point, x_offset, resolution, seed)

    while point.x < x_res:
        point = generate_next_point(point, step, angle_range, y_res)
        game_map.append_point_before_interpolation(point)

    for _ in range(LOOP_ELIMINATION_ACCURACY):
        if len(game_map) <= 10:
            break
        game_map.delete_map_loops(step, filling_points_angle=angle_range)

    return game_map


def get_level_parameters(diff_level, x_res):
    """Function that prepares parameters according to difficulty level of map
        returns X and Y ready to go
    """
    step = None
    angle_range = None

    if diff_level == Difficulty.PATHETIC:
        step = x_res / 3
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
        step = x_res / 150
        angle_range = 16 * math.pi / 18

    elif diff_level == Difficulty.WTF:
        step = x_res / 180
        angle_range = 17 * math.pi / 18

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
