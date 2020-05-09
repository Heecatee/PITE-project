import random
import math
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
from scipy.interpolate import splprep, splev

# ************
class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return '({x},{y})'.format(x=self.x, y=self.y)

    def getPoint(self, x_offset=0):
        point = (self.x + x_offset, self.y)
        return point

    def __getitem__(self, key):
        return self.x if not key else self.y


class Map:
    def __init__(self, starting_point, x_offset, resolution=(1280, 720), seed=None):
        self.points_before_interpolation = []
        self.map_points = []
        self.segments = []
        self.x_offset = x_offset
        self.append(starting_point)
        self.resolution = resolution
        self.seed = seed

    # ************
    def append(self, point):
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

    def X(self):
        """Get list of x coordinates"""
        xs = [point.x for point in self.points_before_interpolation] if not len(self.map_points) else [point[0] for point in self.map_points]
        return xs

    def Y(self):
        """Get list of y coordinates"""
        ys = [point.y for point in self.points_before_interpolation] if not len(self.map_points) else [point[1] for point in self.map_points]
        return ys

    def __len__(self):
        """ Length of map (points, not segments) """
        return len(self.points_before_interpolation) if not len(self.map_points) else len(self.map_points)

    # ************
    def __delitem__(self, index):
        if not len(self.map_points):
            del self.points_before_interpolation[index]
        else:
            del self.map_points[index]

    # ************
    def is_crossing(self, i, j):
        """Function that checks if two lines ((X[i],Y[i]), (X[i+1], Y[i+1])), ((X[j],Y[j]), (X[j+1],Y[j+1])) intersect"""
        if i == j:
            return False
        try:
            return intersect(self.points_before_interpolation[i], self.points_before_interpolation[i + 1],
                             self.points_before_interpolation[j], self.points_before_interpolation[j + 1])
        except IndexError:
            print(f'i = {i}, j = {j}, size = {len(self)}')

    # ************
    def exterminate_loops(self, step, y_res, ran=40):
        """Function that gets rid of most of the loops in map so as to make it a little less crazy"""
        i = 0
        size = len(self.points_before_interpolation)
        while i < size - 1:
            i = size - 2 if i >= size - 1 else i
            left_limit = i - ran if i >= ran else 0
            right_limit = i + ran if i + ran < size else size - 2
            for j in range(left_limit + 1, right_limit):
                j = size - 2 if j >= size - 1 else j
                if self.is_crossing(i, j):
                    del self.points_before_interpolation[i + 1]
                    point = next_point(self.points_before_interpolation[size - 2], step, math.pi * 3 / 4, y_res)
                    self.append(point)
            i += 1

    # ************
    def interpolate(self):
        tck, u = splprep([self.X(), self.Y()], s=0)
        x_array = np.linspace(0, 1, self.resolution[0])
        x_array, y_array = splev(x_array, tck, der=0)
        x_array += self.x_offset
        self.map_points = np.vstack((x_array, y_array)).T
        self.segments = [(self.map_points[i-1], self.map_points[i]) for i in range(1, len(self))]

    def save_to_file(self, filename='test_map.png', fill=False):
        plt.clf()
        plt.figure(figsize=(self.resolution[0] / 100, self.resolution[1] / 100))
        plt.xlim(self.x_offset, self.resolution[0] + self.x_offset)
        plt.ylim(0.0, self.resolution[1])
        if fill:
            plt.fill_between(self.X(), self.Y())
        plt.plot(self.X(), self.Y())
        plt.savefig(filename, dpi=100)

    def show_map(self, fill=False):
        plt.clf()
        plt.figure(figsize=(self.resolution[0] / 100, self.resolution[1] / 100))
        plt.xlim(self.x_offset, self.resolution[0] + self.x_offset)
        plt.ylim(0.0, self.resolution[1])
        if fill:
            plt.fill_between(self.X(), self.Y())
        plt.plot(self.X(), self.Y())
        plt.show()


class Difficulty(Enum):
    """Enum for Difficulty level"""
    PATHETIC = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    REALLY_HARD = 4
    WTF = 5

# ************
def intersect(A, B, C, D):
    """Helper of is_crossing function
    Returns True if line segments AB and CD intersect"""
    ccw = lambda A, B, C: (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# ************
def next_point(prev_point, step, alpha, y_res):
    """Function returning next random point for the generalized map"""
    y_range = 2 * step * math.tan(alpha / 2)
    ymin = prev_point.y - y_range / 2
    ymin = ymin if ymin >= 0 else 0
    ymax = prev_point.y + y_range / 2
    ymax = ymax if ymax < y_res - 20 else y_res - 20
    p = random.random()
    if p >= 0.3:
        x = prev_point.x + step
    else:
        x = prev_point.x - step if prev_point.x > step else prev_point.x + step
    y = random.uniform(ymin, ymax)
    point = Point(x, y)
    return point

# ************
def prepare_map(resolution, step, angle_range, x_offset, y_offset, seed):
    """Function that prepares real map - interpolates random points generated by function next_point
        returns interpolated and smoothed map in X and Y arrays
    """
    x_res = resolution[0]
    y_res = resolution[1]

    x = 0
    y = random.randrange(y_res / 4, y_res * 3 / 5) if not y_offset else y_offset
    point = Point(x, y)
    game_map = Map(point, x_offset, resolution, seed)

    while point.x < x_res:
        point = next_point(point, step, angle_range, y_res)
        game_map.append(point)

    for _ in range(3):
        game_map.exterminate_loops(step, y_res)

    game_map.interpolate()
    return game_map

# ************
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
    game_map = prepare_map(resolution, step, angle_range, x_offset, y_offset, seed)

    return game_map



# USE EXAMPLES
# map1 = generate_map()
# map1.save_to_file()
# map2 = generate_map(diff_level = Difficulty.REALLY_HARD)
# map3 = generate_map(diff_level = Difficulty.EASY)
# map4 = generate_map(diff_level = Difficulty.HARD, y_offset = 120)
# map2.save_to_file()
# map5 = generate_map(map4.get_seed(), diff_level = Difficulty.HARD, x_offset= 1000, y_offset = 120)
