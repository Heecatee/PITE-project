import collections

VELOCITY_COEFFICIENT = 0.5


class Threshold:
    def __init__(self, position=0, velocity=0):
        self.position = position
        self.velocity = velocity


class Cluster:
    def __init__(self, color, threshold, bots):
        self.color = color
        self.threshold = threshold
        self.bots = bots


# get angular velocity proportional to distance from threshold
def get_bot_velocity(threshold_position, bot_position):
    return (bot_position - threshold_position) * VELOCITY_COEFFICIENT