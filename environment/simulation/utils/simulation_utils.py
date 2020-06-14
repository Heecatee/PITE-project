import collections

VELOCITY_COEFFICIENT = 0.85
MAX_BOT_VELOCITY = 40
MIN_BOT_VELOCITY = -40


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
    bot_velocity = (bot_position - threshold_position) * VELOCITY_COEFFICIENT
    if bot_velocity > MAX_BOT_VELOCITY:
        bot_velocity = MAX_BOT_VELOCITY
    elif bot_velocity < MIN_BOT_VELOCITY:
        bot_velocity = MIN_BOT_VELOCITY
    return bot_velocity
