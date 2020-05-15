import collections

VELOCITY_COEFFICIENT = 0.5

Threshold = collections.namedtuple('Threshold', 'position velocity')
Cluster = collections.namedtuple('AgentsPerThreshold', 'color threshold bots')


# get angular velocity proportional to distance from threshold
def get_bot_velocity(threshold_position, bot_position):
    return (bot_position - threshold_position) * VELOCITY_COEFFICIENT