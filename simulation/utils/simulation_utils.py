import collections

VELOCITY_COEFFICIENT = 0.5

Threshold = collections.namedtuple('Threshold', 'position velocity')
Cluster = collections.namedtuple('AgentsPerThreshold', 'color threshold agents')


# get angular velocity proportional to distance from threshold
def get_agent_velocity(threshold_position, agent_position):
    return (agent_position - threshold_position) * VELOCITY_COEFFICIENT