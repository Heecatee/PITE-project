
VELOCITY_COEFFICIENT = 0.5


# get angular velocity proportional to distance from threshold
def get_agent_velocity(threshold_position, agent_position):
    return (agent_position - threshold_position) * VELOCITY_COEFFICIENT