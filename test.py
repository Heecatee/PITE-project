from environment import swarmball_env as swarmball

env = swarmball.SwarmBall()
env.reset()
for _ in range(300):
    env.render()
    env.step(1)
