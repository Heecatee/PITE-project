import environment.swarmball_env as swarmball

env = swarmball.SwarmBall()
for _ in range(300):
    env.render()
    env.step(1)
