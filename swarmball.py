from policy_network.HiveNet import HiveNet
from policy_network.hive_vision.HiveNetVision import HiveNetVision
from environment.swarmball_env import SwarmBall
from a2c.a2c import A2CTrainer
# import neptune



NUM_OF_EPISODES = 3
RENDER_INTERVAL = 1
best_result = 0
in_which_episode = 0

hivenet = HiveNet(kernel_size=3,
                  stride=2,
                  num_of_thresholds=3)


swarmball_env = SwarmBall()
swarmball_env.reset()
# obs, _, _, _ = swarmball_env.step([1,0,1])
# HV = HiveNetVision(kernel_size=3, stride=2, outputs=10)
# print(HV.forward(map_image=obs['picture']))

trainer = A2CTrainer(net=hivenet,
                     out_num=3,
                     environment=swarmball_env,
                     batch_size=12,
                     gamma=0.99,
                     beta_entropy=0.001,
                     learning_rate=0.01,
                     clip_size=0.2)

for episode in range(NUM_OF_EPISODES):
    if episode % RENDER_INTERVAL == 0:
        trainer.data.render = True
    else:
        trainer.data.render = False

    curr_result, _ = trainer.train()

    if curr_result > best_result:
        best_result = curr_result
        in_which_episode = episode
    print(
        f'{episode}. {curr_result}\tBest: {best_result} in episode {in_which_episode}')