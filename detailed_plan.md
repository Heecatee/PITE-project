# **Swarmball.ai**

DETAILED PLAN

**Project meetings:**

- Stand Up: every **Sunday** and **Wednesday** at **6PM**
- Sprint closure / reflections / new sprint / planning: every **Friday** at **6PM**

**Goal of**  **50%**  **project:**

- Minimal value product - that is, some noticeable effects of learning on our own simple environment. Robots will be balls rolling left and right and a map will be a curly solid ground. Single object to move around.

**Roadmap of**  **50%**  **project:**

_Week 1._

Goal - Introduction to topics vital for project realisation; research of how to manage our goal most efficiently using state-of-the-art methods.

- each team member will prepare a presentation of chosen technology used in the project and present it before the rest of the team.
- technologies used: PyMunk, PyGame, OpenAI gym, PyTorch, Neural Networks, Reinforcement Learning
- additionally, all the members will do an extensive research on how such tasks are being handled in the most modern approaches, as well as the approaches suitable for a smaller project such as ours.

_Week 2 - 3._

Goal - environment to run a basic simulation on, presenting how it works.; policy network; a2c algorithm ready to teach it, tested on substitute environment.

- Create algorithm that generates map
- Create first policy network
- Create interface to manipulate bots on plane map
- Create a2c method from scratch in PyTorch
- Prepare environment structure
- Create objects and physics in pymunk
- Create render it in pygame
- Integrate pymunk with generated maps
- Research and discuss baseline learning policy

_Week 4_

Goal - environment fully prepared to train our nanobots on infinite procedurally generated ground; PPO algorithm; baseline learning policy.

- Modify map generating to generate infinite maps
- Complexify generated maps
- extend a2c to PPO
- Integrate elements using gym.env
- prepare training environment such as Google Collab, GPU driven personal machine, a cluster or any other suitable solution (callbacks, tensorboard, computing unit configuration)

**Goal of**  **100%**  **project:**

- Our wildest dreams coming true! Tuned nanobots, possibly enhanced with some features or abilities allowing them to get through far more difficult and dangerous terrain, filled with various traps and obstacles.

**Roadmap of**  **100%**  **project:**

_Week 5._

Goal - tuned environment, tuned network, ran multiple times to figure out what works the best, trained with our best efforts to make it as good as possible.

_Week 6 - 7._

Goal - harder maps; traps and new movable obstacles added; different policy network architectures tried, best chosen; training and tuning continues.

_Week 8._

Goal - Big online party, beverages delivered by nanobots (beware the deadly laser beam, also training and tuning continues).
