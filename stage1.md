# TASKS FOR STAGE 1

### TASK 1 (Karol & Wiola)

1. Build an Open AI environment:

	1. Cart-pole environment to modify:
  	
	>https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
	2. \_\_init\_\_():

	>parameters (patrz PITEEE->Plan Projektu)
	3. step():
	
	>NeuralNet output (patrz PITEEE->Plan Projektu) -> simulate pymunk step
	>implement DEATH FRY
	4. render():

	>pygame or https://github.com/openai/gym/blob/master/gym/envs/classic_control/rendering.py


### TASK 2 (Wojtek & Patryk)
1. Build the Agent on cart-pole environment

	1. general tutorial and example:
	
	>https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial?hl=PL
	2. tf.agents to choose from:
	
	>https://github.com/tensorflow/agents/tree/master/tf_agents/agents
	3. about agents:

	>https://www.reddit.com/r/reinforcementlearning/comments/8w7mn2/comparison_selection_of_rl_algorithms_in/

### TASK 3 (Maria)
1. build Random Map Generator function with parameterized "difficulty level".

	The maps will be a composition of randomly generated fluctuations and obstacles (Noise-like up and down bumps, with the increasing amplitude, and an increasingly diagonal nature. That will make it go from small hills and valleys to a structure simmilar to a Worms Armaggedon map)
	
<p align="center">
  <img src="https://i.imgur.com/aPxkHBW.png">
</p>



# STAGE 1
**Step 1.** Environment with random map within given difficulty
>STEP1 = TASK3 + TASK1

**Step 2.** Replace car-pole environment with STEP1
>STEP2 = STEP1 + TASK2


**Step 3.** Train and evaluate in for_loop for different parameter
