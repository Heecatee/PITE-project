# Swarmball.ai
## Documentation
https://docs.google.com/document/d/1d17AlUiGnsbga3XQSN1vrIGNN-0AY6xPdZwqkYeZTHM/edit?usp=sharing

## Colab notebook for training process
https://colab.research.google.com/drive/1zzOHM2ZQ7ESPBAjRknpCUscERr9TqC0n?usp=sharing

## See our experiments live on Neptune
https://ui.neptune.ai/pradon/Swarmball/wiki/README-38a25739-c6f3-4923-94c5-9fb4ac5afaf4

## Participants
* Maria Korkuć
* Wioletta Kurek
* Wojciech Korzybski
* Patryk Radoń (Team Leader)
* Karol Kocierz

## Short description of the idea (for a detailed plan please go [here](https://github.com/Heecatee/Swarmball/blob/master/detailed_plan.md))

Swarmball.ai is a project based on real-life  problem of adapting the robotic tool to the environment when the task is to simply move some objects around. We have noticed, that there may be a more **versatile** approach.

### Go nanobots!

We will be working on a simplified problem using only **two dimensions**. Instead of one robot, we will be using the **swarm of small balls** with an ability to move right or left. The environment will be a bumpy plane full of obstacles, generated randomly or created by an evil human user. Their task is defined as follows: *Move the object to the far right wall.*

### The hive mind

The steering is going to be fully automated with **artificial intelligence**, adapting to any environment you make it run on. It will not only be forced to make the balls move the object, but to do it as quickly as possible. This will be accomplished by chasing the ball with a **Death Ray**, which will increase in speed as the algorithm becomes more effective.


### Technologies used:
  -PyMunk, 
  -PyGame, 
  -OpenAI gym, 
  -PyTorch, 
  -Neural Networks, 
  -Reinforcement Learning
 
### What will we learn?

This project is mainly aimed at introducing everyone involved to the concepts of deep reinforcement learning, in an end-to-end machine learning project. First, we wil learn how to prepare our own environment from  scratch using python wrappers for phisics engines. After that, we will plan out the training using techniques such as a2c and from that we will dive into the  state-of-art PPO. All that will be done using PyTorch - a deep learning framework, currently leading in the field of deep learning research.
The next step will introduce us to a process of tuning and training the model in a fast sparse environments or using our own GPU boosted machines. All that will hopefully lead us to a working product, and most of all, every single one of us will be able to say that they truely made an Artificial Intelligence.

### Preview
![](https://i.imgur.com/waH6dxF.gif)
