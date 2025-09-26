# CRAFT: Coaching Reinforcement Learning Autonomously using Foundation Models for Multi-Robot Coordination Tasks

[[arXiv]](https://arxiv.org/abs/2509.14380)
[[Project Website]](https://iconlab.negarmehr.com/CRAFT/)

CRAFT is a a framework that leverages the reasoning capabilities of foundation models to act as a “coach” for multi-robot coordination. CRAFT consist of 5 key stages:

* Curriculum generation: A curriculum LLM decomposes the long-horizon coordination task into a sequence of subtasks
* Reward function generation: A reward generation LLM generates a reward function in executable python code based on the natural language descriptions of a subtask.
* Policy evaluation: An evaluation VLM evaluates the success or failure of the policy trained with the LLM-generated reward.
* Reward refinement: If the policy fails to achieve the desired behavior, an advice VLM provides advice on how to change the reward.
* Sequential training of subtasks: we initialize each subtask with the policy learned from the previous one while motivating exploration to learn the new subtask.

## Introduction ##

Learning coordinated behaviors remains a significant challenge in multi-robot systems. Multi-Agent Reinforcement Learning (MARL) offers a promising
framework, but applying MARL to robotics is notoriously difficult due to the high-dimensional action spaces, complex reward design, and the non-stationarity introduced by decentralized decision-making.

To solve this problem, curriculum generation has been studied as an effective approach for learning coordination and strategies in MARL domains by structuring
the training into stages of increasing complexity. However, curriculum design is often non-trivial as it requires domain knowledge to identify key steps
in long-horizon tasks and reasoning ability to monitor the learning progress.

Our key idea is to use human-like reasoning capabilities of foundation models, such as LLM or VLM, as a *coach*, the entity that teaches agents how to coordinate.
Coaching naturally integrates all these skills: a good coach breaks down the task, defines success criteria for each subtask, and
provides actionable feedback to guide improvement. Similarily, CRAFT produces curricula for long-horizon coordination tasks, designs semantically rich reward functions, and
evaluates task–policy alignment using visual information.


## Installation ##

We will announce our code after refactoring.

## Acknowledgement ##

* Our bimanual manipulation environment is from [Robosuite](https://robosuite.ai)
* Our quadruped navigation environment is from [MQE](https://github.com/ziyanx02/multiagent-quadruped-environment)
* We use [SKRL](https://skrl.readthedocs.io/en/latest/) and [OpenRL](https://github.com/OpenRL-Lab/openrl) for MAPPO implementation.