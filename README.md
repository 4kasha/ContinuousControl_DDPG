[//]: # (Image References)

[image1]: media/Reacher_Trained.gif "Trained Agent"

# Continuous Control via DDPG - PyTorch implementation

## Introduction

In this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location (The spherical area in the above video). Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of **33 variables** corresponding to position, rotation, velocity, and angular velocities of the arm. **Each action is a vector with four numbers**, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this project, there are two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience. 
These tasks are episodic, and in order to solve the environment, your agent must get at least an average score of +30 over 100 consecutive episodes. 

## Dependencies

- Python 3.6
- PyTorch 0.4.0
- ML-Agents Beta v0.4

**NOTE** : (_For Windows users_) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

## Getting Started

1. Create (and activate) a new environment with Python 3.6 via Anaconda.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name your_env_name python=3.6
	source activate your_env_name
	```
	- __Windows__: 
	```bash
	conda create --name your_env_name python=3.6 
	activate your_env_name
	```

2. Clone the repository, and navigate to the python/ folder. Then, install several dependencies (see `requirements.txt`).
    ```bash
    git clone https://github.com/4kasha/ContinuousControl_DDPG.git
    cd ContinuousControl_DDPG/python
    pip install .
    ```

3. Download the environment from one of the links below. You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.

    **NOTE** : For this project, you will not need to install Unity. The link above provides you a standalone version. Also the above Reacher environment is similar to, but **not identical to** the Reacher environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher).

4. Place the file in this repository _ContinuousControl_DDPG_ and unzip (or decompress) the file.

## Instructions

- Before running code, change parameters in `train.py`, especially you must change `env_file_name` according to your environment.
- Run the following command to get started with training your own agent!
    ```bash
    python train.py
    ```
- After finishing training weights and scores are saved in the following folder `weights` and `scores` respectively. 


## Tips

- For more details of algolithm description, hyperparameters settings and results, see [REPORT.md](REPORT.md).
- For the examples of training results, see [CC_Results_Example.ipynb](CC_Results_Example.ipynb).
- After training you can test the agent with saved weights in the folder `weights`, see [CC_Watch_Agent.ipynb](CC_Watch_Agent.ipynb). 
- This project is a part of Udacity's [Deep Reinforcement Nanodegree program](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

