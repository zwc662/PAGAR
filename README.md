# Let's do Inverse RL

## Introduction

This repository contains PyTorch (v0.4.1) implementations of **Imitation Learning with Protagonist Antagonist Guided Adversarial Reward (PAGAR)** algorithms.
 
### 2. Installation

* Install [Mujoco](https://github.com/openai/mujoco-py)
* Install [MiniGrid](https://github.com/Farama-Foundation/Minigrid/tree/gym-minigrid-legacy)
  
### 3. Train & Test

#### MiniGrid

* Run GAIL w/ PAGAR to obtain `Protagonist_GAIL`

   * Navigate to [minigrid/pgail-starter-files](https://github.com/zwc662/PAGAR/tree/main/minigrid/pgail-starter-files) folder.
 
* Or run VAIL w/ PAGAR to obtain `Protagonist_GAIL`

   * Navigate to [minigrid/pvail-starter-files](https://github.com/zwc662/PAGAR/tree/main/minigrid/pvail-starter-files) folder.

* Run the following command with $ENV_NAME being `MiniGrid-DoorKey6x6-v0` or `Minigrid-SimpleCrossingS9N1/2/3-v0`

   ~~~
   python -m scripts.train --env $ENV_NAME --no-cuda
   ~~~

#### Mujoco

* Run GAIL w/ PAGAR to obtain `Protagonist_GAIL`

   * Navigate to [mujoco/pgail](https://github.com/zwc662/PAGAR/tree/main/mujoco/pgail) folder.
 
* Or run VAIL w/ PAGAR to obtain `Protagonist_GAIL`

   * Navigate to [mujoco/pvail](https://github.com/zwc662/PAGAR/tree/main/mujoco/pvail) folder.

* Run the following command with $ENV_NAME being `Hopper-v2` or `HalfCheetah-v2`, etc.

   ~~~
   python main.py --env_name $ENV_NAME

### 4. Tensorboard

Note that the results of trainings are automatically saved in `logs` folder. TensorboardX is the Tensorboard-like visualization tool for Pytorch.

For example, navigate to the `mujoco/pgail` or `minigrid/pgail-starter-files` folder to visualize the training process of `Protagonist_GAIL`.

~~~
tensorboard --logdir logs
~~~
 
