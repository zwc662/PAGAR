# Imitation Learning w/ PAGAR

### 1. Introduction

This repository contains PyTorch (v0.4.1) implementations of **Imitation Learning with Protagonist Antagonist Guided Adversarial Reward (PAGAR)** algorithms.
 
### 2. Installation
* Docker build (Recommended)
   * Install [Docker](https://docs.docker.com/get-docker/)
   * Download docker image tar file from (https://drive.google.com/file/d/15mpNuIkEMhgD5y8SVfQVfe8t5VnaTKL4/view?usp=share_link)
   * Run `docker load --input pagar.tar` to load the docker image.
* Local build
   * Install [Mujoco](https://github.com/openai/mujoco-py)
   * Install other dependencies `pip install -r requirements.txt`
  
### 3. Run Algorithm 
* (If using docker) Run `sudo docker run -it -p 6006:6006 --entrypoint /bin/bash pagar` to open docker's shell.
   * `pagar` is the docker image's name. If the loaded image's name is not `pagar`, please use the name of the loaded image's name
* Set the following environment variables by `export VARIABLE_NAME=VARIABLE_VALUE`
   * `ENV`: specifies the benchmark environment; its variable value can be `minigrid` or `mujoco`
   * `TASK`:  specifies the task
      - If `ENV=minigrid`, then its variable value can be `MiniGrid-DoorKey-6x6-v0`, `MiniGrid-SimpleCrossingS9N1-v0`, `MiniGrid-SimpleCrossingS9N2-v0`, `MiniGrid-SimpleCrossingS9N3-v0` `MiniGrid-FourRooms-v0`.
      - If `ENV=mujoco`, then its variable value can be `Hopper-v2`, `Walker2d-v2`, `HalfCheetah-v2`, `InvertedPendulum-v2`, `Swimmer-v2`
   * `ALG`: specifies the algorithm; its variable value can be `pgail` to obtain protagonist_gail, or `pvail` to obtain protagonist_vail.
   * `DEMOS`: specifies the number of demonstrations (only for minigrid tasks).
* Run script `./run.sh $ENV $TASK $ALG $DEMOS` to train the policies

### 4. Tensorboard

Note that the results of trainings are automatically saved in `logs` folder. TensorboardX is the Tensorboard-like visualization tool for Pytorch. 

To visualize the return/iter or return/frame curve, open the browser and go to the url http://localhost:6006
