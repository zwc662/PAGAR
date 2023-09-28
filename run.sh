#!/bin/bash
env=$1
task=$2
alg=$3
demos=$4
echo Your container args are: "$@"

cd $env
cd $alg

if [ $env == 'minigrid' ];
then 
    tensorboard --logdir scripts/logs &
    if [ $task != 'MiniGrid-DoorKey-6x6-v0' ];
    then
        python -m scripts.train --env ${task} --no-cuda --entropy --demonstration ../expert_demo/exp_demo_${task}_${demos}ep.p
    elif [ $task != 'MiniGrid-FourRooms-v0' ];
    then
        python -m scripts.train --env ${task} --no-cuda --entropy --demonstration ../expert_demo/exp_demo_SimpleCrossingS9N1-v0.p
    else
        python -m scripts.train --env ${task} --no-cuda --demonstration ../expert_demo/exp_demo_${task}_${demos}ep.p
    fi
    #tensorboard --logdir ./logs 
else
    tensorboard --logdir logs &
    python main.py --env_name ${task}
    #tensorboard --logdir ./logs 
fi
