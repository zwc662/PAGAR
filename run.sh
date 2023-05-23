#!/bin/bash
env=$1
task=$2
alg=$3
echo Your container args are: "$@"

cd $env
cd $alg
tensorboard --logdir $env/$alg/logs &

if [ $env == 'minigrid' ];
then 
    if [ $task != 'MiniGrid-DoorKey-6x6-v0' ];
    then
        python -m scripts.train --env $task --no-cuda --entropy
    else
        python -m scripts.train --env $task --no-cuda
    fi
    #tensorboard --logdir ./logs 
else
    python main.py --env_name $task
    #tensorboard --logdir ./logs 
fi
