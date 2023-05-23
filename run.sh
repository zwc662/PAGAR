#!/bin/bash
env=$1
task=$2
alg=$3
echo Your container args are: "$@"

cd $env
cd $alg
 
if [ $env == 'minigrid' ]
then 
    python -m scripts.train --env $task &
    cd $
else
    python main.py --env_name $task
fi
