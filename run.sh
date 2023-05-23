env=$1
task=$2
alg=$3

cd $env
cd $alg
pwd
if [ $env -eq 'minigrid' ]
then 
    python -m scripts.train --env $task
else
    python main.py --env_name $task
fi
