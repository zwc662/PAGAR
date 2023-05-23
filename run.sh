env=$1
task=$2
alg=$3

cd $env
cd $alg
if [ $env -ge 'minigrid' ]
then 
    python -m scripts.train --env $task
else
    python main.py --env_name $task
end
    
