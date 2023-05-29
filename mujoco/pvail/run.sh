#!/bin/bash
for IRL_COEF in 1 50 100 200 400 600 800 1000 1500 2000 3000
do
    python main.py --env_name 'HalfCheetah-v2' --irl_coef $IRL_COEF --max_iter_num 1000
done