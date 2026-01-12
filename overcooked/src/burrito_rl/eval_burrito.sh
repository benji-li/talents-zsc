#!/bin/bash

#BASE_COMMAND="python -m burrito_rl.main --config eval_br_burrito --mode eval --ckpt_freq 50 --timesteps_total 2000000"

BASE_COMMAND="python -m burrito_rl.main --config /data/benji/strategy-adaptation/overcooked/src/burrito_rl/config/eval_br_over_pop_open_br.yaml --mode eval --ckpt_freq 50 --timesteps_total 2000000"

for i in {1..1}; do
  NAME="open_newrw-$i"
  $BASE_COMMAND --eval_episodes=3 --name=$NAME --save_vid #--save_render --save_vid 
done
