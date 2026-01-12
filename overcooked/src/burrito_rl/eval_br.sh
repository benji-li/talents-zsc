#!/bin/bash

BASE_COMMAND="python -m burrito_rl.main --config eval_br --mode eval --ckpt_freq 50 --timesteps_total 12000000"

for i in {1..1}; do
  NAME="eval-$i"
  $BASE_COMMAND --name=$NAME --eval_episodes 30 #--save_vid #--model_path="policy_params/$NAME" #--logger="wandb" --wandb_project="Population"
done
