#!/bin/bash

ALGO="bp"
LAYOUT="fc"

BASE_COMMAND="python -m burrito_rl.main --config /data/benji/strategy-adaptation/overcooked/src/burrito_rl/config/gamma_br_${LAYOUT}_${ALGO}.yaml --mode tune --ckpt_freq 30 --timesteps_total 60000000"

for i in {3..3}; do
  NAME="gammabr_gpuoptim_${LAYOUT}_${ALGO}_norw-$i"
  CUDA_VISIBLE_DEVICES=2,3,4 $BASE_COMMAND --name=$NAME --model_path="policy_params/$NAME" #--logger="wandb" --wandb_project="Population"
done
