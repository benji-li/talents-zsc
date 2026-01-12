#!/bin/bash

BASE_COMMAND="python -m burrito_rl.main --config /data/benji/strategy-adaptation/overcooked/src/burrito_rl/config/br_open_bp.yaml --mode tune --ckpt_freq 30 --timesteps_total 60000000"

for i in {1..1}; do
  NAME="bpbr_gpuoptim_open_norw_pop1-$i"
  $BASE_COMMAND --name=$NAME --model_path="policy_params/$NAME" --load_model --restore_path="/data/benji/strategy-adaptation/overcooked/src/burrito_rl/policy_params/bpbr_gpuoptim_open_norw_pop1-1/0000/checkpoint_001530/policies/polBR/policy_state.pkl" #--logger="wandb" --wandb_project="Population"
done
