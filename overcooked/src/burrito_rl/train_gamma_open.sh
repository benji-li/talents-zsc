#!/bin/bash

ALGO="mep"
LAYOUT="hallway"

BASE_COMMAND="python -m burrito_rl.main --config /data/benji/strategy-adaptation/overcooked/src/burrito_rl/config/gamma_br_${LAYOUT}_${ALGO}.yaml --mode tune --ckpt_freq 300 --timesteps_total 60000000"

for i in {2..2}; do
  NAME="FINAL_gammabr_gpuoptim_${LAYOUT}_${ALGO}_norw-$i"
  CUDA_VISIBLE_DEVICES=1,2,3 $BASE_COMMAND --name=$NAME --model_path="policy_params/$NAME" --load_model --restore_path="/data/benji/strategy-adaptation/overcooked/src/burrito_rl/policy_params/FINAL_gammabr_gpuoptim_hallway_mep_norw-2/0000/checkpoint_000360/policies/polBR/policy_state.pkl"
done