#!/bin/bash

ALGO="bp"
LAYOUT="open_3p"
# ALGO="mep"
# LAYOUT="open"
#BASE_COMMAND="python -m burrito_rl.main --config /data/benji/strategy-adaptation/overcooked/src/burrito_rl/config/vae_cluster_br_${LAYOUT}_${ALGO}_11cl.yaml --mode tune --ckpt_freq 300 --timesteps_total 60000000"

BASE_COMMAND="python -m burrito_rl.main --config /data/benji/strategy-adaptation/overcooked/src/burrito_rl/config/vae_cluster_br_${LAYOUT}_${ALGO}.yaml --mode tune --ckpt_freq 300 --timesteps_total 60000000"

for i in {2..2}; do
  NAME="vaebr_DEMO-$i"
  CUDA_VISIBLE_DEVICES=5,6,2 $BASE_COMMAND --name=$NAME --model_path="policy_params/$NAME" --load_model --restore_path="/data/benji/strategy-adaptation/overcooked/src/burrito_rl/policy_params/vaebr_DEMO-$i/0000/checkpoint_000900/policies/polBR/policy_state.pkl" #--logger="wandb" --wandb_project="Population"
done