#!/bin/bash

ALGO="bp"
LAYOUT="open"
# ALGO="mep"
# LAYOUT="open"
BASE_COMMAND="python -m burrito_rl.main --config /data/benji/talents_zsc/overcooked/src/burrito_rl/config/vae_cluster_br_${LAYOUT}_${ALGO}.yaml --mode tune --ckpt_freq 400 --timesteps_total 60000000"

for i in {1..5}; do
  NAME="vaebr_gpuoptim_${LAYOUT}_${ALGO}_2cl_norw_temp1_numhis10-$i"

  CUDA_VISIBLE_DEVICES=2,4,3 $BASE_COMMAND --name=$NAME --model_path="policy_params/$NAME" #--load_model --restore_path="/data/benji/strategy-adaptation/overcooked/src/burrito_rl/policy_params/vaebr_gpuoptim_${LAYOUT}_${ALGO}_norw_temp1_numhis10-$i/0000/checkpoint_001200/policies/polBR/policy_state.pkl"
done