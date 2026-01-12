#!/bin/bash

ALGO="mep"
LAYOUT="fc"
BASE_COMMAND="python -m burrito_rl.main --config /data/benji/strategy-adaptation/overcooked/src/burrito_rl/config/br_${LAYOUT}_${ALGO}.yaml --mode tune --ckpt_freq 300 --timesteps_total 60000000"

for i in {3..3}; do
  NAME="FINAL_bpbr_gpuoptim_${LAYOUT}_${ALGO}_norw-$i"
  #NAME="burrito_3p-$i"
  CUDA_VISIBLE_DEVICES=3,5,2 $BASE_COMMAND --name=$NAME --model_path="policy_params/$NAME" #--load_model --restore_path="/data/benji/strategy-adaptation/overcooked/src/burrito_rl/policy_params/bpbr_gpuoptim_${LAYOUT}_bp_norw-1/0000/checkpoint_001140/policies/polBR/policy_state.pkl"
done

