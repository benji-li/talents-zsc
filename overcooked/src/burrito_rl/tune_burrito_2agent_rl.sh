#!/bin/bash
SRC=$(pwd)
FILE="$SRC/burrito_rl/config/burrito_.yaml"

if [ ! -f "$FILE" ]; then
  echo "Error: $FILE does not exist."
  exit 1
fi

BASE_COMMAND="python -m burrito_rl.main --config burrito_ --mode tune --ckpt_freq 15 --timesteps_total 3000000"

LRS=(0.001)
TRAIN_BATCH_SIZES=(38400)
NUM_SGD_ITERS=(4 2)
# GAMMAS=(0.999 0.99)
ENTROPY_COEFFS=(0.04 0.01)
# 2 x 2 x 2 = 8

for i in {1..3}; do
  for lr in "${LRS[@]}"; do
    for train_batch_size in "${TRAIN_BATCH_SIZES[@]}"; do
      for num_sgd_iter in "${NUM_SGD_ITERS[@]}"; do
        for entropy_coeff in "${ENTROPY_COEFFS[@]}"; do
          yq -i -y ".BASE_CONFIG.entropy_coeff = $entropy_coeff" "$FILE"
          yq -i -y ".BASE_CONFIG.num_sgd_iter = $num_sgd_iter" "$FILE"
          yq -i -y ".BASE_CONFIG.lr = $lr" "$FILE"
          yq -i -y ".BASE_CONFIG.train_batch_size = $train_batch_size" "$FILE"

          NAME="burrito_2ag_nopos_steakmush_27act_entropy${entropy_coeff}_iter${num_sgd_iter}_trial$i"
          $BASE_COMMAND --name=$NAME --model_path="policy_params/$NAME"
        done
      done
    done
  done
done

