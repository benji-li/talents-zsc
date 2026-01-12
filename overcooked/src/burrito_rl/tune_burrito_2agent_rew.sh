#!/bin/bash
SRC=$(pwd)
FILE="$SRC/burrito_rl/config/burrito_.yaml"

if [ ! -f "$FILE" ]; then
  echo "Error: $FILE does not exist."
  exit 1
fi

BASE_COMMAND="python -m burrito_rl.main --config burrito_ --mode tune --ckpt_freq 15 --timesteps_total 3000000"

for i in {1..3}; do
  yq -i -y ".ENV_CONFIG.rew_shaping_params.wrong_dish_discount = 0.1" "$FILE"
  yq -i -y ".ENV_CONFIG.rew_shaping_params.correct_dish_bonus = 2" "$FILE"
  yq -i -y ".ENV_CONFIG.rew_shaping_params.multiplier_params = [[0, 1], [1,10], [2, 20], [3, 30], [100, 40]]" "$FILE"

  NAME="burrito_2ag_nopos_steakmush_27act_deliveryshape_trial$i"
  $BASE_COMMAND --name=$NAME --model_path="policy_params/$NAME"

  yq -i -y ".ENV_CONFIG.rew_shaping_params.wrong_dish_discount = 1" "$FILE"
  yq -i -y ".ENV_CONFIG.rew_shaping_params.correct_dish_bonus = 1" "$FILE"
  yq -i -y ".ENV_CONFIG.rew_shaping_params.multiplier_params = [[0, 1], [1, 30], [3, 40], [100, 50]]" "$FILE"

  NAME="burrito_2ag_nopos_steakmush_27act_multiplier_trial$i"
  $BASE_COMMAND --name=$NAME --model_path="policy_params/$NAME"

  yq -i -y ".ENV_CONFIG.rew_shaping_params.wrong_dish_discount = 0.1" "$FILE"
  yq -i -y ".ENV_CONFIG.rew_shaping_params.correct_dish_bonus = 2" "$FILE"
  yq -i -y ".ENV_CONFIG.rew_shaping_params.multiplier_params = [[0, 1], [1, 30], [3, 40], [100, 50]]" "$FILE"

  NAME="burrito_2ag_nopos_steakmush_27act_both_trial$i"
  $BASE_COMMAND --name=$NAME --model_path="policy_params/$NAME"

  yq -i -y ".ENV_CONFIG.rew_shaping_params.wrong_dish_discount = 1" "$FILE"
  yq -i -y ".ENV_CONFIG.rew_shaping_params.correct_dish_bonus = 1" "$FILE"
  yq -i -y ".ENV_CONFIG.rew_shaping_params.multiplier_params = [[0, 1], [1,10], [2, 20], [3, 30], [100, 40]]" "$FILE"
done

