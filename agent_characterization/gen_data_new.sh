#!/bin/bash

YAML_CONFIG="./overcooked/src/burrito_rl/config/burrito_2p_gendata_bp"

OPEN_POP_DIR="/data/shuyang/overcooked/src/burrito_rl/policy_params/open_bp_population3"
FC_POP_DIR="/Users/benjili/Research/burrito_cooperation/overcooked/src/burrito_rl/bp_data_new/policy_params/forced_coordination_bp_population"
HALLWAY_POP_DIR="/data/shuyang/overcooked/src/burrito_rl/policy_params/hallway_bp_population3"

RING_POP_DIR="/Users/benjili/Research/burrito_cooperation/overcooked/src/burrito_rl/bp_data_new/policy_params/ring_bp_population"

OPEN_POLICIES=(
  "8"
  "33"
  "7"
  "25"
  "14"
  "21"
  "29"
  "42"
  "47"
  "16"
  "10"
)

FC_POLICIES=(
  "3"
  "36"
  "30"
  "34"
  "35"
  "33"
  "8"
  "31"
  "18"
  "10"
)

HALLWAY_POLICIES=(
   "8"
   "25"
    "13"
   "11"
   "15"
   "26"
  "23"
  "29"
  "21"
  "34"
  "10"
)

RING_POLICIES=(
  "13"
  "33"
  "25"
  "31"
  "11"
  "17"
  "21"
  "41"
  "5"
  "2"
  "24"
)
LAYOUT="hallway"
POP_DIR=$HALLWAY_POP_DIR
YAML="${YAML_CONFIG}_${LAYOUT}.yaml"

for POL in "${OPEN_POLICIES[@]}"; do
  echo "Population Directory: $POP_DIR"
  echo "Processing policy: $POL"
  echo "Using Config: $YAML"

  POL1_PATH="$POP_DIR/pol$POL/policy_state.pkl"
  POLBR_PATH="$POP_DIR/polBR$POL/policy_state.pkl"

  echo "POLICY 1: $POL1_PATH"
  echo "BR POLICY: $POLBR_PATH"
  AGENT_PARAM="agent-$POL"

  # Use yq to update the pretrained model paths in-place
  yq -i -y ".BASE_CONFIG.pretrained_model_path[0] = \"$POL1_PATH\"" "$YAML"
  yq -i -y ".BASE_CONFIG.pretrained_model_path[1] = \"$POLBR_PATH\"" "$YAML"

  echo "Running evaluation for $AGENT_PARAM"
  python agent_characterization/gen_data.py \
    --eval-episodes 18 \
    --agent-0 "$AGENT_PARAM" \
    --dataset-path "./data/burrito_${LAYOUT}_bp_6pol.pkl" \
    --config "$YAML" \
    --layout "$LAYOUT" \
    --save-data

  echo "Completing processing for $POL"
  echo "----------------------------------------"
done

echo "All directories processed"
