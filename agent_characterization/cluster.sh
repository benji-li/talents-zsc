#!/bin/bash
SRC=$(pwd)

BASE_COMMAND="python -m analysis.eval_roles"

DATASET_PATH="../data/burrito_open_bp_11pol.pkl"

MODEL_PATH="open_bp/20250627_175003_ep_100/"

MIN_CLUSTERS=2
MAX_CLUSTERS=20

$BASE_COMMAND --model-path="$MODEL_PATH" --dataset-path="$DATASET_PATH" --min-clusters=$MIN_CLUSTERS --max-clusters=$MAX_CLUSTERS
