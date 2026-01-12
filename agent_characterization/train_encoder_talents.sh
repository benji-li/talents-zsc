ALGO="bp"
LAYOUT="open"

windows=(50 100)
horizons=(50 100) 
betas=(0.001 0.005)

windows=(50)
horizons=(50)
betas=(0.005)

# Run all combinations
for w in "${windows[@]}"; do
  for h in "${horizons[@]}"; do
    for b in "${betas[@]}"; do
      echo "Testing: window=$w, horizon=$h, beta=$b"

      python train_encoder.py \
        --window-length $w \
        --dataset-path "../data/burrito_${LAYOUT}_${ALGO}_2pol.pkl" \
        --horizon $h \
        --beta-end $b \
        --num-epochs 50 \
        --save-dir "2pol_${LAYOUT}_${ALGO}" \
        --batch-size 128
    done
  done
done
