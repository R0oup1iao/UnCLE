accelerate launch src/run.py \
    --dataset cluster_lorenz \
    --hierarchy 4 \
    --lambda_bal 0.0 \
    --lambda_ent 0.01 \
    --tau_decay 0.03 \
    # --norm_coords