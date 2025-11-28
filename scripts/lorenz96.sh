accelerate launch src/run.py \
    --dataset lorenz96 \
    --hierarchy 32 8\
    --lambda_bal 0.0 \
    --lambda_ent 0.01 \
    --tau_decay 0.03 \
    # --norm_coords