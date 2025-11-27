accelerate launch src/run.py \
    --dataset GLA \
    --data_path data/real \
    --N 3834 \
    --hierarchy 512 64 8 \
    --window_size 12 \
    --lambda_ent 1e-4 \
    --lambda_bal 2.0 \
    --tau_decay 0.005 \
    --batch_size 16