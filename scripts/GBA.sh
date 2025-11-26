accelerate launch src/run.py \
    --dataset GBA \
    --data_path data/real \
    --N 2352 \
    --hierarchy 512 256 32 8 \
    --window_size 12 \
    --lambda_ent 1e-4 \
    --lambda_bal 2.0 \
    --tau_decay 0.005 \
    --batch_size 16