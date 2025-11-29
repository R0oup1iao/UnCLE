accelerate launch src/run.py \
    --dataset GBA \
    --data_path data/real \
    --N 2352 \
    --hierarchy 512 64 8 \
    --window_size 12 \
    --batch_size 8

accelerate launch src/run.py \
    --dataset GLA \
    --data_path data/real \
    --N 3834 \
    --hierarchy 512 64 8 \
    --batch_size 8