UnCLe: UnCoupling for Causal Discovery

Implementation of UnCLe (Towards Scalable Dynamic Causal Discovery in Non-linear Temporal Systems). This project provides a complete pipeline for generating synthetic causal data, training the UnCLENet model, and visualizing dynamic causal graphs.

📂 Directory Structure

.
├── data/               # Data storage
├── results/            # Checkpoints and visualization outputs
├── src/                # Source code
│   ├── model.py        # UnCLENet architecture
│   ├── train.py        # Training loop
│   ├── tcn.py          # TCN components
│   ├── metrics.py      # Evaluation metrics (AUROC, F1, etc.)
│   ├── visualize.py    # Heatmap and GIF generation
│   └── ...
├── scripts/            # Executable scripts
│   ├── generate_data.py
│   ├── run_training.py
│   └── run_inference.py
└── requirements.txt


🛠️ Installation

Install dependencies:

pip install -r requirements.txt


🚀 Usage Workflow

1. Generate Synthetic Data

Generate synthetic datasets (Lorenz96, NC8, TVSEM) with Ground Truth.

# Generate Lorenz96 (20 vars, 5 replicas)
python scripts/generate_data.py --dataset lorenz96 --num_replicas 5

# Generate NC8 (8 vars, Non-linear)
python scripts/generate_data.py --dataset nc8 --num_replicas 5

# Generate TVSEM (Time-varying causal switches)
python scripts/generate_data.py --dataset tvsem --num_replicas 5


2. Train Model

Train UnCLENet using the generated data. You can use preset configurations (lorenz96, nc8, tvsem) or override parameters.

Train on Lorenz96:

python scripts/run_training.py \
    --dataset_path data/synthetic/lorenz96 \
    --config_name lorenz96 \
    --output_dir results/checkpoints/lorenz96

Train on TVSEM:

python scripts/run_training.py \
    --dataset_path data/synthetic/tvsem \
    --config_name tvsem \
    --output_dir results/checkpoints/tvsem

Train on NC8 (Quick Experiment):

python scripts/run_training.py \
    --dataset_path data/synthetic/nc8 \
    --config_name nc8 \
    --recon_epochs 200 \
    --joint_epochs 500


3. Inference & Evaluation

Run inference to generate static and dynamic causal graphs, calculate metrics (AUROC, F1, etc.), and visualize the evolution of causality as a GIF.

python scripts/run_inference.py \
    --model_path results/checkpoints/lorenz96/model.pth \
    --data_path data/synthetic/lorenz96 \
    --config_name lorenz96 \
    --output_dir results/eval/lorenz96 \
    --make_gif

python scripts/run_inference.py \
    --model_path results/checkpoints/tvsem/model.pth \
    --data_path data/synthetic/tvsem \
    --config_name tvsem \
    --output_dir results/eval/tvsem \
    --make_gif


📊 Outputs

After running inference, check results/eval/lorenz96/ for:

metrics_static.json: JSON file containing AUROC, AUPRC, F1, ACC, Recall.

heatmap_static_est.png: Estimated static causal graph.

heatmap_static_gt.png: Ground Truth static graph.

causal_evolution.gif: Animation of the dynamic causal graph over time.