import sys
import os
import argparse
import torch
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataloader import get_dataloader
from src.model import UnCLENet
from src.utils import set_seed
from src.metrics import count_accuracy
from src.visualize import plot_static_heatmap, create_dynamic_gif
from src.config import get_config  # [New] 引入配置加载

def main():
    parser = argparse.ArgumentParser(description="Run UnCLe Inference & Evaluation")
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True, help="Path to trained model.pth")
    parser.add_argument('--data_path', type=str, required=True, help="Path to data folder")
    parser.add_argument('--config_name', type=str, default='lorenz96', 
                        choices=['lorenz96', 'nc8', 'tvsem', 'default'], 
                        help="Config preset name used for training (Must match training!)")
    
    parser.add_argument('--output_dir', type=str, default='results/evaluation', help="Output directory")
    
    # 可选覆盖参数 (通常不需要，除非你想覆盖 Config 中的设定)
    parser.add_argument('--n_vars', type=int, help="Override n_vars from config if needed")
    
    # 可视化控制
    parser.add_argument('--make_gif', action='store_true', help="Generate GIF for dynamic graph")
    parser.add_argument('--gif_len', type=int, default=200, help="Number of frames for GIF")
    
    args = parser.parse_args()
    
    # 1. Load Config
    # 必须加载与训练时相同的配置，以保证模型结构一致
    config = get_config(args.config_name)
    
    # 如果命令行强制指定了 n_vars，则覆盖配置
    if args.n_vars is not None:
        config.n_vars = args.n_vars

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--- Running Inference with Config: {args.config_name} ---")
    print(f"Model Params: N={config.n_vars}, Latent={config.latent_dim}, "
          f"TCN_Levels={config.tcn_levels}, Hidden={config.tcn_hidden}")

    # 2. Init Model (使用 Config 初始化)
    model = UnCLENet(
        N=config.n_vars,
        latent_C=config.latent_dim,
        tcn_levels_unc=config.tcn_levels,
        tcn_hidden_unc=config.tcn_hidden,
        kernel_size=config.kernel_size,
        dropout=config.dropout
    ).to(device)
    
    # 加载权重
    print(f"Loading weights from {args.model_path}...")
    try:
        # 尝试使用安全模式加载
        try:
            state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        except TypeError:
            # 兼容旧版本 PyTorch
            state_dict = torch.load(args.model_path, map_location=device)
            
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"\n[Error] Model loading failed: {e}")
        print(f"[Tip] Please ensure '--config_name {args.config_name}' matches the config used during training.")
        return

    model.eval()

    # 3. Load Data
    loader = get_dataloader(args.data_path, batch_size=1, shuffle=False)
    try:
        data_batch, gt_batch = next(iter(loader))
    except StopIteration:
        print("[Error] DataLoader is empty. Please check your data path.")
        return

    data_sample = data_batch[[0], ...].to(device) # (1, N, T)
    gt_sample = gt_batch[0].numpy()               # (N, N) or (T, N, N)

    # ---------------------------------------------------------
    # 4. Static Inference & Metrics
    # ---------------------------------------------------------
    print("Calculating Static Causal Graph (Aggregation)...")
    
    # UnCLENet.Psi shape: (C, Target, Source) -> Mean -> (Target, Source)
    # Need Transpose to match GT convention [Cause, Effect] i.e., (Source, Target)
    est_static_matrix = model.static_causal_aggregation().cpu().numpy().T 
    
    # Handle GT dimensions
    if gt_sample.ndim == 3: 
        print("Ground Truth is dynamic. Averaging for static metric comparison.")
        gt_static = np.mean(gt_sample, axis=0) 
        gt_static = (gt_static > 0).astype(int)
    else:
        gt_static = gt_sample
        
    # Calculate Metrics
    metrics = count_accuracy(gt_static, est_static_matrix)
    print("\n--- Static Graph Metrics ---")
    print(json.dumps(metrics, indent=4))
    
    # Save Metrics
    with open(os.path.join(args.output_dir, 'metrics_static.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Visualize Static
    plot_static_heatmap(
        est_static_matrix, 
        "Estimated Static Graph (Source->Target)", 
        os.path.join(args.output_dir, "heatmap_static_est.png")
    )
    plot_static_heatmap(
        gt_static, 
        "Ground Truth Static Graph", 
        os.path.join(args.output_dir, "heatmap_static_gt.png")
    )

    # ---------------------------------------------------------
    # 5. Dynamic Inference & Visualization
    # ---------------------------------------------------------
    print("\nCalculating Dynamic Causal Strengths (Perturbation)...")
    dynamic_strengths = model.dynamic_causal_strengths(data_sample).cpu().numpy()
    
    print(f"Dynamic Strengths Shape: {dynamic_strengths.shape}")
    
    if args.make_gif:
        print(f"Generating GIF ({args.gif_len} frames)...")
        gif_data = dynamic_strengths[:, :, :args.gif_len]
        create_dynamic_gif(
            gif_data, 
            os.path.join(args.output_dir, "causal_evolution.gif"),
            fps=10
        )

    # Save raw results
    np.save(os.path.join(args.output_dir, "est_static.npy"), est_static_matrix)
    np.save(os.path.join(args.output_dir, "est_dynamic.npy"), dynamic_strengths)
    
    print(f"\nAll results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()