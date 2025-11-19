import sys
import os
import argparse
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_dataloader
from src.model import UnCLENet
from src.train import train_model
from src.utils import set_seed
from src.config import get_config

def main():
    parser = argparse.ArgumentParser(description="Train UnCLe Model")
    
    # 必须参数
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to dataset folder")
    parser.add_argument('--config_name', type=str, default='lorenz96', choices=['lorenz96', 'nc8', 'tvsem', 'default'], help="Config preset name")
    
    # 可选覆盖参数 (Optional Overrides)
    parser.add_argument('--output_dir', type=str, default=None, help="Dir to save model (default: results/checkpoints/<config_name>)")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--recon_epochs', type=int, default=None)
    parser.add_argument('--joint_epochs', type=int, default=None)
    
    args = parser.parse_args()
    
    # 1. Load Config
    config = get_config(args.config_name)
    
    # 2. Apply Overrides (如果有命令行输入，则覆盖 config 中的默认值)
    if args.seed is not None: config.seed = args.seed
    if args.recon_epochs is not None: config.recon_epochs = args.recon_epochs
    if args.joint_epochs is not None: config.joint_epochs = args.joint_epochs
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join('results', 'checkpoints', args.config_name)
    
    # 3. Setup
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"--- Running Training with Config: {args.config_name} ---")
    print(f"Batch Size: {config.batch_size}, LR: {config.lr}, Latent: {config.latent_dim}")
    
    # 4. Data
    loader = get_dataloader(args.dataset_path, batch_size=config.batch_size)
    
    # 5. Model
    model = UnCLENet(
        N=config.n_vars, 
        latent_C=config.latent_dim,
        tcn_levels_unc=config.tcn_levels,
        tcn_hidden_unc=config.tcn_hidden,
        kernel_size=config.kernel_size,
        dropout=config.dropout
    ).to(device)
    
    # 6. Train
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(
        model=model, 
        dataloader=loader, 
        device=device, 
        output_dir=args.output_dir,
        config=config # 传递整个 config 对象
    )

if __name__ == "__main__":
    main()