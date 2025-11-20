import sys
import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
import numpy as np
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from tqdm.auto import tqdm

# 确保能导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import ST_CausalFormer
from dataloader import get_data_context

# ==========================================
# Visualization Logic
# ==========================================
def save_plots(model, meta, est_patch_graph, output_dir):
    """保存 6 子图可视化 (适配真实数据 & 动态聚类)"""
    N = meta['coords'].shape[0]
    coords = meta['coords']
    
    # [修改点] Patch ID 从模型里取，因为是 KMeans 动态算的
    if hasattr(model, 'patch_labels'):
        patch_ids = model.patch_labels 
    else:
        # Fallback if unwrap failed or using single gpu model directly
        patch_ids = np.zeros(N)

    est_fine = model.fine_model.static_causal_aggregation().detach().cpu().numpy().T
    est_coarse = est_patch_graph.detach().cpu().numpy().T
    mask_vis = model.psi_mask.mean(0).cpu().numpy().T
    
    gt_fine = meta['gt_fine']
    # 真实数据没有 Coarse GT，画个占位符
    gt_coarse = meta.get('gt_coarse')
    if gt_coarse is None:
        gt_coarse = np.zeros_like(est_coarse)

    plt.switch_backend('Agg') 
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Spatial
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(coords[:, 0], coords[:, 1], c=patch_ids, cmap='tab10', s=50)
    if N <= 32: # 节点少的时候才显示数字，不然太乱
        for i in range(N): ax1.text(coords[i,0], coords[i,1], str(i), fontsize=8)
    ax1.set_title(f"Spatial Layout ({N} Nodes)")

    # 2. GT Coarse
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(gt_coarse, cmap='Blues', vmin=0)
    ax2.set_title("GT Coarse (N/A for Real Data)")

    # 3. Est Coarse
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(est_coarse, cmap='Reds', vmin=0)
    ax3.set_title("Est Coarse")

    # 4. GT Fine
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(gt_fine, cmap='Blues', vmin=0)
    ax4.set_title("GT Fine (Node)")

    # 5. Est Fine
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(est_fine, cmap='Reds', vmin=0)
    ax5.set_title("Est Fine (Node)")

    # 6. Mask
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(mask_vis, cmap='Greens', vmin=0)
    ax6.set_title("Adaptive Spatial Mask")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "result_full.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

# ==========================================
# Training Phases
# ==========================================
def train_phase_coarse(model, loader, optimizer, accelerator, epochs):
    model.train() 
    progress_bar = tqdm(range(epochs), disable=not accelerator.is_local_main_process, desc="Phase 1: Coarse")
    
    for ep in progress_bar:
        epoch_loss = 0.0
        for batch in loader:
            # [Fix] 兼容性处理：如果是 list/tuple 则取第0个，否则直接用
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            # 此时 x 应该是 (Batch, N, T)
            
            optimizer.zero_grad()
            (x_rec, x_pre, _), x_patch = model(x, mode='coarse')
            
            loss_rec = F.mse_loss(x_rec, x_patch)
            loss_pre = F.mse_loss(x_pre, x_patch[..., 1:])
            
            # [DDP Fix] Unwrap 访问参数
            unwrapped = accelerator.unwrap_model(model)
            loss_l1 = 2e-3 * unwrapped.coarse_model.psi_l1()
            
            loss = loss_rec + loss_pre + loss_l1
                   
            accelerator.backward(loss)
            optimizer.step()
            epoch_loss += loss.item()
        
        # 记录 epoch 平均 loss
        accelerator.log({
            "coarse/total_loss": epoch_loss / len(loader),
            "coarse/mse_rec": loss_rec.item(),
            "coarse/mse_pre": loss_pre.item(),
            "coarse/l1_psi": loss_l1.item()
        })
            
        progress_bar.set_postfix(loss=epoch_loss / len(loader))

def train_phase_fine(model, loader, optimizer, accelerator, epochs):
    model.train()
    progress_bar = tqdm(range(epochs), disable=not accelerator.is_local_main_process, desc="Phase 2: Fine")
    
    for ep in progress_bar:
        epoch_loss = 0.0
        for batch in loader:
            # [Fix] 同上
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            optimizer.zero_grad()
            x_rec, x_pre, _ = model(x, mode='fine')
            
            loss_rec = F.mse_loss(x_rec, x)
            loss_pre = F.mse_loss(x_pre, x[..., 1:])
            
            unwrapped = accelerator.unwrap_model(model)
            loss_l1 = 1e-3 * unwrapped.fine_model.psi_l1()
            
            loss = loss_rec + loss_pre + loss_l1
            
            accelerator.backward(loss)
            
            # [DDP Fix] 梯度操作需 Unwrap
            if unwrapped.fine_model.Psi.grad is not None:
                unwrapped.fine_model.Psi.grad.mul_(unwrapped.psi_mask.to(x.device))
                
            optimizer.step()
            epoch_loss += loss.item()
            
        accelerator.log({
            "fine/total_loss": epoch_loss / len(loader),
            "fine/mse_rec": loss_rec.item(),
            "fine/mse_pre": loss_pre.item(),
            "fine/l1_psi": loss_l1.item()
        })
            
        progress_bar.set_postfix(loss=epoch_loss / len(loader))

# ==========================================
# Main
# ==========================================
def main(args):
    # [DDP Fix] 允许部分参数不参与计算 (针对两阶段训练)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        log_with="wandb",
        project_dir=args.output_dir,
        kwargs_handlers=[ddp_kwargs]
    )
    set_seed(args.seed)
    
    # W&B Tracker Init
    if accelerator.is_main_process:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        accelerator.init_trackers(
            project_name=args.project_name, 
            config=vars(args),
            init_kwargs={"wandb": {"entity": args.wandb_entity}}
        )
        accelerator.print(f"🚀 Experiment: {args.dataset}, N={args.N}, Patches={args.k_patches}")

    # 2. Data Loading
    train_loader, val_loader, meta = get_data_context(args)

    # 3. Model Init
    model = ST_CausalFormer(
        N=args.N, 
        coords=meta['coords'], 
        k_patches=args.k_patches, 
        latent_C=args.latent_C,
        d_model=args.d_model
    )

    # 4. Optimizers
    opt_coarse = torch.optim.Adam(model.coarse_model.parameters(), lr=args.lr_coarse)
    opt_fine = torch.optim.Adam(model.fine_model.parameters(), lr=args.lr_fine)

    # 5. Prepare
    model, opt_coarse, opt_fine, train_loader, val_loader = accelerator.prepare(
        model, opt_coarse, opt_fine, train_loader, val_loader
    )

    # === Phase 1: Coarse Training ===
    train_phase_coarse(model, train_loader, opt_coarse, accelerator, args.epochs_coarse)
    
    # === Intermediate: Mask Update ===
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    # [Fix] update_mask 内部已经处理了 device 问题
    patch_graph, thresh, ratio = unwrapped_model.update_mask()
    
    if accelerator.is_main_process:
        accelerator.print(f"✅ Mask Updated. Threshold: {thresh:.4f}, Ratio: {ratio:.2%}")
        accelerator.log({"mask/threshold": thresh, "mask/ratio": ratio})

    # === Phase 2: Fine Training ===
    train_phase_fine(model, train_loader, opt_fine, accelerator, args.epochs_fine)

    # === Visualization & Upload ===
    if accelerator.is_main_process:
        accelerator.print("🎨 Generating Plots...")
        path = save_plots(unwrapped_model, meta, patch_graph, args.output_dir)
        accelerator.print(f"✅ Result saved to disk: {path}")
        
        wandb.log({"final_result_plot": wandb.Image(path)})
        accelerator.print("☁️ Result uploaded to W&B")
        
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ST-CausalFormer Training Script")
    
    # ================= Data Params =================
    parser.add_argument("--dataset", type=str, default="lorenz96", choices=["lorenz96", "tvsem", "nc8"])
    parser.add_argument("--data_path", type=str, default="data/synthetic")
    parser.add_argument("--replica_id", type=int, default=0)
    
    # ================= Model Params =================
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--k_patches", type=int, default=8)
    parser.add_argument("--latent_C", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    
    # ================= Training Params =================
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    
    parser.add_argument("--epochs_coarse", type=int, default=30)
    parser.add_argument("--epochs_fine", type=int, default=30)
    parser.add_argument("--lr_coarse", type=float, default=3e-3)
    parser.add_argument("--lr_fine", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    
    # ================= System =================
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--project_name", type=str, default="ST-CausalFormer-Exp")
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    args = parser.parse_args()
    
    main(args)