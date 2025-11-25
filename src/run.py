import datetime
import sys
import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
import numpy as np
import json 
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from tqdm.auto import tqdm

# 确保能导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import ST_CausalFormer
from dataloader import get_data_context, CausalTimeSeriesDataset, load_from_disk
from visualize import create_dynamic_gif
from metrics import count_accuracy

# ==========================================
# Helper Functions
# ==========================================
def compute_entropy_loss(S):
    """S: (B, N, K) -> 最小化个体熵"""
    entropy = -torch.sum(S * torch.log(S + 1e-6), dim=-1)
    return entropy.mean()

def compute_balance_loss(S):
    """S: (B, N, K) -> 负载均衡 (KL Divergence)"""
    K = S.shape[-1]
    cluster_usage = S.mean(dim=0).mean(dim=0)
    neg_entropy = torch.sum(cluster_usage * torch.log(cluster_usage + 1e-6))
    max_entropy = np.log(K)
    return neg_entropy + max_entropy

def pretty_dict(d):
    return {int(k): int(v) for k, v in d.items()}

# ==========================================
# Dynamic Causal Discovery Logic
# ==========================================
@torch.no_grad()
def compute_dynamic_strengths_batch(model, x, device):
    """
    计算单个 Window 的因果强度。
    x: (1, N, T)
    Output: (N, N, T-1)
    """
    model.eval()
    fine = model.fine_model
    if hasattr(model, 'psi_mask'):
        fine.Psi.data.mul_(model.psi_mask.to(device))
    
    B, N, T = x.shape
    # 1. 原始预测误差
    x_recon, x_pred, z_orig = fine(x)
    eps_orig = (x_pred - x[..., 1:]) ** 2 # (1, N, T-1)
    
    strengths = torch.zeros(N, N, T-1, device=device)
    
    # 2. 逐节点扰动 (Perturbation)
    # 为了加速，可以考虑批量扰动，但 N=716 显存可能不够，这里维持逐个循环
    for j in range(N):
        x_j_pert = x[:, j, :].clone()
        perm = torch.randperm(T, device=device)
        x_j_pert = x_j_pert[:, perm] # Shuffle time axis for node j
        
        # Uncouple perturbed node
        z_j_new = fine.uncouple(x_j_pert.unsqueeze(1)) # (1, 1, T, C)
        
        # Mix with original latent
        z_mixed = z_orig.clone()
        z_mixed[:, j, :, :] = z_j_new.squeeze(1)
        
        # Predict again
        zhat_mixed = fine.predict_latent_next(z_mixed)
        x_pred_mixed = fine.recouple(zhat_mixed[..., :-1, :])
        
        eps_pert = (x_pred_mixed - x[..., 1:]) ** 2
        
        # Delta Error: 如果扰动 j 导致误差增加，说明 j 是因
        # Mean over batch (dim 0) is just taking the value since B=1
        delta = F.relu(eps_pert - eps_orig).mean(dim=0) # (N, T-1)
        
        strengths[j, :, :] = delta
        
    return strengths

# ==========================================
# Visualization Logic
# ==========================================
def save_plots(model, meta, est_patch_graph, output_dir):
    N = meta['coords'].shape[0]
    coords = meta['coords']
    if hasattr(model, 'patch_labels'):
        patch_ids = model.patch_labels 
    else:
        patch_ids = np.zeros(N)

    est_fine = model.fine_model.static_causal_aggregation().detach().cpu().numpy().T
    est_coarse = est_patch_graph.detach().cpu().numpy().T
    mask_vis = model.psi_mask.mean(0).cpu().numpy().T
    
    gt_fine = meta.get('gt_fine')
    if gt_fine is None: gt_fine = np.zeros_like(est_fine)
    gt_coarse = meta.get('gt_coarse')
    if gt_coarse is None: gt_coarse = np.zeros_like(est_coarse)

    plt.switch_backend('Agg') 
    fig = plt.figure(figsize=(15, 10))
    
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(coords[:, 0], coords[:, 1], c=patch_ids, cmap='tab10', s=50)
    ax1.set_title(f"Spatial Clusters (L1)")

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(gt_coarse, cmap='Blues', vmin=0)
    ax2.set_title("GT Coarse (Ref)")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(est_coarse, cmap='Reds', vmin=0)
    ax3.set_title("Est Coarse (Top Level)")

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(gt_fine, cmap='Blues', vmin=0)
    ax4.set_title("GT Fine")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(est_fine, cmap='Reds', vmin=0)
    ax5.set_title("Est Fine")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(mask_vis, cmap='Greens', vmin=0)
    ax6.set_title("DiffPool Generated Mask")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "result_full.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

# ==========================================
# Training Phases
# ==========================================
def train_phase_hierarchy(model, loader, optimizer, accelerator, args):
    model.train() 
    progress_bar = tqdm(range(args.epochs_coarse), disable=not accelerator.is_local_main_process, desc="Phase 1: Hierarchy")
    
    for ep in progress_bar:
        epoch_loss = 0.0
        epoch_rec = 0.0
        epoch_bal = 0.0
        tau = max(0.1, np.exp(-ep * args.tau_decay)) 

        for batch in loader:
            if isinstance(batch, (list, tuple)): x = batch[0]
            else: x = batch
            optimizer.zero_grad()
            results = model(x, mode='hierarchy', tau=tau)
            
            total_loss = 0.0
            total_rec_loss = 0.0
            total_bal_loss = 0.0
            
            for res in results:
                (x_rec, x_pre, _), x_patch, S = res['out'], res['target'], res['S']
                loss_rec = F.mse_loss(x_rec, x_patch)
                loss_pre = F.mse_loss(x_pre, x_patch[..., 1:])
                loss_ent = args.lambda_ent * compute_entropy_loss(S)
                loss_bal = args.lambda_bal * compute_balance_loss(S)
                
                total_loss += loss_rec + loss_pre + loss_ent + loss_bal
                total_rec_loss += (loss_rec + loss_pre).item()
                total_bal_loss += loss_bal.item()
            
            unwrapped = accelerator.unwrap_model(model)
            l1_loss = 0.0
            for cm in unwrapped.coarse_models:
                l1_loss += args.lambda_l1 * cm.psi_l1()
            
            loss = total_loss + l1_loss
            accelerator.backward(loss)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_rec += total_rec_loss
            epoch_bal += total_bal_loss
        
        metrics = {
            "hierarchy/total_loss": epoch_loss / len(loader),
            "hierarchy/rec_loss": epoch_rec / len(loader),
            "hierarchy/balance_loss": epoch_bal / len(loader),
            "tau": tau
        }
        accelerator.log(metrics)
        progress_bar.set_postfix(loss=metrics["hierarchy/total_loss"], tau=f"{tau:.2f}")

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        print(f"\n🔍 [Debug] Hierarchy Cluster Analysis (Post-Training):")
        for i, S in enumerate(unwrapped.S_list):
            if S is not None:
                S_hard = S.mean(0).argmax(dim=1).cpu().numpy()
                unique, counts = np.unique(S_hard, return_counts=True)
                dist = dict(zip(unique, counts))
                print(f"   Level {i} (Dim {unwrapped.hierarchy[i]}) Dist: {pretty_dict(dist)}")

def train_phase_fine(model, loader, optimizer, accelerator, args):
    model.train()
    progress_bar = tqdm(range(args.epochs_fine), disable=not accelerator.is_local_main_process, desc="Phase 2: Fine (Masked)")
    
    for ep in progress_bar:
        epoch_loss = 0.0
        for batch in loader:
            if isinstance(batch, (list, tuple)): x = batch[0]
            else: x = batch
            optimizer.zero_grad()
            x_rec, x_pre, _ = model(x, mode='fine')
            unwrapped = accelerator.unwrap_model(model)
            loss_rec = F.mse_loss(x_rec, x)
            loss_pre = F.mse_loss(x_pre, x[..., 1:])
            loss_l1 = 1e-3 * unwrapped.fine_model.psi_l1()
            loss = loss_rec + loss_pre + loss_l1
            accelerator.backward(loss)
            if unwrapped.fine_model.Psi.grad is not None:
                unwrapped.fine_model.Psi.grad.mul_(unwrapped.psi_mask.to(x.device))
            optimizer.step()
            epoch_loss += loss.item()
        accelerator.log({"fine/loss": epoch_loss / len(loader)})
        progress_bar.set_postfix(loss=epoch_loss / len(loader))

# ==========================================
# Main
# ==========================================
def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", project_dir=args.output_dir, kwargs_handlers=[ddp_kwargs])
    set_seed(args.seed)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, args.dataset, timestamp)
    
    if accelerator.is_main_process:
        if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
        accelerator.init_trackers(project_name=args.project_name, config=vars(args), 
                                  init_kwargs={"wandb": {"entity": args.wandb_entity}})
        accelerator.print(f"🚀 Experiment: {args.dataset}, N={args.N}, Hierarchy={args.hierarchy}")

    train_loader, val_loader, meta = get_data_context(args)

    model = ST_CausalFormer(
        N=args.N, 
        coords=meta['coords'], 
        hierarchy=args.hierarchy, 
        latent_C=args.latent_C, 
        d_model=args.d_model,
        seq_len=args.window_size
    )

    params_hierarchy = []
    for cm in model.coarse_models: params_hierarchy.extend(list(cm.parameters()))
    for p in model.poolers: params_hierarchy.extend(list(p.parameters()))
    
    opt_hierarchy = torch.optim.Adam(params_hierarchy, lr=args.lr_coarse)
    opt_fine = torch.optim.Adam(model.fine_model.parameters(), lr=args.lr_fine)

    model, opt_hierarchy, opt_fine, train_loader, val_loader = accelerator.prepare(
        model, opt_hierarchy, opt_fine, train_loader, val_loader
    )

    # 1. Train Hierarchy
    train_phase_hierarchy(model, train_loader, opt_hierarchy, accelerator, args)
    
    # 2. Update Mask
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    patch_graph, thresh, ratio = unwrapped_model.update_mask()
    if accelerator.is_main_process: 
        accelerator.print(f"✅ Cascaded Mask Generated. Ratio: {ratio:.2%}")

    # 3. Train Fine
    train_phase_fine(model, train_loader, opt_fine, accelerator, args)

    # 4. Save Model & Results
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # --- Save Model ---
        model_path = os.path.join(args.output_dir, "model.pth")
        accelerator.save(unwrapped_model.state_dict(), model_path)
        accelerator.print(f"💾 Model saved to {model_path}")

        # --- Static Evaluation ---
        est_static = unwrapped_model.fine_model.static_causal_aggregation().detach().cpu().numpy()
        gt_fine = meta.get('gt_fine')
        
        if gt_fine is not None:
            metrics = count_accuracy(gt_fine, est_static)
            accelerator.print("\n🏆 Metrics: " + json.dumps(metrics, indent=4))
            wandb.log(metrics)
        
        path_static = save_plots(unwrapped_model, meta, patch_graph, args.output_dir)
        try: wandb.log({"static_result": wandb.Image(path_static)})
        except: pass
        
        # --- Full Dynamic Inference ---
        accelerator.print("\n🎬 Computing Full Dynamic Causal Graph (Iterating full timeline)...")
        torch.cuda.empty_cache()
        
        # 1. 创建 Full Timeline Loader (Stride=WindowSize, No Shuffle)
        # 我们需要重新加载原始数据以构建 sequential loader
        base_path = getattr(args, 'data_path', 'data/synthetic')
        data_np, _, _ = load_from_disk(base_path, args.dataset, args.replica_id)
        
        # Standardize using same stats
        mean = data_np.mean(axis=0)
        std = data_np.std(axis=0) + 1e-5
        data_np = (data_np - mean) / std
        
        # Use Stride=WindowSize to cover non-overlapping segments (or close to it)
        # 这样拼接起来就是连续的时间线
        full_ds = CausalTimeSeriesDataset(data_np, args.window_size, stride=args.window_size, mode='train', split_ratio=1.0)
        full_loader = torch.utils.data.DataLoader(full_ds, batch_size=1, shuffle=False)
        
        all_strengths = []
        
        try:
            for batch in tqdm(full_loader, desc="Inferencing"):
                batch = batch.to(accelerator.device) # (1, N, Window)
                
                # Compute strength for this window
                # shape: (N, N, Window-1)
                curr_strengths = compute_dynamic_strengths_batch(unwrapped_model, batch, accelerator.device)
                all_strengths.append(curr_strengths.cpu().numpy())
            
            # Concatenate along time axis (last axis)
            # List of (N, N, W-1) -> (N, N, Total_Time)
            final_dynamic_adj = np.concatenate(all_strengths, axis=2)
            
            print(f"✅ Full Dynamic Graph Computed. Shape: {final_dynamic_adj.shape}")
            np.save(os.path.join(args.output_dir, "est_dynamic.npy"), final_dynamic_adj)
            
            # GIF Generation (Optional, just first 200 frames for preview)
            gif_path = os.path.join(args.output_dir, "causal_evolution_preview.gif")
            create_dynamic_gif(final_dynamic_adj[:, :, :200], gif_path, fps=10)
            try: wandb.log({"dynamic_evolution": wandb.Video(gif_path, fps=10, format="gif")})
            except: pass

        except Exception as e:
            accelerator.print(f"❌ Dynamic inference failed: {e}")

        accelerator.print("☁️ All results uploaded.")
        
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="lorenz96")
    parser.add_argument("--data_path", type=str, default="data/synthetic")
    parser.add_argument("--replica_id", type=int, default=0)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--norm_coords", action="store_true")
    parser.add_argument("--hierarchy", type=int, nargs='+', default=[32, 8])
    parser.add_argument("--latent_C", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs_coarse", type=int, default=100)
    parser.add_argument("--epochs_fine", type=int, default=100)
    parser.add_argument("--lr_coarse", type=float, default=3e-3)
    parser.add_argument("--lr_fine", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--project_name", type=str, default="ST-CausalFormer-Exp")
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    parser.add_argument("--lambda_ent", type=float, default=1e-3)
    parser.add_argument("--lambda_bal", type=float, default=1.0)
    parser.add_argument("--lambda_l1", type=float, default=2e-3)
    parser.add_argument("--tau_decay", type=float, default=0.01)
    parser.add_argument("--k_patches", type=int, default=8)

    args = parser.parse_args()
    main(args)