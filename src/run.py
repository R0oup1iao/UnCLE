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
from dataloader import get_data_context
from visualize import create_dynamic_gif
from metrics import count_accuracy

# ==========================================
# Helper Functions
# ==========================================
def compute_entropy_loss(S):
    """
    S: (B, N, K) 分配矩阵
    目标: 最小化熵，鼓励 S 接近 One-hot
    """
    entropy = -torch.sum(S * torch.log(S + 1e-6), dim=-1)
    return entropy.mean()

def pretty_dict(d):
    """Convert numpy/tensor keys/values to python int for clean printing"""
    return {int(k): int(v) for k, v in d.items()}

# ==========================================
# Dynamic Causal Discovery Logic
# ==========================================
@torch.no_grad()
def compute_dynamic_strengths(model, x, device):
    model.eval()
    fine = model.fine_model
    if hasattr(model, 'psi_mask'):
        fine.Psi.data.mul_(model.psi_mask.to(device))
    
    B, N, T = x.shape
    x_recon, x_pred, z_orig = fine(x)
    eps_orig = (x_pred - x[..., 1:]) ** 2
    eps_orig_mean = eps_orig.mean(dim=0)
    
    strengths = torch.zeros(N, N, T-1, device=device)
    for j in range(N):
        x_j_pert = x[:, j, :].clone()
        perm = torch.randperm(T, device=device)
        x_j_pert = x_j_pert[:, perm]
        
        z_j_new = fine.uncouple(x_j_pert.unsqueeze(1))
        z_mixed = z_orig.clone()
        z_mixed[:, j, :, :] = z_j_new.squeeze(1)
        
        zhat_mixed = fine.predict_latent_next(z_mixed)
        x_pred_mixed = fine.recouple(zhat_mixed[..., :-1, :])
        
        eps_pert = (x_pred_mixed - x[..., 1:]) ** 2
        eps_pert_mean = eps_pert.mean(dim=0)
        
        delta = F.relu(eps_pert_mean - eps_orig_mean)
        strengths[j, :, :] = delta
        
    return strengths

# ==========================================
# Visualization Logic
# ==========================================
def save_plots(model, meta, est_patch_graph, output_dir):
    N = meta['coords'].shape[0]
    coords = meta['coords']
    
    # Patch Labels 来自第一层聚类 (Fine -> Level 1)
    if hasattr(model, 'patch_labels'):
        patch_ids = model.patch_labels 
    else:
        patch_ids = np.zeros(N)

    est_fine = model.fine_model.static_causal_aggregation().detach().cpu().numpy().T
    
    # est_patch_graph 现在是 Top Level Graph
    est_coarse = est_patch_graph.detach().cpu().numpy().T
    
    mask_vis = model.psi_mask.mean(0).cpu().numpy().T
    
    gt_fine = meta.get('gt_fine')
    if gt_fine is None: gt_fine = np.zeros_like(est_fine)
        
    gt_coarse = meta.get('gt_coarse')
    if gt_coarse is None: gt_coarse = np.zeros_like(est_coarse)

    plt.switch_backend('Agg') 
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Spatial Layout (Level 1 Clusters)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(coords[:, 0], coords[:, 1], c=patch_ids, cmap='tab10', s=50)
    ax1.set_title(f"Spatial Clusters (L1)")

    # 2. GT Coarse
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(gt_coarse, cmap='Blues', vmin=0)
    ax2.set_title("GT Coarse (Ref)")

    # 3. Est Coarse (Top Level)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(est_coarse, cmap='Reds', vmin=0)
    ax3.set_title("Est Coarse (Top Level)")

    # 4. GT Fine
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(gt_fine, cmap='Blues', vmin=0)
    ax4.set_title("GT Fine")

    # 5. Est Fine
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(est_fine, cmap='Reds', vmin=0)
    ax5.set_title("Est Fine")

    # 6. Mask
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
def train_phase_hierarchy(model, loader, optimizer, accelerator, epochs):
    model.train() 
    progress_bar = tqdm(range(epochs), disable=not accelerator.is_local_main_process, desc="Phase 1: Hierarchy")
    
    for ep in progress_bar:
        epoch_loss = 0.0
        epoch_ent = 0.0
        tau = max(0.1, np.exp(-ep * 0.03)) 

        for batch in loader:
            if isinstance(batch, (list, tuple)): x = batch[0]
            else: x = batch
            
            optimizer.zero_grad()
            
            # Forward returns a list of results for each level
            results = model(x, mode='hierarchy', tau=tau)
            
            total_loss = 0.0
            total_ent_loss = 0.0
            
            # Sum loss across all levels
            for res in results:
                (x_rec, x_pre, _), x_patch, S = res['out'], res['target'], res['S']
                
                loss_rec = F.mse_loss(x_rec, x_patch)
                loss_pre = F.mse_loss(x_pre, x_patch[..., 1:])
                loss_ent = 1e-2 * compute_entropy_loss(S)
                
                total_loss += loss_rec + loss_pre + loss_ent
                total_ent_loss += loss_ent
            
            # L1 Reg for all coarse models
            unwrapped = accelerator.unwrap_model(model)
            l1_loss = 0.0
            for cm in unwrapped.coarse_models:
                l1_loss += 2e-3 * cm.psi_l1()
            
            loss = total_loss + l1_loss
            
            accelerator.backward(loss)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_ent += total_ent_loss.item()
        
        accelerator.log({
            "hierarchy/total_loss": epoch_loss / len(loader),
            "hierarchy/ent": epoch_ent / len(loader),
            "tau": tau
        })
        progress_bar.set_postfix(loss=epoch_loss / len(loader))

    # [Debug] Pretty Print Distribution
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        print(f"\n🔍 [Debug] Hierarchy Cluster Analysis:")
        for i, S in enumerate(unwrapped.S_list):
            if S is not None:
                S_hard = S.mean(0).argmax(dim=1).cpu().numpy()
                unique, counts = np.unique(S_hard, return_counts=True)
                dist = dict(zip(unique, counts))
                print(f"   Level {i} (Dim {unwrapped.hierarchy[i]}) Dist: {pretty_dict(dist)}")

def train_phase_fine(model, loader, optimizer, accelerator, epochs):
    model.train()
    progress_bar = tqdm(range(epochs), disable=not accelerator.is_local_main_process, desc="Phase 2: Fine (Masked)")
    
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
    
    if accelerator.is_main_process:
        if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
        accelerator.init_trackers(project_name=args.project_name, config=vars(args), 
                                  init_kwargs={"wandb": {"entity": args.wandb_entity}})
        accelerator.print(f"🚀 Experiment: {args.dataset}, N={args.N}, Hierarchy={args.hierarchy}")

    train_loader, val_loader, meta = get_data_context(args)

    # Init Model with Argument Hierarchy
    model = ST_CausalFormer(
        N=args.N, 
        coords=meta['coords'], 
        hierarchy=args.hierarchy,   # Use Args
        latent_C=args.latent_C, 
        d_model=args.d_model,
        seq_len=args.window_size
    )

    # Optimizer
    params_hierarchy = []
    for cm in model.coarse_models: params_hierarchy.extend(list(cm.parameters()))
    for p in model.poolers: params_hierarchy.extend(list(p.parameters()))
    
    opt_hierarchy = torch.optim.Adam(params_hierarchy, lr=args.lr_coarse)
    opt_fine = torch.optim.Adam(model.fine_model.parameters(), lr=args.lr_fine)

    model, opt_hierarchy, opt_fine, train_loader, val_loader = accelerator.prepare(
        model, opt_hierarchy, opt_fine, train_loader, val_loader
    )

    # 1. Train Hierarchy
    train_phase_hierarchy(model, train_loader, opt_hierarchy, accelerator, args.epochs_coarse)
    
    # 2. Update Mask (Cascaded)
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    
    # [Fix] patch_graph is assigned here
    patch_graph, thresh, ratio = unwrapped_model.update_mask()
    
    if accelerator.is_main_process: 
        accelerator.print(f"✅ Cascaded Mask Generated. Ratio: {ratio:.2%}")

    # 3. Train Fine
    train_phase_fine(model, train_loader, opt_fine, accelerator, args.epochs_fine)

    # 4. Evaluation
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print("\n📊 Evaluation & Visualization...")
        
        est_static = unwrapped_model.fine_model.static_causal_aggregation().detach().cpu().numpy()
        gt_fine = meta.get('gt_fine')
        
        if gt_fine is not None:
            metrics = count_accuracy(gt_fine, est_static)
            accelerator.print("\n🏆 Causal Discovery Metrics:")
            accelerator.print(json.dumps(metrics, indent=4))
            wandb.log(metrics)
        else:
            accelerator.print("⚠️ Ground Truth not available. Skipping metrics.")

        # [Fix] Now patch_graph variable exists and is passed correctly
        path_static = save_plots(unwrapped_model, meta, patch_graph, args.output_dir)
        try: wandb.log({"static_result": wandb.Image(path_static)})
        except: pass
        
        accelerator.print("🎬 Computing Dynamic Causal Graph...")
        torch.cuda.empty_cache() 
        try:
            val_iter = iter(val_loader)
            val_batch = next(val_iter)
            if isinstance(val_batch, (list, tuple)): val_batch = val_batch[0]
            val_batch = val_batch.to(accelerator.device)
            
            dynamic_strengths = compute_dynamic_strengths(unwrapped_model, val_batch, accelerator.device)
            dynamic_strengths_np = dynamic_strengths.cpu().numpy()
            
            np.save(os.path.join(args.output_dir, "est_dynamic.npy"), dynamic_strengths_np)
            
            gif_path = os.path.join(args.output_dir, "causal_evolution.gif")
            gif_len = min(dynamic_strengths_np.shape[2], 200)
            create_dynamic_gif(dynamic_strengths_np[:, :, :gif_len], gif_path, fps=10)
            accelerator.print(f"✅ GIF saved: {gif_path}")
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
    
    # [New] Hierarchy Argument (list of ints)
    parser.add_argument("--hierarchy", type=int, nargs='+', default=[32, 8], 
                        help="Hierarchy levels, e.g. --hierarchy 32 8")
    
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
    
    # Deprecated arg (kept for compatibility but not used)
    parser.add_argument("--k_patches", type=int, default=8, help="Deprecated, use --hierarchy")

    args = parser.parse_args()
    main(args)