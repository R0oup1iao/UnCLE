import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import wandb
import json
from tqdm.auto import tqdm
from dataloader import load_from_disk, CausalTimeSeriesDataset
from visualize import create_dynamic_gif
from metrics import count_accuracy

# ==========================================
# 1. Hierarchical Dynamic Engine (Fast)
# ==========================================
def compute_hierarchical_dynamic_graph(model, x, device):
    """
    Computes dynamic causal graph using the Hierarchy to accelerate.
    Strategy:
    1. Compute Gradient Saliency at the Top (Coarsest) Level.
    2. Project the coarse graph back to fine level using Assignment Matrix S.
    
    Speedup: O(N) -> O(K), where K is number of patches (e.g. 8).
    For N=3000, K=8, this is ~375x faster.
    """
    model.eval()
    
    # --- Step 1: Get Hierarchy Info (S matrices and Top Input) ---
    with torch.no_grad():
        # results contains info for all levels [Fine ... Top]
        results = model(x.to(device))
    
    # Identify Top Level
    top_level_idx = len(results) - 1
    top_res = results[top_level_idx]
    
    # --- Step 2: Compute Total Projection Matrix S (Fine -> Top) ---
    # S_list[i] maps level i -> i+1
    # We want S_total: Fine -> Top
    S_total = torch.eye(model.dims[0], device=device)
    
    # Iterate through poolers to chain S matrices
    for i in range(len(results) - 1):
        S_curr = results[i]['S'] # (B, N_curr, N_next)
        if S_curr is None: break
        
        # Average over batch (usually B=1 during eval)
        S_curr = S_curr.mean(dim=0) # (N_curr, N_next)
        
        S_total = torch.mm(S_total, S_curr)
        
    # S_total is now (N_fine, K_top). shape: (N, K)
    
    # --- Step 3: Compute Gradients at Top Level (The Fast Part) ---
    # We detach the input to the top layer to treat it as a leaf node
    x_top = top_res['x_target'].detach().clone() # (B, K, T)
    x_top.requires_grad_(True)
    
    top_layer = model.layers[top_level_idx]
    
    # Run Top Layer Forward
    # We need x_pred
    _, x_pred_top, _ = top_layer(x_top) # (B, K, T-1)
    
    B, K, T_pred = x_pred_top.shape
    
    # Compute Jacobian (K x K)
    # Since K is small (e.g. 8 or 32), this loop is extremely fast.
    coarse_adj = torch.zeros(K, K, T_pred, device=device)
    
    for target_i in range(K):
        target_signal = x_pred_top[:, target_i, :].sum()
        grads = torch.autograd.grad(target_signal, x_top, retain_graph=True)[0] # (B, K, T)
        # Saliency[source, target]
        coarse_adj[:, target_i, :] = grads[0, :, :-1].abs()
        
    # --- Step 4: Project Back to Fine Level (Broadcasting) ---
    # Formula: Adj_Fine = S @ Adj_Coarse @ S.T
    # Dimensions: (N,K) @ (K,K,T) @ (K,N) -> (N,N,T)
    
    fine_adj = torch.zeros(model.dims[0], model.dims[0], T_pred, device='cpu') # Store on CPU
    S_total = S_total.to(device)
    
    # Iterate over time chunks to save GPU memory
    chunk_size = 10 
    for t_start in range(0, T_pred, chunk_size):
        t_end = min(t_start + chunk_size, T_pred)
        
        # (K, K, Chunk)
        sub_coarse = coarse_adj[:, :, t_start:t_end] 
        
        # Reshape sub_coarse to 2D: (K, K*Chunk)
        sub_flat = sub_coarse.reshape(K, -1)
        
        # Left Project: temp = S @ sub_flat -> (N, K*Chunk)
        temp = torch.mm(S_total, sub_flat)
        
        # Reshape temp: (N, K, Chunk)
        temp = temp.reshape(model.dims[0], K, -1)
        
        # Right Project (Target): Result[..., t] = temp[..., t] @ S.T
        for t_local in range(t_end - t_start):
            mat_t = temp[:, :, t_local] # (N, K)
            res_t = torch.mm(mat_t, S_total.t()) # (N, N)
            fine_adj[:, :, t_start + t_local] = res_t.cpu()

    return fine_adj # Returns CPU tensor (Source, Target, Time)

# ==========================================
# 2. Static Visualization & Metrics
# ==========================================
def evaluate_static_graphs(model, meta, output_dir, metrics=None):
    """
    Generates and saves static visualization plots.
    """
    model.update_hard_mask()
    
    N = meta['coords'].shape[0]
    coords = meta['coords']
    patch_ids = model.patch_labels
    
    # Get Static Adjacencies
    # 【修复】移除了 .T，保持 (Source, Target) 格式与 GT 一致
    est_fine = model.fine_model.graph.get_soft_graph().detach().cpu().numpy()
    
    thresh = metrics.get('Best_Threshold', 0.1) if metrics else 0.1
    est_fine_hard = (est_fine > thresh).astype(float)
    
    gt_fine = meta.get('gt_fine')
    if gt_fine is None: gt_fine = np.zeros_like(est_fine)

    plt.switch_backend('Agg') 
    fig = plt.figure(figsize=(20, 5))
    
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(coords[:, 0], coords[:, 1], c=patch_ids, cmap='tab20', s=50)
    ax1.set_title(f"Learned Clusters (N={len(np.unique(patch_ids))})")

    # Plot GT (Y=Source, X=Target)
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(gt_fine, cmap='Blues', vmin=0)
    ax2.set_title("GT Fine (Y:Src, X:Tgt)")
    ax2.set_xlabel("Target")
    ax2.set_ylabel("Source")

    # Plot Est (Y=Source, X=Target)
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(est_fine, cmap='Reds', vmin=0)
    ax3.set_title("Est Fine (Soft)")
    ax3.set_xlabel("Target")
    ax3.set_ylabel("Source")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.imshow(est_fine_hard, cmap='Greens', vmin=0)
    ax4.set_title(f"Est Fine (Hard > {thresh:.2f})")
    ax4.set_xlabel("Target")
    ax4.set_ylabel("Source")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "result_static.png")
    plt.savefig(save_path)
    plt.close()
    
    return save_path

# ==========================================
# 3. Main Evaluation Orchestrator
# ==========================================
def run_full_evaluation(model, args, accelerator, meta):
    accelerator.print("\n📊 Starting Full Evaluation...")
    model.eval()
    
    # --- 1. Static Analysis ---
    if accelerator.is_main_process:
        # 【修复】移除了 .T
        est_fine = model.fine_model.graph.get_soft_graph().detach().cpu().numpy()
        gt_fine = meta.get('gt_fine')
        
        metrics = {}
        if gt_fine is not None:
            # 这里的输入必须是对齐的 (Source, Target)
            metrics = count_accuracy(gt_fine, est_fine)
            accelerator.print("\n🏆 Static Metrics:", json.dumps(metrics, indent=4))
            wandb.log(metrics)
            with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)

        plot_path = evaluate_static_graphs(model, meta, args.output_dir, metrics)
        wandb.log({"static_result": wandb.Image(plot_path)})
        accelerator.print(f"✅ Static plots saved to {plot_path}")

    accelerator.wait_for_everyone()

    # --- 2. Dynamic Inference (Fast Hierarchical Method) ---
    if accelerator.is_main_process:
        accelerator.print(f"\n🎬 Computing Full Dynamic Causal Graph (Hierarchical Projection Method)...")
        torch.cuda.empty_cache()
        
        try:
            # Load Data
            base_path = getattr(args, 'data_path', 'data/synthetic')
            data_np, _, _ = load_from_disk(base_path, args.dataset, args.replica_id)
            
            # Standardize
            mean = data_np.mean(axis=0)
            std = data_np.std(axis=0) + 1e-5
            data_np = (data_np - mean) / std
            
            # Create sequential loader (Batch=1)
            full_ds = CausalTimeSeriesDataset(data_np, args.window_size, stride=args.window_size, mode='train', split_ratio=1.0)
            full_loader = torch.utils.data.DataLoader(full_ds, batch_size=1, shuffle=False)
            
            all_strengths = []
            max_frames = 200 # Total frames needed
            frames_generated = 0
            
            for batch in tqdm(full_loader, desc="Hierarchical Inference"):
                batch = batch.to(accelerator.device)
                
                # Use the new fast function
                # Returns CPU tensor (N, N, T_chunk)
                curr_adj = compute_hierarchical_dynamic_graph(model, batch, accelerator.device)
                
                all_strengths.append(curr_adj.numpy())
                
                frames_generated += curr_adj.shape[2]
                if frames_generated >= max_frames:
                    break
            
            # Concatenate
            final_dynamic_adj = np.concatenate(all_strengths, axis=2)
            final_dynamic_adj = final_dynamic_adj[:, :, :max_frames]
            
            # Save
            np.save(os.path.join(args.output_dir, "est_dynamic.npy"), final_dynamic_adj)
            accelerator.print(f"✅ Saved dynamic graph shape: {final_dynamic_adj.shape}")

            # Visualization
            # Normalize globally
            final_dynamic_adj = final_dynamic_adj / (final_dynamic_adj.max() + 1e-9)
            
            gif_path = os.path.join(args.output_dir, "causal_evolution.gif")
            # 注意：visualize.py 中的 create_dynamic_gif 会进行一次转置用于绘图
            # 如果那里也需要 Source/Target 对齐，请检查 visualize.py
            create_dynamic_gif(final_dynamic_adj, gif_path, fps=10)
            wandb.log({"dynamic_evolution": wandb.Video(gif_path, fps=10, format="gif")})
            accelerator.print(f"✅ GIF saved to {gif_path}")
            
        except Exception as e:
            accelerator.print(f"❌ Dynamic inference failed: {e}")
            import traceback
            traceback.print_exc()

    accelerator.wait_for_everyone()