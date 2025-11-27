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
# 1. Dynamic Causality Engine (Gradient Based)
# ==========================================
def compute_dynamic_strengths_batch(model, x, device):
    """
    Computes dynamic causal strengths via Gradient Saliency (Jacobian).
    Mathematically precise and much smoother than perturbation.
    Returns: (N_source, N_target, Time)
    """
    model.eval()
    
    # 1. Prepare Input (Require Grad)
    # x: (B, N, T)
    x_in = x.detach().clone().to(device)
    x_in.requires_grad_(True)
    
    # 2. Forward Pass
    # We want the prediction of the Fine Model (Level 0)
    # Unified model returns a list of results.
    results = model(x_in) 
    
    # Find the result for level 0 (Fine)
    # Depending on hierarchy order, usually index 0 is N nodes.
    # We check the shape to be sure.
    fine_result = None
    for res in results:
        if res['x_pred'].shape[1] == x.shape[1]:
            fine_result = res
            break
            
    if fine_result is None:
        raise ValueError("Could not find fine-level prediction in model output.")

    x_pred = fine_result['x_pred'] # (B, N, T-1)
    
    B, N, T_pred = x_pred.shape
    
    # 3. Compute Saliency (Jacobian)
    # Saliency[j, i, t] = | d(x_pred[t, i]) / d(x_in[t, j]) |
    # Loop over Target Node i to keep memory manageable
    
    dynamic_adj = torch.zeros(N, N, T_pred, device=device)
    
    for target_i in range(N):
        # We sum over time to scalarize. 
        # Since the model is causal (Masked), x_pred[t] ONLY depends on x_in[0...t].
        # So d(Sum_Pred)/d(x_in[t]) effectively captures the influence at time t.
        target_signal = x_pred[:, target_i, :].sum()
        
        # Compute Gradients
        # retain_graph=True is needed because we backprop multiple times (for each target)
        # But to save memory, we can re-forward? No, N=128 is small enough for retain_graph usually.
        # If OOM occurs, we can move forward inside the loop.
        grads = torch.autograd.grad(target_signal, x_in, retain_graph=True)[0] # (B, N, T)
        
        # grads: (B, Source_N, T)
        # We align time: Input at t affects Pred at t (which is x_{t+1})
        # So we take grads[..., :-1] to match x_pred length
        
        # Take absolute value as "Strength"
        # Average over batch (usually B=1 during inference)
        dynamic_adj[:, target_i, :] = grads[0, :, :-1].abs()

    # Cleanup graph
    x_in.grad = None
    
    return dynamic_adj

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
    est_fine = model.fine_model.graph.get_soft_graph().detach().cpu().numpy().T
    
    thresh = metrics.get('Best_Threshold', 0.1) if metrics else 0.1
    est_fine_hard = (est_fine > thresh).astype(float)
    
    gt_fine = meta.get('gt_fine')
    if gt_fine is None: gt_fine = np.zeros_like(est_fine)

    plt.switch_backend('Agg') 
    fig = plt.figure(figsize=(20, 5))
    
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(coords[:, 0], coords[:, 1], c=patch_ids, cmap='tab20', s=50)
    ax1.set_title(f"Learned Clusters (N={len(np.unique(patch_ids))})")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(gt_fine, cmap='Blues', vmin=0)
    ax2.set_title("GT Fine")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(est_fine, cmap='Reds', vmin=0)
    ax3.set_title("Est Fine (Soft)")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.imshow(est_fine_hard, cmap='Greens', vmin=0)
    ax4.set_title(f"Est Fine (Hard > {thresh:.2f})")

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
    est_fine = model.fine_model.graph.get_soft_graph().detach().cpu().numpy().T
    gt_fine = meta.get('gt_fine')
    
    metrics = {}
    if gt_fine is not None:
        metrics = count_accuracy(gt_fine, est_fine)
        accelerator.print("\n🏆 Static Metrics:", json.dumps(metrics, indent=4))
        if accelerator.is_main_process:
            wandb.log(metrics)
            with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)

    # --- 2. Static Plotting ---
    if accelerator.is_main_process:
        plot_path = evaluate_static_graphs(model, meta, args.output_dir, metrics)
        wandb.log({"static_result": wandb.Image(plot_path)})
        accelerator.print(f"✅ Static plots saved to {plot_path}")

    # --- 3. Dynamic Inference (Gradient Based) ---
    if accelerator.is_main_process:
        accelerator.print("\n🎬 Computing Full Dynamic Causal Graph (Gradient Method)...")
        # Clear cache for heavy gradient computation
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
            # Use smaller window stride to get smoother animation if needed, 
            # but window_size stride is faster.
            full_ds = CausalTimeSeriesDataset(data_np, args.window_size, stride=args.window_size, mode='train', split_ratio=1.0)
            full_loader = torch.utils.data.DataLoader(full_ds, batch_size=1, shuffle=False)
            
            all_strengths = []
            
            # Limit number of frames to generate to avoid waiting forever
            max_frames = 200 
            frames_generated = 0
            
            for batch in tqdm(full_loader, desc="Calculating Gradients"):
                batch = batch.to(accelerator.device) # (1, N, T)
                
                # Compute Gradient Saliency
                curr_adj = compute_dynamic_strengths_batch(model, batch, accelerator.device)
                
                # curr_adj: (N, N, T-1) -> numpy
                curr_np = curr_adj.detach().cpu().numpy()
                all_strengths.append(curr_np)
                
                frames_generated += curr_np.shape[2]
                if frames_generated >= max_frames:
                    break
            
            # Concatenate
            final_dynamic_adj = np.concatenate(all_strengths, axis=2)
            # Clip to max frames
            final_dynamic_adj = final_dynamic_adj[:, :, :max_frames]
            
            # Normalize for Visualization clarity
            # Gradient values can be small, so we normalize by max value
            final_dynamic_adj = final_dynamic_adj / (final_dynamic_adj.max() + 1e-9)
            
            np.save(os.path.join(args.output_dir, "est_dynamic.npy"), final_dynamic_adj)
            
            # Create GIF
            gif_path = os.path.join(args.output_dir, "causal_evolution.gif")
            create_dynamic_gif(final_dynamic_adj, gif_path, fps=10)
            
            wandb.log({"dynamic_evolution": wandb.Video(gif_path, fps=10, format="gif")})
            accelerator.print(f"✅ GIF saved to {gif_path}")
            
        except Exception as e:
            accelerator.print(f"❌ Dynamic inference failed: {e}")
            import traceback
            traceback.print_exc()

    accelerator.wait_for_everyone()