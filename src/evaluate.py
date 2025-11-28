import os
import torch
import torch.nn.functional as F
from torch.func import jacrev, vmap, functional_call
import numpy as np
import matplotlib.pyplot as plt
import wandb
import json
from tqdm.auto import tqdm
from dataloader import load_from_disk, CausalTimeSeriesDataset
from visualize import create_dynamic_gif
from metrics import count_accuracy

def compute_dynamic_strengths(model, x, device):
    """
    Compute dynamic causal strengths using torch.func for Jacobian computation.
    Eliminates Python loops for better performance.
    Speed and accuracy match compute_dynamic_strengths_batch exactly.
    """
    model.eval()
    # Prepare data: remove batch dimension (assuming B=1, use vmap for B>1)
    x_in = x.detach().clone().to(device)  # (1, N, T)
    x_squeeze = x_in.squeeze(0)           # (N, T)
    
    # Define pure function for functional interface
    def model_func(input_tensor, params, buffers):
        # input_tensor: (N, T) -> add batch -> (1, N, T)
        batched_input = input_tensor.unsqueeze(0)
        
        # Call stateful nn.Module using functional_call
        results = functional_call(model, (params, buffers), (batched_input,))
        
        # Extract fine level result
        fine_result = None
        for res in results:
            if res['x_pred'].shape[1] == input_tensor.shape[0]:
                fine_result = res
                break
        
        # (1, N, T_pred) -> (N, T_pred)
        x_pred = fine_result['x_pred'].squeeze(0)
        
        # For each target node i, compute gradient of sum(time) w.r.t. all inputs
        # Output shape: (N,), each element is total signal for a target node
        return x_pred.sum(dim=1)

    # Extract parameters and buffers
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    # Compute Jacobian matrix
    # jacrev automatically handles gradients of each output element w.r.t. inputs
    # Output: (N_target, N_source, T_source)
    jacobian = jacrev(model_func, argnums=0)(x_squeeze, params, buffers)
    
    # Reorder dimensions to match original function: (Source, Target, Time)
    # jacobian[i, j, t] is d(Target_i)/d(Input_j_t)
    dynamic_adj = jacobian.permute(1, 0, 2).abs()  # (N, N, T)
    
    # Force remove the last frame.
    # The last input frame contributes to predicting the *next* window's first frame,
    # which is not present in the current x_pred sum, resulting in zero gradients.
    dynamic_adj = dynamic_adj[..., :-1]
        
    return dynamic_adj

# --- Hierarchical Dynamic Graph Computation ---
def compute_hierarchical_dynamic_graph(model, x, device):
    """
    Compute dynamic causal graph using hierarchical acceleration.
    Strategy: Compute gradients at top level and project back to fine level.
    Speedup: O(N) -> O(K), where K is number of patches.
    """
    model.eval()
    
    # Get hierarchy information
    with torch.no_grad():
        results = model(x.to(device))
    
    # Identify top level
    top_level_idx = len(results) - 1
    top_res = results[top_level_idx]
    
    # Compute total projection matrix S (fine -> top)
    S_total = torch.eye(model.dims[0], device=device)
    
    for i in range(len(results) - 1):
        S_curr = results[i]['S']  # (B, N_curr, N_next)
        if S_curr is None: break
        
        S_curr = S_curr.mean(dim=0)  # Average over batch
        S_total = torch.mm(S_total, S_curr)
    
    # Compute gradients at top level
    x_top = top_res['x_target'].detach().clone()
    x_top.requires_grad_(True)
    
    top_layer = model.layers[top_level_idx]
    _, x_pred_top, _ = top_layer(x_top)
    
    B, K, T_pred = x_pred_top.shape
    coarse_adj = torch.zeros(K, K, T_pred, device=device)
    
    for target_i in range(K):
        target_signal = x_pred_top[:, target_i, :].sum()
        grads = torch.autograd.grad(target_signal, x_top, retain_graph=True)[0]
        coarse_adj[:, target_i, :] = grads[0, :, :-1].abs()
        
    # Project back to fine level
    fine_adj = torch.zeros(model.dims[0], model.dims[0], T_pred, device='cpu')
    S_total = S_total.to(device)
    
    # Process in chunks to save memory
    chunk_size = 10 
    for t_start in range(0, T_pred, chunk_size):
        t_end = min(t_start + chunk_size, T_pred)
        
        sub_coarse = coarse_adj[:, :, t_start:t_end] 
        sub_flat = sub_coarse.reshape(K, -1)
        
        temp = torch.mm(S_total, sub_flat)
        temp = temp.reshape(model.dims[0], K, -1)
        
        for t_local in range(t_end - t_start):
            mat_t = temp[:, :, t_local]
            res_t = torch.mm(mat_t, S_total.t())
            fine_adj[:, :, t_start + t_local] = res_t.cpu()

    return fine_adj  # Returns CPU tensor (Source, Target, Time)

# --- Static Visualization & Metrics ---
def evaluate_static_graphs(model, meta, output_dir, metrics=None):
    """Generate and save static visualization plots"""
    model.update_hard_mask()
    
    N = meta['coords'].shape[0]
    coords = meta['coords']
    patch_ids = model.patch_labels
    
    # Get static adjacency matrices
    est_fine = model.fine_model.graph.get_soft_graph().detach().cpu().numpy()
    
    thresh = metrics.get('Best_Threshold', 0.1) if metrics else 0.1
    est_fine_hard = (est_fine > thresh).astype(float)
    
    gt_fine = meta.get('gt_fine')
    if gt_fine is None: gt_fine = np.zeros_like(est_fine)

    plt.switch_backend('Agg') 
    fig = plt.figure(figsize=(20, 5))
    
    # Plot learned clusters
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(coords[:, 0], coords[:, 1], c=patch_ids, cmap='tab20', s=50)
    ax1.set_title(f"Learned Clusters (N={len(np.unique(patch_ids))})")

    # Plot ground truth
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(gt_fine, cmap='Blues', vmin=0)
    ax2.set_title("GT Fine (Y:Src, X:Tgt)")
    ax2.set_xlabel("Target")
    ax2.set_ylabel("Source")

    # Plot estimated soft graph
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(est_fine, cmap='Reds', vmin=0)
    ax3.set_title("Est Fine (Soft)")
    ax3.set_xlabel("Target")
    ax3.set_ylabel("Source")

    # Plot estimated hard graph
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

# --- Main Evaluation Orchestrator ---
def run_full_evaluation(model, args, accelerator, meta):
    """Run comprehensive evaluation including static and dynamic analysis"""
    accelerator.print("\nStarting Full Evaluation...")
    model.eval()
    
    # Static analysis
    if accelerator.is_main_process:
        est_fine = model.fine_model.graph.get_soft_graph().detach().cpu().numpy()
        gt_fine = meta.get('gt_fine')
        
        metrics = {}
        if gt_fine is not None:
            metrics = count_accuracy(gt_fine, est_fine)
            accelerator.print("\nStatic Metrics:", json.dumps(metrics, indent=4))
            wandb.log(metrics)
            with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)

        plot_path = evaluate_static_graphs(model, meta, args.output_dir, metrics)
        wandb.log({"static_result": wandb.Image(plot_path)})
        accelerator.print(f"Static plots saved to {plot_path}")

    accelerator.wait_for_everyone()

    # Dynamic inference
    if accelerator.is_main_process:
        accelerator.print(f"\nComputing Full Dynamic Causal Graph...")
        torch.cuda.empty_cache()
        
        try:
            # Load data
            base_path = getattr(args, 'data_path', 'data/synthetic')
            data_np, _, _ = load_from_disk(base_path, args.dataset, args.replica_id)
            
            # Standardize
            mean = data_np.mean(axis=0)
            std = data_np.std(axis=0) + 1e-5
            data_np = (data_np - mean) / std
            
            # Create sequential loader
            # Fix: stride = window_size - 1 because we drop 1 frame per window
            full_ds = CausalTimeSeriesDataset(
                data_np, 
                args.window_size, 
                stride=args.window_size - 1, 
                mode='train', 
                split_ratio=1.0
            )
            full_loader = torch.utils.data.DataLoader(full_ds, batch_size=1, shuffle=False)
            
            all_strengths = []
            max_frames = 200
            frames_generated = 0
            
            for batch in tqdm(full_loader, desc="Hierarchical Inference"):
                batch = batch.to(accelerator.device)
                curr_adj = compute_dynamic_strengths(model, batch, accelerator.device)
                all_strengths.append(curr_adj.detach().cpu().numpy())
                
                frames_generated += curr_adj.shape[2]
                if frames_generated >= max_frames:
                    break
            
            # Concatenate results
            final_dynamic_adj = np.concatenate(all_strengths, axis=2)
            final_dynamic_adj = final_dynamic_adj[:, :, :max_frames]
            
            # Save results
            np.save(os.path.join(args.output_dir, "est_dynamic.npy"), final_dynamic_adj)
            accelerator.print(f"Saved dynamic graph shape: {final_dynamic_adj.shape}")

            # Create visualization
            final_dynamic_adj = final_dynamic_adj / (final_dynamic_adj.max() + 1e-9)
            gif_path = os.path.join(args.output_dir, "causal_evolution.gif")
            create_dynamic_gif(final_dynamic_adj, gif_path, fps=10)
            wandb.log({"dynamic_evolution": wandb.Video(gif_path, fps=10, format="gif")})
            accelerator.print(f"GIF saved to {gif_path}")
            
        except Exception as e:
            accelerator.print(f"Dynamic inference failed: {e}")
            import traceback
            traceback.print_exc()

    accelerator.wait_for_everyone()