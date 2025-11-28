import os
import torch
import torch.nn.functional as F
from torch.func import jacrev, functional_call
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
    计算动态因果强度 (Jacobian based)
    """
    model.eval()
    x_in = x.detach().clone().to(device)  # (1, N, T)
    x_squeeze = x_in.squeeze(0)           # (N, T)
    
    def model_func(input_tensor, params, buffers):
        batched_input = input_tensor.unsqueeze(0)
        # 提取 Level 0 (Fine) 的预测
        results = functional_call(model, (params, buffers), (batched_input,))
        fine_result = results[0]
        x_pred = fine_result['x_pred'].squeeze(0)
        return x_pred.sum(dim=1)

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    # Jacobian: (N_target, N_source, T_source)
    jacobian = jacrev(model_func, argnums=0)(x_squeeze, params, buffers)
    
    # 调整维度并去除最后无效帧
    dynamic_adj = jacobian.permute(1, 0, 2).abs()
    return dynamic_adj[..., :-1]

def evaluate_static_graphs(model, meta, output_dir, metrics=None):
    """
    绘制静态分析图 (Swin Windows, GT, Est Soft, Est Hard)
    """
    # 1. 提取 Fine Graph (Level 0)
    est_fine = model.layers[0].graph.get_soft_graph().detach().cpu().numpy()
    
    # 2. 提取 Swin Window 划分 (Level 0 -> 1 Pooler)
    if hasattr(model, 'structure_S_0'):
        S_matrix = model.structure_S_0.detach().cpu().numpy() # (N, K)
        patch_ids = S_matrix.argmax(axis=1)
    else:
        patch_ids = np.zeros(est_fine.shape[0])

    # 硬阈值化
    thresh = metrics.get('Best_Threshold', 0.1) if metrics else 0.1
    est_fine_hard = (est_fine > thresh).astype(float)
    
    gt_fine = meta.get('gt_fine', np.zeros_like(est_fine))
    coords = meta['coords']
    
    # 绘图
    plt.switch_backend('Agg') 
    fig = plt.figure(figsize=(20, 5))
    
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.scatter(coords[:, 0], coords[:, 1], c=patch_ids, cmap='tab20', s=20)
    ax1.set_title(f"Swin Windows (K={len(np.unique(patch_ids))})")

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

def run_full_evaluation(model, args, accelerator, meta):
    accelerator.print("\n📊 Starting Full Evaluation...")
    model.eval()
    
    # 1. 静态图评估
    if accelerator.is_main_process:
        est_fine = model.layers[0].graph.get_soft_graph().detach().cpu().numpy()
        gt_fine = meta.get('gt_fine')
        
        metrics = {}
        if gt_fine is not None:
            metrics = count_accuracy(gt_fine, est_fine)
            accelerator.print("Static Metrics:", json.dumps(metrics, indent=4))
            wandb.log(metrics)
            with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=4)

        plot_path = evaluate_static_graphs(model, meta, args.output_dir, metrics)
        wandb.log({"static_result": wandb.Image(plot_path)})

    accelerator.wait_for_everyone()

    # 2. 动态图推断 (仅主进程)
    if accelerator.is_main_process:
        accelerator.print(f"\nComputing Dynamic Graph (Jacobian)...")
        try:
            base_path = getattr(args, 'data_path', 'data/synthetic')
            data_np, _, _ = load_from_disk(base_path, args.dataset, args.replica_id)
            mean = data_np.mean(axis=0)
            std = data_np.std(axis=0) + 1e-5
            data_np = (data_np - mean) / std
            
            full_ds = CausalTimeSeriesDataset(
                data_np, args.window_size, stride=args.window_size - 1, mode='train', split_ratio=1.0
            )
            full_loader = torch.utils.data.DataLoader(full_ds, batch_size=1, shuffle=False)
            
            all_strengths = []
            max_frames = 100 
            frames_generated = 0
            
            for batch in tqdm(full_loader, desc="Dynamic Inference"):
                batch = batch.to(accelerator.device)
                curr_adj = compute_dynamic_strengths(model, batch, accelerator.device)
                all_strengths.append(curr_adj.detach().cpu().numpy())
                
                frames_generated += curr_adj.shape[2]
                if frames_generated >= max_frames: break
            
            final_dynamic_adj = np.concatenate(all_strengths, axis=2)[:, :, :max_frames]
            
            np.save(os.path.join(args.output_dir, "est_dynamic.npy"), final_dynamic_adj)
            
            # GIF Visualization
            gif_path = os.path.join(args.output_dir, "causal_evolution.gif")
            vis_adj = final_dynamic_adj / (final_dynamic_adj.max() + 1e-9)
            create_dynamic_gif(vis_adj, gif_path, fps=8)
            wandb.log({"dynamic_evolution": wandb.Video(gif_path, fps=8, format="gif")})
            accelerator.print(f"✅ GIF saved to {gif_path}")
            
        except Exception as e:
            accelerator.print(f"❌ Dynamic inference failed: {e}")

    accelerator.wait_for_everyone()