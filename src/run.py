import sys
import os
import argparse
import torch
import torch.nn.functional as F
import wandb
import numpy as np
import datetime
from datetime import timedelta
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import set_seed
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import ST_CausalFormer
from dataloader import get_data_context
from evaluate import run_full_evaluation

# ==========================================
# Helpers
# ==========================================
def compute_entropy_loss(S):
    entropy = -torch.sum(S * torch.log(S + 1e-6), dim=-1)
    return entropy.mean()

def compute_balance_loss(S):
    K = S.shape[-1]
    cluster_usage = S.mean(dim=0).mean(dim=0) 
    neg_entropy = torch.sum(cluster_usage * torch.log(cluster_usage + 1e-6))
    max_entropy = np.log(K)
    return neg_entropy + max_entropy

# ==========================================
# Training
# ==========================================
def train_one_epoch(model, loader, optimizer, accelerator, args, epoch):
    model.train() 
    total_loss = 0.0
    
    # Anneal Tau: 1.0 -> 0.1
    tau = max(0.1, np.exp(-epoch * args.tau_decay))
    
    for batch in loader:
        if isinstance(batch, (list, tuple)): x = batch[0]
        else: x = batch
        
        optimizer.zero_grad()
        
        # Unified Forward
        results = model(x, tau=tau)
        
        batch_loss = 0.0
        
        for res in results:
            level = res['level']
            x_rec, x_pred, x_target = res['x_rec'], res['x_pred'], res['x_target']
            S = res['S']
            
            l_rec = F.mse_loss(x_rec, x_target)
            l_pre = F.mse_loss(x_pred, x_target[..., 1:])
            
            batch_loss += (l_rec + l_pre)
            
            if S is not None:
                l_ent = args.lambda_ent * compute_entropy_loss(S)
                l_bal = args.lambda_bal * compute_balance_loss(S)
                batch_loss += l_ent + l_bal

        # Structural Sparsity
        l1_loss = args.lambda_l1 * accelerator.unwrap_model(model).get_structural_l1_loss()
        
        final_loss = batch_loss + l1_loss
        
        accelerator.backward(final_loss)
        optimizer.step()
        total_loss += final_loss.item()
        
    return total_loss / len(loader), tau

# ==========================================
# Main
# ==========================================
def main(args):
    # 1. Setup Accelerator with DDP & Timeout
    # 设置 3 小时超时，防止大图计算时被 Kill
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=180))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = Accelerator(
        log_with="wandb", 
        project_dir=args.output_dir,
        kwargs_handlers=[init_kwargs, ddp_kwargs]
    )
    set_seed(args.seed)
    
    # 2. Timestamped Output Directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, args.dataset, timestamp)
    
    if accelerator.is_main_process:
        if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
        # Init WandB with new run name
        run_name = f"{args.dataset}-{timestamp}"
        accelerator.init_trackers(project_name=args.project_name, config=vars(args), init_kwargs={"wandb": {"name": run_name}})
        accelerator.print(f"🚀 Experiment: {args.dataset} | Output Dir: {args.output_dir}")

    # 3. Load Data
    train_loader, val_loader, meta = get_data_context(args)

    # 4. Init Model
    model = ST_CausalFormer(
        N=args.N, 
        coords=meta['coords'], 
        hierarchy=args.hierarchy, 
        latent_C=args.latent_C, 
        d_model=args.d_model,
        seq_len=args.window_size
    )

    # 5. Inference Only Logic
    if args.inference_only:
        if not args.model_path:
            raise ValueError("❌ --inference_only requires --model_path to be specified!")
        
        accelerator.print(f"📥 Loading pre-trained model from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(accelerator.device)
        
        # Run Evaluation directly
        run_full_evaluation(model, args, accelerator, meta)
        accelerator.end_training()
        return

    # 6. Training Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # 7. Training Loop
    epochs = args.epochs
    progress_bar = tqdm(range(epochs), disable=not accelerator.is_local_main_process, desc="Unified Training")
    
    for ep in progress_bar:
        loss, tau = train_one_epoch(model, train_loader, optimizer, accelerator, args, ep)
        
        if accelerator.is_main_process:
            wandb.log({"train/loss": loss, "tau": tau})
            progress_bar.set_postfix(loss=f"{loss:.4f}", tau=f"{tau:.2f}")

    # 8. Post-Training Wrap-up
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    
    if accelerator.is_main_process:
        # Save Model
        save_path = os.path.join(args.output_dir, "model.pth")
        torch.save(unwrapped_model.state_dict(), save_path)
        accelerator.print(f"💾 Model saved to {save_path}")

    # 9. Run Full Evaluation
    run_full_evaluation(unwrapped_model, args, accelerator, meta)
        
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data & Task
    parser.add_argument("--dataset", type=str, default="lorenz96")
    parser.add_argument("--data_path", type=str, default="data/synthetic")
    parser.add_argument("--replica_id", type=int, default=0)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--norm_coords", action="store_true")
    
    # Model
    parser.add_argument("--hierarchy", type=int, nargs='+', default=[32, 8])
    parser.add_argument("--latent_C", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--k_patches", type=int, default=8) # Backward compatibility if needed
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lambda_ent", type=float, default=1e-3)
    parser.add_argument("--lambda_bal", type=float, default=1.0)
    parser.add_argument("--lambda_l1", type=float, default=2e-3)
    parser.add_argument("--tau_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)

    # Logging & Inference
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--project_name", type=str, default="ST-CausalFormer-Unified")
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    # Inference Only Mode
    parser.add_argument("--inference_only", action="store_true", help="Skip training and run inference only")
    parser.add_argument("--model_path", type=str, default=None, help="Path to .pth file for inference")

    args = parser.parse_args()
    main(args)