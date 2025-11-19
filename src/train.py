import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm
from .model import UnCLENet
from .utils import get_logger

def train_model(
    model: UnCLENet,
    dataloader,
    device,
    output_dir,
    config  # 传入 config 对象
):
    logger = get_logger(output_dir)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    logger.info(f"Start Training... Device: {device}")
    logger.info(f"Config: {config}")
    
    # --- Stage 1: Reconstruction Pretraining ---
    logger.info("Stage 1: Reconstruction Pretraining")
    
    # 使用 tqdm 包装 range
    pbar_recon = tqdm(range(config.recon_epochs), desc="[Stage 1: Recon]", unit="epoch")
    
    for epoch in pbar_recon:
        model.train()
        total_loss = 0
        
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            
            z = model.uncouple(x)
            x_recon = model.recouple(z)
            
            loss = F.mse_loss(x_recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        
        # 更新进度条后缀
        pbar_recon.set_postfix({'MSE': f'{avg_loss:.6f}'})
        
        # 偶尔记录到文件，避免日志太长
        if (epoch + 1) % 50 == 0:
            logger.info(f"[Recon] Ep {epoch+1} | MSE: {avg_loss:.6f}")

    # --- Stage 2: Joint Training ---
    logger.info("Stage 2: Joint Training (Recon + Pred + L1)")
    
    pbar_joint = tqdm(range(config.joint_epochs), desc="[Stage 2: Joint]", unit="epoch")
    
    for epoch in pbar_joint:
        model.train()
        epoch_recon = 0
        epoch_pred = 0
        epoch_l1 = 0
        
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            
            x_recon, x_pred, _ = model.forward(x)
            
            loss_recon = F.mse_loss(x_recon, x)
            loss_pred = F.mse_loss(x_pred, x[..., 1:])
            loss_l1 = config.lambda_l1 * model.psi_l1()
            
            loss_total = loss_recon + config.alpha * loss_pred + loss_l1
            
            loss_total.backward()
            optimizer.step()
            
            epoch_recon += loss_recon.item()
            epoch_pred += loss_pred.item()
            epoch_l1 += loss_l1.item()

        avg_recon = epoch_recon / len(dataloader)
        avg_pred = epoch_pred / len(dataloader)
        avg_l1 = epoch_l1 / len(dataloader)
        
        # 进度条显示关键指标
        pbar_joint.set_postfix({
            'Recon': f'{avg_recon:.4f}', 
            'Pred': f'{avg_pred:.4f}', 
            'L1': f'{avg_l1:.4f}'
        })

        if (epoch + 1) % 50 == 0:
            logger.info(f"[Joint] Ep {epoch+1} | Recon: {avg_recon:.5f} | Pred: {avg_pred:.5f} | L1: {avg_l1:.5f}")
            
    # Save Model
    save_path = os.path.join(output_dir, "model.pth")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
    
    return model