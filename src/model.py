import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .tcn import SimpleTCN

class UnCLENet(nn.Module):
    def __init__(
        self,
        N: int,
        latent_C: int = 16,
        tcn_levels_unc: int = 4,
        tcn_hidden_unc: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
        activation: nn.Module = nn.Tanh() # 论文常用 Tanh 或 ReLU
    ):
        super().__init__()
        self.N = N
        self.C = latent_C
        
        # --- Uncoupler ---
        # 输入单变量 (C_in=1)，输出 Hidden
        self.unc_tcn = SimpleTCN(in_channels=1, num_levels=tcn_levels_unc, 
                                 hidden_channels=tcn_hidden_unc, kernel_size=kernel_size, dropout=dropout)
        self.unc_proj = nn.Linear(tcn_hidden_unc, latent_C)

        # --- Recoupler ---
        # 输入 Latent (C_in=latent_C)，输出 Hidden -> 1
        self.rec_tcn = SimpleTCN(in_channels=latent_C, num_levels=tcn_levels_unc, 
                                 hidden_channels=tcn_hidden_unc, kernel_size=kernel_size, dropout=dropout)
        self.rec_proj = nn.Linear(tcn_hidden_unc, 1)

        # --- Dependency Matrices (Psi) ---
        # 优化点：初始化为较小的随机数，避免初始 Loss 过大，利于 L1 稀疏化
        self.Psi = nn.Parameter(torch.randn(latent_C, N, N) * 0.02)
        
        self.latent_act = activation

    def uncouple(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T) -> z: (B, N, T, C)
        """
        B, N, T = x.shape
        # Flatten N into Batch: (B*N, T, 1)
        x_flat = x.view(B * N, T).unsqueeze(-1)
        z_hidden = self.unc_tcn(x_flat)      # (B*N, T, hidden)
        z = self.unc_proj(z_hidden)          # (B*N, T, C)
        return z.view(B, N, T, self.C)

    def recouple(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, N, T, C) -> x_recon: (B, N, T)
        """
        B, N, T, C = z.shape
        # Flatten N into Batch: (B*N, T, C)
        z_flat = z.view(B * N, T, C)
        h = self.rec_tcn(z_flat)             # (B*N, T, hidden)
        xhat = self.rec_proj(h).squeeze(-1)  # (B*N, T)
        return xhat.view(B, N, T)

    def predict_latent_next(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, N, T, C)
        return zhat_next: (B, N, T, C) -> 对应 z_{t+1} 的预测
        """
        # Einstein Summation: (Batch, Time, Channel, Node) x (Channel, Node_out, Node_in)
        # z: (B, N, T, C) -> permute -> (B, T, C, N)
        z_perm = z.permute(0, 2, 3, 1) 
        
        # Psi: (C, N, N) -> (C, To, From)
        # z_perm: ..., From
        # Output: ..., To
        # equation: 'btcn,cnm->btcm' (n is source node, m is target node)
        # 注意：Psi 的维度定义需统一。这里假设 Psi[c, i, j] 表示 j -> i 的权重 (i是行，j是列)
        # 那么乘法应该是 Matrix @ Vector。
        # z_perm[..., N] 是列向量。Psi @ z。
        zhat = torch.einsum('btcn,cnm->btcm', z_perm, self.Psi)
        
        zhat = self.latent_act(zhat)
        # 恢复维度 (B, T, C, N) -> (B, N, T, C)
        return zhat.permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.uncouple(x)              # (B, N, T, C)
        x_recon = self.recouple(z)        # (B, N, T)
        
        zhat_next = self.predict_latent_next(z) # (B, N, T, C), zhat_next[:, t] is pred for t+1
        
        # 切片对齐：我们预测的是 x_{t+1}
        # 使用 z_{0...T-2} 预测出的 latent 来重构 x_{1...T-1}
        x_pred = self.recouple(zhat_next[..., :-1, :]) 
        
        return x_recon, x_pred, z

    def psi_l1(self):
        return torch.sum(torch.abs(self.Psi))

    @torch.no_grad()
    def static_causal_aggregation(self) -> torch.Tensor:
        # L2 Norm aggregation across channels
        # Psi shape: (C, Target, Source) -> return (Target, Source)
        return torch.sqrt(torch.mean(self.Psi ** 2, dim=0))

    @torch.no_grad()
    def dynamic_causal_strengths(self, x: torch.Tensor, perm_seed: Optional[int] = 42):
        """
        高效扰动推断
        """
        B, N, T = x.shape
        _, x_pred_orig, z_orig = self.forward(x)
        
        # Baseline Error (aligned to t=1..T-1)
        eps_orig = (x_pred_orig - x[..., 1:]) ** 2 
        eps_orig_mean = eps_orig.mean(dim=0) # (N, T-1)

        strengths = torch.zeros(N, N, T-1, device=x.device)

        for j in range(N):
            # 1. 扰动变量 j
            if perm_seed is not None:
                torch.manual_seed(perm_seed)
            
            x_j_pert = x[:, j, :].clone()
            perm_idx = torch.randperm(T, device=x.device)
            x_j_pert = x_j_pert[:, perm_idx]
            
            # 2. 仅重新计算 j 的 Latent
            z_j_new = self.uncouple(x_j_pert.unsqueeze(1)).squeeze(1) # (B, T, C)
            
            # 3. 混合 Latent
            z_mixed = z_orig.clone()
            z_mixed[:, j, :, :] = z_j_new
            
            # 4. 预测与重构
            zhat_mixed = self.predict_latent_next(z_mixed)
            x_pred_mixed = self.recouple(zhat_mixed[..., :-1, :])
            
            # 5. 计算误差增益
            eps_pert = (x_pred_mixed - x[..., 1:]) ** 2
            eps_pert_mean = eps_pert.mean(dim=0)
            
            # j -> i 的强度
            delta = F.relu(eps_pert_mean - eps_orig_mean)
            strengths[j, :, :] = delta # (Source j, Target i, Time)

        return strengths