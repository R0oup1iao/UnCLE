# src/unclenet.py
"""
UnCLENet - PyTorch implementation sketch following:
Bi et al., "UnCLe: Towards Scalable Dynamic Causal Discovery in Non-linear Temporal Systems" (NeurIPS 2025)
Implements:
 - Shared Uncoupler (TCN encoder) and Recoupler (TCN decoder)
 - C dependency matrices Psi (shape [C, N, N]) for auto-regressive latent prediction
 - Two-stage training: reconstruction pretrain, then joint (recon + alpha * pred + L1)
 - Post-hoc temporal permutation perturbation for dynamic causal strengths
References in paper: LRecon, LPred, LL1, perturbation-based Δε. 
"""
from typing import List, Tuple, Optional
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Minimal causal TCN blocks (adapted pattern used in many TCN implementations)
# ------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, padding, dropout=0.0):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class SimpleTCN(nn.Module):
    """
    A compact TCN usable as Uncoupler / Recoupler.
    Input shapes:
      - If processing univariate series: (batch, seq_len, channels_in) or (batch, channels_in, seq_len).
    This TCN returns features in (batch, seq_len, out_channels) (transposed).
    """
    def __init__(self, in_channels:int, num_levels:int=4, hidden_channels:int=32, kernel_size:int=3, dropout:float=0.0):
        super().__init__()
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden_channels
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_ch, hidden_channels, kernel_size, dilation, padding, dropout))
        self.network = nn.Sequential(*layers)
        # final conv to produce out channels per time step
        self.final = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)

    def forward(self, x):
        """
        强制要求输入为 (B, T, C)
        """
        if x.dim() != 3:
            raise ValueError(f"SimpleTCN expected input (B, T, C), got {x.shape}")

        B, T, C = x.shape
        x = x.transpose(1, 2)     # (B, C, T)

        y = self.network(x)       # (B, hidden, T)
        y = self.final(y)         # (B, hidden, T)
        return y.transpose(1, 2)  # (B, T, hidden)


# ------------------------------
# UnCLENet
# ------------------------------
class UnCLENet(nn.Module):
    """
    Implements the core UnCLe components:
      - Shared Uncoupler (TCN encoder): maps each variable's univariate series to latent (T x C)
      - Shared Recoupler (TCN decoder): maps latent (T x C) back to reconstructed univariate series
      - Dependency matrices Psi: learnable parameters shape (C, N, N), used for predicting next-step latent across variables
    Input assumptions:
      x: tensor (batch, N, T)  (each of N variables has length T). Values are floats.
    Outputs:
      - recon_x: (batch, N, T)
      - pred_x: (batch, N, T)  predictions for t+1 (last column unused)
    """
    def __init__(
        self,
        N: int,
        latent_C: int = 16,
        tcn_levels_unc: int = 4,
        tcn_hidden_unc: int = 64,
        kernel_size:int = 3,
        dropout:float = 0.2,
        activation:nn.Module = nn.ReLU()
    ):
        super().__init__()
        self.N = N
        self.C = latent_C
        # Uncoupler: maps univariate series -> per-var latent channels (T x C)
        # We'll use a shared TCN that expects per-var input channels=1 and returns features hidden-> then a linear to C
        self.unc_tcn = SimpleTCN(in_channels=1, num_levels=tcn_levels_unc, hidden_channels=tcn_hidden_unc, kernel_size=kernel_size, dropout=dropout)
        # map tcn hidden -> C channels per time
        self.unc_proj = nn.Linear(tcn_hidden_unc, latent_C)

        # Recoupler: maps per-var latent (T x C) -> reconstructed univariate series
        # We'll treat latent channels as multi-channel input (C), feed through a TCN that expects C channels and returns hidden->linear to 1
        self.rec_tcn = SimpleTCN(in_channels=latent_C, num_levels=tcn_levels_unc, hidden_channels=tcn_hidden_unc, kernel_size=kernel_size, dropout=dropout)
        self.rec_proj = nn.Linear(tcn_hidden_unc, 1)

        # Dependency matrices: C matrices each N x N (predict next-step latent per channel)
        # Implement as parameter of shape (C, N, N)
        self.Psi = nn.Parameter(torch.randn(latent_C, N, N) * 0.01)

        # activation for latent prediction
        self.latent_act = activation

    def uncouple(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, T)
        returns z: (B, N, T, C)
        """
        B, N, T = x.shape
        # process each variable via shared TCN: vectorize by flattening batch* N
        x_flat = x.view(B * N, T).unsqueeze(-1)         # (B*N, T, 1)
        # TCN expects (B, T, C_in) or (B, C_in, T) - our SimpleTCN accepts (B, T, C_in)
        z_hidden = self.unc_tcn(x_flat)                 # (B*N, T, hidden)
        z = self.unc_proj(z_hidden)                     # (B*N, T, C)
        z = z.view(B, N, T, self.C)
        return z

    def recouple(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, N, T, C)
        returns x_recon: (B, N, T)
        """
        B, N, T, C = z.shape
        z_flat = z.view(B * N, T, C)
        h = self.rec_tcn(z_flat)                        # (B*N, T, hidden)
        xhat = self.rec_proj(h).squeeze(-1)             # (B*N, T)
        xhat = xhat.view(B, N, T)
        return xhat

    def predict_latent_next(self, z: torch.Tensor) -> torch.Tensor:
        """
        Auto-regressive prediction in latent space using Psi:
          For each channel c:
            zhat_c[:, t+1] = sigma( Psi_c @ zc[:, t] )
        Implementation vectorized:
          - z: (B, N, T, C)
        returns:
          zhat: (B, N, T, C)  -- predictions aligned so zhat[..., t+1] is predicted from z[..., t]
          Note: last time index (t= T-1) will produce zhat[..., T] (unused), and we will compare for t in [0..T-2] -> targets t+1
        """
        B, N, T, C = z.shape
        # 重排维度: (B, N, T, C) -> (B, T, C, N)
        z_perm = z.permute(0, 2, 3, 1)  # (B, T, C, N)
        
        # 向量化矩阵乘法: (B, T, C, N) @ (C, N, N) -> (B, T, C, N)
        # 使用einsum: 'btcn,cnm->btcm'
        zhat = torch.einsum('btcn,cnm->btcm', z_perm, self.Psi)
        
        # 应用激活函数
        zhat = self.latent_act(zhat)
        
        # 恢复原始维度: (B, T, C, N) -> (B, N, T, C)
        zhat = zhat.permute(0, 3, 1, 2).contiguous()
        return zhat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward used for training:
          x: (B, N, T)
        returns:
          x_recon: (B, N, T)
          x_pred:  (B, N, T)  predictions for t+1 aligned with target at t+1 (first time step of x_pred corresponds to prediction for t=0 -> t+1)
          z:        (B, N, T, C)
        """
        z = self.uncouple(x)             # (B,N,T,C)
        x_recon = self.recouple(z)       # (B,N,T)
        zhat = self.predict_latent_next(z)   # (B,N,T,C)
        # map predicted latents for each var back to original space via recoupler
        xhat = self.recouple(zhat[..., :-1, :])       # (B,N,T)
        # xhat[..., t] is prediction for x[..., t] computed from previous latent at t-1;
        # therefore for loss alignment, treat predictions at indices 1..T-1 as predictions of x[...,1..T-1]
        return x_recon, xhat, z

    # --------------------------
    # Utility for L1 on Psi
    # --------------------------
    def psi_l1(self):
        return torch.sum(torch.abs(self.Psi))

    # --------------------------
    # Post-hoc perturbation (temporal permutation for variable j)
    # --------------------------
    # @torch.no_grad()
    # def perturb_and_predict(self, x: torch.Tensor, var_idx: int, perm_seed: Optional[int] = None) -> torch.Tensor:
    #     """
    #     Permute time ordering of variable var_idx (same permutation across batch),
    #     then return predictions xhat_perturbed (B,N,T) from the trained model.
    #     """
    #     B, N, T = x.shape
    #     if perm_seed is not None:
    #         torch.manual_seed(perm_seed)
    #     perm = torch.randperm(T)
    #     x_pert = x.clone()
    #     # apply same permutation across batch for variable var_idx
    #     x_pert[:, var_idx, :] = x_pert[:, var_idx, :][:, perm]
    #     # forward
    #     _, xhat_pert, _ = self.forward(x_pert)
    #     return xhat_pert

    @torch.no_grad()
    def dynamic_causal_strengths(self, x: torch.Tensor, perturb_mode='permute', perm_seed:Optional[int]=42):
        """
        Compute Δε^{\\j}_{i,t} = max(0, eps_pert - eps_orig)
        Returns matrix: strengths (N, N, T) where strengths[j, i, t] is effect on i at t caused by perturbing j.
        (Paper defines Ahat_t,Pert_{j,i} from ∆ε^j_{i,t}). See paper eqs. (8)-(10). :contentReference[oaicite:3]{index=3}
        """
        B, N, T = x.shape
        # get baseline prediction
        x_recon, x_pred, z_orig = self.forward(x)   # (B,N,T)
        eps_orig = (x_pred - x[..., 1:]) ** 2      # (B,N,T-1)
        # We'll average across batch dimension to get per-timepoint mean error
        eps_orig_mean = eps_orig.mean(dim=0)  # (N,T-1)

        strengths = torch.zeros(N, N, T-1, device=x.device)
        for j in range(N):
            if perm_seed is not None:
                torch.manual_seed(perm_seed)
            
            # 仅对变量 j 进行扰动
            x_j_perturbed = x[:, j, :].clone() # (B, T)
            perm_indices = torch.randperm(T, device=x.device)
            x_j_perturbed = x_j_perturbed[:, perm_indices]
            
            z_j_new = self.uncouple(x_j_perturbed.unsqueeze(1)) # (B, 1, T, C)
            z_mixed = z_orig.clone()
            z_mixed[:, j, :, :] = z_j_new.squeeze(1)
            
            zhat_mixed = self.predict_latent_next(z_mixed)
            x_pred_mixed = self.recouple(zhat_mixed[..., :-1, :])
            
            eps_pert = (x_pred_mixed - x[..., 1:]) ** 2
            eps_pert_mean = eps_pert.mean(dim=0) # (N, T-1)
            
            delta = F.relu(eps_pert_mean - eps_orig_mean)
            strengths[j, :, :] = delta
        return strengths  # (N, N, T)

    @torch.no_grad()
    def static_causal_aggregation(self, method: str = 'l2_norm') -> torch.Tensor:
        """
        从依赖矩阵聚合静态因果图
        method: 'l2_norm', 'mean', 'max'
        """
        if method == 'l2_norm':
            # L2范数聚合，如论文公式(11)
            static_graph = torch.sqrt(torch.mean(self.Psi ** 2, dim=0))
        elif method == 'mean':
            static_graph = torch.mean(torch.abs(self.Psi), dim=0)
        elif method == 'max':
            static_graph = torch.max(torch.abs(self.Psi), dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return static_graph  # (N, N)
# ------------------------------
# Training skeleton
# ------------------------------
def train_unclenet(
    model: UnCLENet,
    data: torch.Tensor,
    recon_epochs: int = 500,
    joint_epochs: int = 1000,
    alpha: float = 1.0,
    lambda_l1: float = 1e-3,
    lr: float = 1e-3,
    device: Optional[torch.device] = None
):
    """
    data: (B, N, T) training dataset - in practice you'd use DataLoader / batches; here simple full-batch skeleton.
    Training follows paper: 1) pretrain reconstruction (LRecon), 2) joint minimize LRecon + alpha LPred + LL1.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = data.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Stage 1: reconstruction pretrain
    for ep in range(recon_epochs):
        model.train()
        opt.zero_grad()
        x_recon, _, _ = model.forward(x)
        loss_recon = F.mse_loss(x_recon, x)
        loss_recon.backward()
        opt.step()
        if (ep+1) % max(1, recon_epochs//5) == 0:
            print(f"[Recon] ep {ep+1}/{recon_epochs} loss_recon={loss_recon.item():.6f}")

    # Stage 2: joint training
    for ep in range(joint_epochs):
        model.train()
        opt.zero_grad()
        x_recon, x_pred, _ = model.forward(x)
        # align pred: predictions x_pred[..., t] were generated from z[..., t-1], paper uses predicted x_{t+1} vs true x_{t+1}
        # so compute prediction loss over t=1..T-1 comparing x_pred[..., :-1] with x[..., :-1] OR shift appropriately.
        # Our implementation: x_pred[..., t] is model's prediction for x[..., t] based on z[..., t-1]; so we compute MSE on t in [1..T-1]
        pred_loss = F.mse_loss(x_pred, x[..., 1:])
        recon_loss = F.mse_loss(x_recon, x)
        l1 = lambda_l1 * model.psi_l1()
        total = recon_loss + alpha * pred_loss + l1
        total.backward()
        opt.step()
        if (ep+1) % max(1, joint_epochs//10) == 0:
            print(f"[Joint] ep {ep+1}/{joint_epochs} total={total.item():.6f} recon={recon_loss.item():.6f} pred={pred_loss.item():.6f} l1={l1.item():.6f}")

    return model

# ------------------------------
# Example usage skeleton
# ------------------------------
if __name__ == "__main__":
    # small smoke-test using synthetic data
    B = 8
    N = 128
    T = 200
    # simulate simple sin signals with small coupling noise
    t = torch.linspace(0, 6.28, T)
    base = torch.sin(t).unsqueeze(0).unsqueeze(0).repeat(B, N, 1)  # (B,N,T)
    noise = 0.05 * torch.randn(B, N, T)
    x = base + noise

    model = UnCLENet(N=N, latent_C=8, tcn_levels_unc=3, tcn_hidden_unc=32, kernel_size=3, dropout=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_unclenet(model, x, recon_epochs=20, joint_epochs=50, alpha=1.0, lambda_l1=1e-4, lr=1e-3, device=device)
    # compute dynamic strengths
    x = x.to(device)
    dynamic_strengths = model.dynamic_causal_strengths(x, perturb_mode='permute', perm_seed=0)  # (N,N,T)
    print("dynamic strengths shape:", dynamic_strengths.shape)
    static_strengths = model.static_causal_aggregation(method='l2_norm')  # (N,N,T)
    print("static strengths shape:", static_strengths.shape)
