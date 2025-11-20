import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.cluster import KMeans

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class CausalTransformerBlock(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=d_model*2, 
                                                   dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_dim)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        B, T, C = x.shape
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        mask = self._generate_square_subsequent_mask(T).to(x.device)
        out = self.transformer(x, mask=mask, is_causal=True)
        return self.output_proj(out)

class CausalFormer(nn.Module):
    def __init__(self, N, latent_C=8, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.N = N
        self.C = latent_C
        self.unc_trans = CausalTransformerBlock(input_dim=1, output_dim=latent_C, 
                                                d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.rec_trans = CausalTransformerBlock(input_dim=latent_C, output_dim=1, 
                                                d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.Psi = nn.Parameter(torch.randn(latent_C, N, N) * 0.02)
        self.latent_act = nn.Tanh()

    def uncouple(self, x):
        B, N, T = x.shape
        z = self.unc_trans(x.view(B*N, T, 1))
        return z.view(B, N, T, self.C)

    def recouple(self, z):
        B, N, T, C = z.shape
        xhat = self.rec_trans(z.view(B*N, T, C))
        return xhat.view(B, N, T)

    def predict_latent_next(self, z):
        z_perm = z.permute(0, 2, 3, 1) 
        zhat = torch.einsum('btcn,cnm->btcm', z_perm, self.Psi)
        return self.latent_act(zhat).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        z = self.uncouple(x)
        x_recon = self.recouple(z)
        zhat_next = self.predict_latent_next(z)
        x_pred = self.recouple(zhat_next[..., :-1, :])
        return x_recon, x_pred, z

    def psi_l1(self):
        return torch.sum(torch.abs(self.Psi))

    def static_causal_aggregation(self):
        return torch.sqrt(torch.mean(self.Psi ** 2, dim=0))

class ST_CausalFormer(nn.Module):
    def __init__(self, N, coords, k_patches=4, latent_C=8, d_model=64):
        super().__init__()
        self.k_patches = k_patches
        self.N = coords.shape[0]
        # KMeans 最好固定 random_state 以保证多卡一致性
        kmeans = KMeans(n_clusters=k_patches, random_state=42, n_init=10)
        self.patch_labels = kmeans.fit_predict(coords)
        
        self.coarse_model = CausalFormer(N=k_patches, latent_C=latent_C, d_model=d_model)
        self.fine_model = CausalFormer(N=N, latent_C=latent_C, d_model=d_model)
        self.register_buffer('psi_mask', torch.ones(latent_C, N, N))
        
    def forward(self, x, mode='fine'):
        if mode == 'coarse':
            # 1. 聚合
            x_patch = self.aggregate_to_patches(x)
            # 2. 运行 coarse model
            # 返回 output 和 target(x_patch) 方便计算 loss
            return self.coarse_model(x_patch), x_patch
            
        elif mode == 'fine':
            # Fine 阶段逻辑
            # 1. 应用 mask (注意：mask 更新最好放在 forward 里或 forward 前，这里保持原有逻辑)
            self.fine_model.Psi.data.mul_(self.psi_mask.to(x.device))
            # 2. 运行 fine model
            return self.fine_model(x)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def aggregate_to_patches(self, x):
        B, N, T = x.shape
        patch_x = torch.zeros(B, self.k_patches, T, device=x.device)
        for k in range(self.k_patches):
            idx = np.where(self.patch_labels == k)[0]
            if len(idx) > 0:
                patch_x[:, k, :] = x[:, idx, :].mean(dim=1)
        return patch_x

    def update_mask(self, threshold=None):
        patch_graph = self.coarse_model.static_causal_aggregation().cpu()
        
        if threshold is None:
            mask_diag = ~torch.eye(self.k_patches, dtype=torch.bool)
            off_diag_values = patch_graph[mask_diag]
            threshold = off_diag_values.mean() + 0.5 * off_diag_values.std()
            
        mask = torch.zeros((self.N, self.N))
        patch_binary = (patch_graph > threshold).float()
        
        for p_tgt in range(self.k_patches):
            for p_src in range(self.k_patches):
                if patch_binary[p_tgt, p_src] > 0 or p_tgt == p_src:
                    src_idx = np.where(self.patch_labels == p_src)[0]
                    tgt_idx = np.where(self.patch_labels == p_tgt)[0]
                    for t in tgt_idx:
                        mask[t, src_idx] = 1.0
                        
        target_device = self.fine_model.Psi.device 
        self.psi_mask = mask.unsqueeze(0).repeat(self.fine_model.C, 1, 1).to(target_device)
        return patch_graph, threshold, self.psi_mask.mean().item()

    def forward_fine(self, x):
        # 应用 mask
        self.fine_model.Psi.data.mul_(self.psi_mask.to(x.device))
        return self.fine_model(x)