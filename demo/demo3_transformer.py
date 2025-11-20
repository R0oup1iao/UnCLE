import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math

# ==========================================
# 1. Transformer 组件 (V2 增强版)
# ==========================================

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

# ==========================================
# 2. ST-CausalFormer
# ==========================================

class ST_CausalFormer(nn.Module):
    def __init__(self, N, coords, k_patches=4, latent_C=8):
        super().__init__()
        self.k_patches = k_patches
        self.N = coords.shape[0]
        kmeans = KMeans(n_clusters=k_patches, random_state=42, n_init=10)
        self.patch_labels = kmeans.fit_predict(coords)
        print(f"Spatial Hierarchy: {self.N} nodes -> {self.k_patches} patches")

        self.coarse_model = CausalFormer(N=k_patches, latent_C=latent_C)
        self.fine_model = CausalFormer(N=N, latent_C=latent_C)
        self.register_buffer('psi_mask', torch.ones(latent_C, N, N))

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
            print(f"--- Auto Threshold: {threshold:.4f} ---")
            
        mask = torch.zeros((self.N, self.N))
        patch_binary = (patch_graph > threshold).float()
        
        for p_tgt in range(self.k_patches):
            for p_src in range(self.k_patches):
                if patch_binary[p_tgt, p_src] > 0 or p_tgt == p_src:
                    src_idx = np.where(self.patch_labels == p_src)[0]
                    tgt_idx = np.where(self.patch_labels == p_tgt)[0]
                    for t in tgt_idx:
                        mask[t, src_idx] = 1.0
                        
        self.psi_mask = mask.unsqueeze(0).repeat(self.fine_model.C, 1, 1).to(patch_graph.device)
        print(f"--- Mask Updated: Ratio {self.psi_mask[0].mean().item():.2%} ---")
        return patch_graph

    def forward_fine(self, x):
        self.fine_model.Psi.data.mul_(self.psi_mask.to(x.device))
        return self.fine_model(x)

# ==========================================
# 3. 数据生成
# ==========================================

def generate_spatial_data(N=32, T=400, k_patches=4):
    np.random.seed(42)
    centers = np.array([[0,0], [0,10], [10,0], [10,10]])
    coords = []
    patch_ids = []
    nodes_per = N // k_patches
    for i, c in enumerate(centers):
        coords.append(c + np.random.randn(nodes_per, 2) * 1.5)
        patch_ids.extend([i]*nodes_per)
    coords = np.vstack(coords)
    
    gt = np.zeros((N, N))
    patch_gt = np.zeros((k_patches, k_patches))
    patch_gt[0, 1] = 1; patch_gt[2, 3] = 1
    np.fill_diagonal(patch_gt, 1) 
    
    patch_ids = np.array(patch_ids)
    for src in range(N):
        for tgt in range(N):
            p_src, p_tgt = patch_ids[src], patch_ids[tgt]
            prob = 0.3 if p_src == p_tgt else (0.1 if patch_gt[p_src, p_tgt] else 0.0)
            if np.random.rand() < prob and src != tgt:
                gt[src, tgt] = 1
                
    data = np.zeros((T, N))
    data[:5] = np.random.randn(5, N)
    for t in range(5, T):
        prev = data[t-1]
        signal = np.dot(prev, gt)
        data[t] = 0.7 * np.tanh(signal) + 0.2 * data[t-1] + np.random.randn(N) * 0.1
    data = (data - data.mean(0)) / data.std(0)
    return torch.FloatTensor(data.T).unsqueeze(0), coords, gt, patch_gt, patch_ids

# ==========================================
# 4. Main (6 Subplots Visualization)
# ==========================================

if __name__ == "__main__":
    N, T, K = 32, 400, 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== ST-CausalFormer Full Demo (N={N}, K={K}) ===")
    
    # 1. Data
    x, coords, gt_fine, gt_coarse, patch_ids = generate_spatial_data(N, T, K)
    x = x.to(device)
    
    # 2. Model
    model = ST_CausalFormer(N, coords, k_patches=K, latent_C=8).to(device)
    
    # === Phase 1: Coarse ===
    print("\n>>> Phase 1: Coarse Training (Transformer)")
    opt_c = torch.optim.Adam(model.coarse_model.parameters(), lr=3e-3) 
    x_patch = model.aggregate_to_patches(x)
    
    for ep in range(400):
        opt_c.zero_grad()
        x_rec, x_pre, _ = model.coarse_model(x_patch)
        # V2参数：L1=2e-3
        loss = F.mse_loss(x_rec, x_patch) + F.mse_loss(x_pre, x_patch[..., 1:]) + 2e-3 * model.coarse_model.psi_l1()
        loss.backward()
        opt_c.step()
        if ep % 100 == 0: print(f"  Ep {ep}: Loss {loss.item():.4f}")
        
    est_patch_graph = model.update_mask(threshold=None) # Auto
    
    # === Phase 2: Fine ===
    print("\n>>> Phase 2: Fine Training (Transformer)")
    opt_f = torch.optim.Adam(model.fine_model.parameters(), lr=1e-3)
    
    for ep in range(300):
        opt_f.zero_grad()
        x_rec, x_pre, _ = model.forward_fine(x)
        loss = F.mse_loss(x_rec, x) + F.mse_loss(x_pre, x[..., 1:]) + 1e-3 * model.fine_model.psi_l1()
        loss.backward()
        model.fine_model.Psi.grad.mul_(model.psi_mask.to(device))
        opt_f.step()
        if ep % 100 == 0: print(f"  Ep {ep}: Loss {loss.item():.4f}")

    # === Visualize (FULL 6 Subplots) ===
    print("\n>>> Visualizing Results...")
    # 转置以匹配 GT: [Source, Target]
    est_fine = model.fine_model.static_causal_aggregation().detach().cpu().numpy().T
    est_coarse = est_patch_graph.detach().cpu().numpy().T
    mask_vis = model.psi_mask.mean(0).cpu().numpy().T

    fig = plt.figure(figsize=(15, 10))
    
    # 1. Spatial
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(coords[:, 0], coords[:, 1], c=patch_ids, cmap='tab10', s=100)
    for i in range(N): ax1.text(coords[i,0], coords[i,1], str(i), fontsize=8)
    ax1.set_title("Spatial Layout & Patches")

    # 2. GT Coarse
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(gt_coarse, cmap='Blues', vmin=0)
    ax2.set_title("GT Coarse")
    ax2.set_xlabel("Effect"); ax2.set_ylabel("Cause")

    # 3. Est Coarse
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(est_coarse, cmap='Reds', vmin=0)
    ax3.set_title("Est Coarse (Transformer)")
    ax3.set_xlabel("Effect"); ax3.set_ylabel("Cause")

    # 4. GT Fine
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(gt_fine, cmap='Blues', vmin=0)
    ax4.set_title("GT Fine (Node)")
    ax4.set_xlabel("Effect"); ax4.set_ylabel("Cause")

    # 5. Est Fine
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(est_fine, cmap='Reds', vmin=0)
    ax5.set_title("Est Fine (Node)")
    ax5.set_xlabel("Effect"); ax5.set_ylabel("Cause")

    # 6. Mask
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(mask_vis, cmap='Greens', vmin=0)
    ax6.set_title("Adaptive Spatial Mask")
    ax6.set_xlabel("Effect"); ax6.set_ylabel("Cause")

    plt.tight_layout()
    plt.savefig('transformer_full_demo.png')
    print("Saved result to 'transformer_full_demo.png'")
    plt.show()