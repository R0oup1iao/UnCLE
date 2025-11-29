import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.cluster import KMeans

# --- 1. 基础组件 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 预计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class CausalTransformerBlock(nn.Module):
    """
    标准的 Transformer Encoder Block，带 Causal Mask 缓存优化
    """
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, 
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, output_dim)

    def _generate_square_subsequent_mask(self, sz):
        # 生成上三角 Mask
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        B, T, C = x.shape
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # 缓存 Mask 避免重复计算
        if not hasattr(self, 'causal_mask') or self.causal_mask.size(0) != T:
            mask = self._generate_square_subsequent_mask(T).to(x.device)
            self.register_buffer('causal_mask', mask, persistent=False)
            
        out = self.transformer(x, mask=self.causal_mask, is_causal=True)
        return self.output_proj(out)

# --- 2. 因果图层 (Basis Decomposition) ---
class CausalGraphLayer(nn.Module):
    """
    学习节点间因果关系的图层
    优化: 使用基分解 (Basis Decomposition) 大幅降低显存占用
    """
    def __init__(self, N, C, num_bases=4):
        super().__init__()
        self.N = N
        self.C = C
        self.num_bases = min(num_bases, C)

        # 结构矩阵: 学习通用的连接结构 (N x N)
        self.adjacency = nn.Parameter(torch.eye(N) + torch.randn(N, N) * 0.01)
        
        # 基权重: K 种基础交互模式 (K x N x N)
        self.basis_weights = nn.Parameter(torch.randn(self.num_bases, N, N) * 0.02)
        
        # 组合系数: 每个 Channel 如何组合基模式 (C x K)
        self.channel_coeffs = nn.Parameter(torch.randn(C, self.num_bases)) 
        
        self.register_buffer('static_mask', torch.ones(N, N))

    def forward(self, z, dynamic_mask=None):
        """
        z: (B, N, T, C)
        dynamic_mask: (N, N) 用于局部窗口限制
        """
        mask = self.static_mask
        if dynamic_mask is not None:
            mask = mask * dynamic_mask

        # 重构权重: W_eff = Coeffs @ Bases -> (C, N, N)
        eff_weights = torch.einsum('ck,knm->cnm', self.channel_coeffs, self.basis_weights)
        
        # 最终邻接矩阵: A_final = (C, N, N)
        A_final = eff_weights * self.adjacency.unsqueeze(0) * mask.unsqueeze(0)
        
        # 图传播: z_out = z_in * A
        # 利用广播机制: (..., 1, N) @ (..., N, N)
        z_in = z.permute(0, 2, 3, 1).unsqueeze(-2) 
        z_out = torch.matmul(z_in, A_final) 
        z_out = z_out.squeeze(-2)
        
        return torch.tanh(z_out).permute(0, 3, 1, 2).contiguous() 

    def structural_l1_loss(self):
        return torch.sum(torch.abs(self.adjacency))
    
    def get_soft_graph(self):
        with torch.no_grad():
            W = torch.einsum('ck,knm->cnm', self.channel_coeffs, self.basis_weights)
            W_mean = torch.mean(torch.abs(W), dim=0)
            return torch.abs(self.adjacency) * W_mean * self.static_mask

class CausalFormer(nn.Module):
    """
    结合 Transformer 和 Graph 的基本单元
    """
    def __init__(self, N, latent_C=8, d_model=64, nhead=4, num_layers=2, num_bases=4):
        super().__init__()
        self.N = N
        self.C = latent_C
        self.unc_trans = CausalTransformerBlock(input_dim=1, output_dim=latent_C, 
                                                d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.rec_trans = CausalTransformerBlock(input_dim=latent_C, output_dim=1, 
                                                d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.graph = CausalGraphLayer(N, latent_C, num_bases=num_bases)

    def forward(self, x, mask=None):
        B, N, T = x.shape
        # Encoder: x -> z
        z = self.unc_trans(x.reshape(B*N, T, 1)).view(B, N, T, self.C)
        # Recon: z -> x_recon
        x_recon = self.rec_trans(z.view(B*N, T, self.C)).view(B, N, T)
        
        # Dynamics: z_t -> z_{t+1} (Mask 限制局部 Window)
        zhat_next = self.graph(z, dynamic_mask=mask)
        
        # Predict: z_{t+1} -> x_{t+1}
        x_pred = self.rec_trans(zhat_next[..., :-1, :].contiguous().view(B*N, T-1, self.C)).view(B, N, T-1)
        return x_recon, x_pred, z

# --- 3. 几何池化器 (Geometric Pooler) ---
class GeometricPooler(nn.Module):
    """
    基于物理坐标的确定性池化 (Swin Patch Partition)
    """
    def __init__(self, num_patches):
        super().__init__()
        self.num_patches = num_patches
        self.register_buffer('S_matrix', None) # 固定的分配矩阵
        self.initialized = False

    def init_structure(self, coords):
        if self.initialized: return
        
        # 坐标标准化
        coords_np = coords.detach().float().cpu().numpy()
        c_mean = coords_np.mean(axis=0)
        c_std = coords_np.std(axis=0) + 1e-5
        coords_norm = (coords_np - c_mean) / c_std
        
        # K-Means 切分物理空间
        kmeans = KMeans(n_clusters=self.num_patches, n_init=20, random_state=42)
        labels = kmeans.fit_predict(coords_norm)
        
        # 生成 Hard Assignment S (N, K)
        N = coords.shape[0]
        S_hard = torch.zeros(N, self.num_patches, device=coords.device)
        labels_tensor = torch.tensor(labels, device=coords.device).long()
        S_hard.scatter_(1, labels_tensor.unsqueeze(1), 1.0)
        
        self.S_matrix = S_hard
        self.initialized = True
        print(f"🔲 Geometric Window Initialized: N={N} -> K={self.num_patches}")

    def forward(self, x, coords):
        if not self.initialized:
            self.init_structure(coords)
        B = x.shape[0]
        return self.S_matrix.unsqueeze(0).expand(B, -1, -1)

# --- 4. 整体模型 (Graph Swin Transformer) ---
class ST_CausalFormer(nn.Module):
    def __init__(self, N, coords, hierarchy=[32, 8], latent_C=8, d_model=64, num_bases=4):
        super().__init__()
        self.dims = [N] + hierarchy
        self.num_levels = len(self.dims)
        
        if not torch.is_tensor(coords):
            coords = torch.tensor(coords).float()
        self.register_buffer('coords', coords)
        
        self.layers = nn.ModuleList()
        self.poolers = nn.ModuleList()
        
        # 注册 Buffer 用于可视化 (Fix: 之前漏了这个)
        for i in range(self.num_levels - 1):
            self.register_buffer(f'structure_S_{i}', torch.zeros(self.dims[i], self.dims[i+1]))
        
        for i in range(self.num_levels):
            # 每一层用 Basis Decomposition 节省显存
            self.layers.append(CausalFormer(N=self.dims[i], latent_C=latent_C, d_model=d_model, num_bases=num_bases))
            
            if i < self.num_levels - 1:
                # 几何 Pooler 负责 Patch Merging
                self.poolers.append(GeometricPooler(num_patches=self.dims[i+1]))
    
    def get_structural_l1_loss(self):
        loss = 0
        for layer in self.layers:
            loss += layer.graph.structural_l1_loss()
        return loss

    def _get_local_mask(self, coords, k=16):
        """
        KNN Mask: 泛化的 Shifted Window，只允许连接几何邻居
        """
        dists = torch.cdist(coords, coords)
        curr_k = min(k, coords.shape[0])
        _, indices = torch.topk(dists, curr_k, dim=1, largest=False)
        
        mask = torch.zeros_like(dists)
        mask.scatter_(1, indices, 1.0)
        # 对称化 (A是B邻居 -> B也是A邻居) + 自环
        mask = ((mask + mask.t()) > 0).float()
        mask.fill_diagonal_(1.0)
        return mask

    def forward(self, x, tau=1.0):
        xs = [x]
        curr_coords = self.coords
        S_list = []
        
        # --- Bottom-up: 特征聚合 (Patch Merging) ---
        for i in range(self.num_levels - 1):
            S = self.poolers[i](xs[-1], curr_coords)
            S_list.append(S)
            
            # 保存 S 矩阵供 Visualization 使用 (Fix: 之前漏了更新)
            with torch.no_grad():
                getattr(self, f'structure_S_{i}').copy_(S[0])
            
            # 特征聚合: Next Level X
            S_mat = S[0]
            S_norm = S_mat / (S_mat.sum(dim=0, keepdim=True) + 1e-6)
            
            # (B, N, T) -> (B, K, T)
            # 使用 matmul 避免 expand 导致的 non-contiguous 问题
            curr_x_T = xs[-1].permute(0, 2, 1)
            next_x = torch.matmul(curr_x_T, S_norm).permute(0, 2, 1)
            xs.append(next_x)
            
            # 更新下一层坐标 (Super-Node 中心)
            curr_coords = torch.mm(S_norm.t(), curr_coords)
            
        # --- Causal Discovery ---
        results = []
        for i in range(self.num_levels):
            mask = None
            if i == 0: 
                # Level 0 (Fine): 使用 KNN Mask 限制搜索空间 (Swin Local Attention)
                mask = self._get_local_mask(self.coords, k=16)
            
            # Level > 0 (Coarse): 节点少，全连接 (Global Attention)
            x_rec, x_pred, z = self.layers[i](xs[i], mask=mask)
            
            results.append({
                'level': i,
                'x_rec': x_rec,
                'x_pred': x_pred,
                'x_target': xs[i],
                'S': S_list[i] if i < len(S_list) else None
            })
            
        return results