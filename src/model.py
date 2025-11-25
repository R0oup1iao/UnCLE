import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ==========================================
# Basic Components (Transformer, PE)
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
# DiffPool Components
# ==========================================
class LearnableSpatialPooler(nn.Module):
    def __init__(self, input_seq_len, k_patches, d_model=64):
        super().__init__()
        # 输入特征: 时间序列长度 (T) + 空间坐标 (2)
        self.input_dim = input_seq_len + 2
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, k_patches)
        )
        
    def forward(self, x, coords, temperature=1.0):
        """
        x: (B, N, T) 原始时间序列
        coords: (N, 2) 空间坐标
        """
        B, N, T = x.shape
        
        # 1. 准备输入特征
        # 扩展 coords 到 Batch: (B, N, 2)
        coords_batch = coords.unsqueeze(0).repeat(B, 1, 1).to(x.device)
        
        # 拼接: (B, N, T+2)
        feat = torch.cat([x, coords_batch], dim=-1) 
        
        # 2. 预测 Logits
        logits = self.net(feat) # (B, N, K)
        
        # 3. Gumbel Softmax
        if self.training:
            S = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
        else:
            S = F.softmax(logits / temperature, dim=-1)
            
        return S, logits

class ST_CausalFormer(nn.Module):
    def __init__(self, N, coords, hierarchy=[32, 8], latent_C=8, d_model=64, seq_len=100):
        """
        hierarchy: list, e.g. [32, 8] 表示 Level 1 有 32 个 Patch，Level 2 有 8 个 Patch
        """
        super().__init__()
        self.N = N
        self.hierarchy = hierarchy
        self.levels = len(hierarchy)
        
        # 注册静态坐标
        self.register_buffer('coords', torch.tensor(coords).float())
        
        # --- 1. 构建多层 Poolers ---
        # 每一层 Pooler 负责将 Current Dim -> Next Dim
        self.poolers = nn.ModuleList()
        
        # --- 2. 构建多层 Causal Models ---
        # 每一层 Coarse Model 负责在 Next Dim 上进行预测
        self.coarse_models = nn.ModuleList()
        
        current_dim = N # 初始为原始节点数
        
        for h_dim in hierarchy:
            # Pooler: N_prev -> N_curr
            self.poolers.append(LearnableSpatialPooler(input_seq_len=seq_len, k_patches=h_dim, d_model=d_model))
            
            # Causal Model: N_curr
            self.coarse_models.append(CausalFormer(N=h_dim, latent_C=latent_C, d_model=d_model))
            
            current_dim = h_dim
            
        # --- 3. Fine Model (最底层的模型) ---
        self.fine_model = CausalFormer(N=N, latent_C=latent_C, d_model=d_model)
        self.register_buffer('psi_mask', torch.ones(latent_C, N, N))
        
        # 缓存每一层的 S 矩阵 (List of Tensors)
        self.S_list = [None] * self.levels 

    @property
    def patch_labels(self):
        """
        用于可视化：返回第一层聚类结果 (Fine -> Level 1)
        """
        if self.S_list[0] is None:
            return np.zeros(self.N)
        # (B, N, K) -> Mean over Batch -> Argmax
        return self.S_list[0].mean(0).argmax(dim=-1).detach().cpu().numpy()

    def forward(self, x, mode='fine', tau=1.0):
        # x: (B, N, T)
        
        if mode == 'hierarchy':
            # 存储每一层的输出，用于计算 Loss
            results = [] 
            
            curr_x = x
            curr_coords = self.coords
            
            for i in range(self.levels):
                # 1. Pooler: Learn S (B, N_prev, N_curr)
                S, _ = self.poolers[i](curr_x, curr_coords, temperature=tau)
                self.S_list[i] = S.detach() # 缓存用于 Mask 生成
                
                # 2. Aggregation (Coarsening)
                # X_curr = S^T @ X_prev
                S_trans = S.transpose(1, 2)
                # 归一化 S 转置，避免数值膨胀
                S_norm = S_trans / (S_trans.sum(dim=-1, keepdim=True) + 1e-6)
                
                next_x = torch.bmm(S_norm, curr_x)
                
                # 3. Coords Aggregation (用于下一层聚类)
                # Coords_curr = S_norm @ Coords_prev
                coords_batch = curr_coords.unsqueeze(0).repeat(x.shape[0], 1, 1) # (B, N, 2)
                next_coords_batch = torch.bmm(S_norm, coords_batch)
                curr_coords = next_coords_batch.mean(dim=0) # 简化：取平均
                
                # 4. Run Causal Model at this level
                model_out = self.coarse_models[i](next_x)
                
                results.append({
                    'out': model_out,    # (rec, pred, z)
                    'target': next_x,    # 用于计算 Recon Loss
                    'S': S               # 用于计算 Entropy Loss
                })
                
                # 传递给下一层
                curr_x = next_x
                
            return results
            
        elif mode == 'fine':
            # Fine 阶段只需应用 Mask
            self.fine_model.Psi.data.mul_(self.psi_mask.to(x.device))
            return self.fine_model(x)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def update_mask(self, threshold=None):
        """
        级联 Mask 生成: Top -> ... -> Mid -> Fine
        """
        # 1. 获取最顶层的图 (Top Level Graph)
        top_model = self.coarse_models[-1]
        top_graph = top_model.static_causal_aggregation().detach().cpu()
        
        if threshold is None:
            mask_diag = ~torch.eye(top_graph.shape[0], dtype=torch.bool)
            off_diag_values = top_graph[mask_diag]
            threshold = off_diag_values.mean() + 0.5 * off_diag_values.std()
            
        # 初始 Mask (Top Level)
        curr_mask = (top_graph > threshold).float()
        
        # 2. 逐层向下投影 (Back-projection)
        # S_list: [L0->L1, L1->L2, ...]
        # 反向遍历: L(n-1)->L(n) ... L0->L1
        for i in range(self.levels - 1, -1, -1):
            S = self.S_list[i]
            if S is None:
                # Fallback
                curr_dim = self.hierarchy[i]
                prev_dim = self.N if i == 0 else self.hierarchy[i-1]
                curr_mask = torch.ones(prev_dim, prev_dim)
                continue
            
            # S: (B, N_child, N_parent)
            S_avg = S.mean(dim=0).cpu()
            parents = S_avg.argmax(dim=1).numpy() # 每个子节点所属的父节点 ID
            
            n_child = S_avg.shape[0]
            next_mask = torch.zeros(n_child, n_child)
            
            # 投影逻辑: 如果 Parent A -> Parent B，则所有属于 A 的 child -> 所有属于 B 的 child
            for c_tgt in range(n_child):
                for c_src in range(n_child):
                    p_tgt = parents[c_tgt]
                    p_src = parents[c_src]
                    
                    # 连接条件: 父节点之间有连接 OR 同一个父节点内
                    if curr_mask[p_tgt, p_src] > 0 or p_tgt == p_src:
                        next_mask[c_tgt, c_src] = 1.0
            
            curr_mask = next_mask
            
        # 循环结束时，curr_mask 已经是 Fine Level Mask
        target_device = self.fine_model.Psi.device
        self.psi_mask = curr_mask.unsqueeze(0).repeat(self.fine_model.C, 1, 1).to(target_device)
        
        # 返回顶层图用于展示
        return top_graph, threshold, self.psi_mask.mean().item()