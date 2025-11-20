import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import copy

# ==========================================
# 1. 基础组件 (TCN & Standard UnCLe)
# ==========================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class SimpleTCN(nn.Module):
    def __init__(self, in_channels, num_levels, hidden_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden_channels
            layers += [TemporalBlock(in_ch, hidden_channels, kernel_size, stride=1, dilation=dilation,
                                     padding=(kernel_size-1) * dilation, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(hidden_channels, hidden_channels, 1)

    def forward(self, x):
        if x.dim() != 3: raise ValueError("TCN expects (B, T, C)")
        x = x.transpose(1, 2) # (B, C, T)
        y = self.network(x)
        y = self.final_conv(y)
        return y.transpose(1, 2)

class UnCLENet(nn.Module):
    def __init__(self, N, latent_C=8, tcn_levels=3, tcn_hidden=32, kernel_size=3):
        super().__init__()
        self.N = N
        self.C = latent_C
        self.unc_tcn = SimpleTCN(1, tcn_levels, tcn_hidden, kernel_size)
        self.unc_proj = nn.Linear(tcn_hidden, latent_C)
        self.rec_tcn = SimpleTCN(latent_C, tcn_levels, tcn_hidden, kernel_size)
        self.rec_proj = nn.Linear(tcn_hidden, 1)
        # 初始化 Psi 为较小值，便于稀疏化
        self.Psi = nn.Parameter(torch.randn(latent_C, N, N) * 0.02)
        self.latent_act = nn.Tanh()

    def uncouple(self, x):
        B, N, T = x.shape
        x_flat = x.view(B * N, T).unsqueeze(-1)
        z = self.unc_proj(self.unc_tcn(x_flat))
        return z.view(B, N, T, self.C)

    def recouple(self, z):
        B, N, T, C = z.shape
        xhat = self.rec_proj(self.rec_tcn(z.view(B*N, T, C))).squeeze(-1)
        return xhat.view(B, N, T)

    def predict_latent_next(self, z):
        # z: (B, N, T, C)
        z_perm = z.permute(0, 2, 3, 1) # (B, T, C, N)
        # einsum: (Batch, Time, Channel, Source_Node) x (Channel, Target_Node, Source_Node) -> (B, T, C, Target_Node)
        # 这里我们定义 Psi shape 为 (C, Target, Source)
        zhat = torch.einsum('btcn,cnm->btcm', z_perm, self.Psi) 
        return self.latent_act(zhat).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        z = self.uncouple(x)
        x_recon = self.recouple(z)
        zhat_next = self.predict_latent_next(z)
        x_pred = self.recouple(zhat_next[..., :-1, :])
        return x_recon, x_pred, z

    # [Fix] 补充缺失的 L1 Loss 方法
    def psi_l1(self):
        return torch.sum(torch.abs(self.Psi))

    def static_causal_aggregation(self):
        # L2 Norm over channels: (Target, Source)
        return torch.sqrt(torch.mean(self.Psi ** 2, dim=0))

# ==========================================
# 2. ST-UnCLENet (Spatio-Temporal Wrapper)
# ==========================================

class SpatialHierarchy:
    def __init__(self, coords, k_patches=4):
        """
        coords: (N, 2)
        """
        self.k_patches = k_patches
        self.N = coords.shape[0]
        
        # 简单使用 KMeans 划分 Patch
        kmeans = KMeans(n_clusters=k_patches, random_state=42, n_init=10)
        self.patch_labels = kmeans.fit_predict(coords)
        self.centroids = kmeans.cluster_centers_
        
        print(f"Spatial Hierarchy Built: {self.N} nodes -> {self.k_patches} patches")
        for k in range(k_patches):
            count = np.sum(self.patch_labels == k)
            print(f"  Patch {k}: {count} nodes")

    def get_fine_mask_from_coarse(self, patch_adj, threshold=0.05):
        """
        根据 Patch 间的连接 (K, K)，生成节点间的 Mask (N, N)
        patch_adj: (Target_Patch, Source_Patch)
        """
        mask = torch.zeros((self.N, self.N))
        # 二值化
        patch_binary = (patch_adj > threshold).float()
        
        for p_tgt in range(self.k_patches):
            for p_src in range(self.k_patches):
                # 如果 Patch 间有连接，或者 Patch 内部 (自环)，则允许连接
                if patch_binary[p_tgt, p_src] > 0 or p_tgt == p_src:
                    src_idx = np.where(self.patch_labels == p_src)[0]
                    tgt_idx = np.where(self.patch_labels == p_tgt)[0]
                    
                    # 利用广播赋值
                    # mask[tgt, src] = 1
                    for t in tgt_idx:
                        mask[t, src_idx] = 1.0
        return mask

class ST_UnCLENet(nn.Module):
    def __init__(self, N, coords, k_patches=4, latent_C=8):
        super().__init__()
        self.spatial = SpatialHierarchy(coords, k_patches)
        
        # Coarse Model: 处理 Patch 级数据 (K nodes)
        self.coarse_model = UnCLENet(N=k_patches, latent_C=latent_C)
        
        # Fine Model: 处理 Node 级数据 (N nodes)
        self.fine_model = UnCLENet(N=N, latent_C=latent_C)
        
        # 注册 Mask (不作为参数)
        self.register_buffer('psi_mask', torch.ones(latent_C, N, N))

    def aggregate_to_patches(self, x):
        """
        x: (B, N, T) -> patch_x: (B, K, T)
        Simple Mean Pooling
        """
        B, N, T = x.shape
        patch_x = torch.zeros(B, self.spatial.k_patches, T, device=x.device)
        for k in range(self.spatial.k_patches):
            idx = np.where(self.spatial.patch_labels == k)[0]
            if len(idx) > 0:
                patch_x[:, k, :] = x[:, idx, :].mean(dim=1)
        return patch_x

    def update_mask(self, threshold=0.1):
        # 1. 获取 Coarse 静态图 (K, K)
        patch_graph = self.coarse_model.static_causal_aggregation().cpu()
        
        # --- [优化] 自适应阈值策略 ---
        if threshold is None:
            # 策略：保留 Patch 自环 + 显著强于背景噪声的连接
            # 1. 提取非对角线元素作为背景噪声参考
            mask_diag = ~torch.eye(self.spatial.k_patches, dtype=torch.bool)
            off_diag_values = patch_graph[mask_diag]
            
            if len(off_diag_values) > 0:
                mu = off_diag_values.mean()
                std = off_diag_values.std()
                # 阈值设为：均值 + 0.5倍标准差 (可根据稀疏度需求调整)
                threshold = mu + 0.5 * std
            else:
                threshold = 0.0 # 只有1个Patch时
                
            print(f"--- Auto Threshold: {threshold:.4f} (Mean: {mu:.4f}, Std: {std:.4f}) ---")
        
        # 2. 扩展为 Node Mask (N, N)
        node_mask = self.spatial.get_fine_mask_from_coarse(patch_graph, threshold)
        
        # 3. 广播到 C 通道
        self.psi_mask = node_mask.unsqueeze(0).repeat(self.fine_model.C, 1, 1).to(self.fine_model.Psi.device)
        
        ratio = self.psi_mask[0].mean().item()
        print(f"--- Mask Updated ---")
        print(f"Active Connections Ratio: {ratio:.2%} (Reduced from 100%)")
        return patch_graph

    def forward_fine(self, x):
        # 应用 Mask 到 Psi (Soft Constraint or Hard Constraint)
        # 在 forward 中做 hard masking 用于计算
        # 注意：为了梯度能传回去但不更新被 Mask 的部分，通常在 optimizer step 后 zero_grad，
        # 或者在这里直接乘。这里直接乘比较简单。
        self.fine_model.Psi.data.mul_(self.psi_mask)
        return self.fine_model(x)

# ==========================================
# 3. 模拟数据生成 (带空间偏置)
# ==========================================

def generate_spatial_data(N=32, T=500, k_patches=4):
    """
    生成空间聚类的数据。
    规则：
    1. 坐标聚类为 4 堆。
    2. Patch 内部连接概率高。
    3. Patch 之间连接概率低 (模拟宏观因果)。
    """
    np.random.seed(42)
    
    # 1. 生成坐标 (4个中心)
    centers = np.array([[0,0], [0,10], [10,0], [10,10]])
    coords = []
    patch_ids = []
    nodes_per_patch = N // k_patches
    
    for i, c in enumerate(centers):
        # 在中心周围生成高斯分布的点
        pts = c + np.random.randn(nodes_per_patch, 2) * 1.5
        coords.append(pts)
        patch_ids.extend([i]*nodes_per_patch)
        
    coords = np.vstack(coords)
    patch_ids = np.array(patch_ids)
    
    # 2. 生成 Ground Truth (N, N) [Cause, Effect] (Source, Target)
    # 习惯上 GT 矩阵 row=Source, col=Target
    gt = np.zeros((N, N))
    
    # 定义 Patch 间的宏观因果：0->1, 2->3 (Chain)
    patch_gt = np.zeros((k_patches, k_patches))
    patch_gt[0, 1] = 1
    patch_gt[2, 3] = 1
    # 也可以加一点自环
    np.fill_diagonal(patch_gt, 1) 

    for src in range(N):
        p_src = patch_ids[src]
        for tgt in range(N):
            p_tgt = patch_ids[tgt]
            
            prob = 0
            if p_src == p_tgt:
                prob = 0.3 # 同 Patch 内部容易有连接
            elif patch_gt[p_src, p_tgt] == 1:
                prob = 0.1 # 有宏观因果的 Patch 之间可能有连接
            else:
                prob = 0.0 # 其他 Patch 之间无连接
            
            if np.random.rand() < prob and src != tgt:
                gt[src, tgt] = 1
                
    # 3. 生成时间序列 (VAR Process)
    data = np.zeros((T, N))
    data[:5, :] = np.random.randn(5, N)
    
    # 简单的非线性 VAR
    for t in range(5, T):
        # x_t = 0.8 * x_{t-1} * A + noise
        # 加上一点非线性 sin
        prev = data[t-1]
        # 简单的线性聚合
        # signal = prev @ gt (Source->Target)
        signal = np.dot(prev, gt) 
        data[t] = 0.7 * np.tanh(signal) + 0.2 * data[t-1] + np.random.randn(N) * 0.1

    # Normalize
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    
    return torch.FloatTensor(data.T).unsqueeze(0), coords, gt, patch_gt, patch_ids

# ==========================================
# 4. 主流程
# ==========================================

if __name__ == "__main__":
    # --- Settings ---
    N = 32
    T = 400
    K = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"=== ST-UnCLe Smoke Test (N={N}, K={K}) ===")
    
    # 1. Generate Data
    x, coords, gt_fine, gt_coarse, patch_ids = generate_spatial_data(N, T, K)
    x = x.to(device)
    print(f"Data Shape: {x.shape}")
    
    # 2. Init Model
    model = ST_UnCLENet(N, coords, k_patches=K, latent_C=8).to(device)
    
    # ============================
    # Phase 1: Coarse Training
    # ============================
    print("\n>>> Phase 1: Coarse Training (Patch Level)")
    optimizer_c = torch.optim.Adam(model.coarse_model.parameters(), lr=1e-2)
    
    # 聚合数据
    x_patch = model.aggregate_to_patches(x)
    
    for ep in range(200):
        optimizer_c.zero_grad()
        # Patch Level Forward
        x_recon, x_pred, _ = model.coarse_model(x_patch)
        
        loss = F.mse_loss(x_recon, x_patch) + \
               F.mse_loss(x_pred, x_patch[..., 1:]) + \
               1e-3 * model.coarse_model.psi_l1()
               
        loss.backward()
        optimizer_c.step()
        if ep % 50 == 0: print(f"  Ep {ep}: Loss {loss.item():.4f}")
        
    # Get Mask
    estimated_patch_graph = model.update_mask(threshold=None) 
    
    # ============================
    # Phase 2: Fine Training
    # ============================
    print("\n>>> Phase 2: Fine Training (Node Level with Mask)")
    optimizer_f = torch.optim.Adam(model.fine_model.parameters(), lr=5e-3)
    
    for ep in range(300):
        optimizer_f.zero_grad()
        
        # Fine Level Forward (Internally applies mask)
        x_recon, x_pred, _ = model.forward_fine(x)
        
        loss = F.mse_loss(x_recon, x) + \
               F.mse_loss(x_pred, x[..., 1:]) + \
               1e-3 * model.fine_model.psi_l1()
               
        loss.backward()
        
        # 关键：在更新前再次把 Mask 区域的梯度置 0 (虽然 forward 乘了，但为了保险)
        model.fine_model.Psi.grad.mul_(model.psi_mask)
        
        optimizer_f.step()
        if ep % 50 == 0: print(f"  Ep {ep}: Loss {loss.item():.4f}")

    # ============================
    # Visualization
    # ============================
    print("\n>>> Visualizing Results...")
    
    # 1. 获取推断结果
    # 注意模型输出是 (Target, Source)，我们需要转置为 (Source, Target) 以匹配 GT
    est_fine = model.fine_model.static_causal_aggregation().detach().cpu().numpy().T
    est_coarse = estimated_patch_graph.detach().cpu().numpy().T # (Source, Target)
    
    # 2. Plot
    fig = plt.figure(figsize=(15, 10))
    
    # Subplot 1: Spatial Layout
    ax1 = fig.add_subplot(2, 3, 1)
    scatter = ax1.scatter(coords[:, 0], coords[:, 1], c=patch_ids, cmap='tab10', s=100)
    for i in range(N):
        ax1.text(coords[i,0], coords[i,1], str(i), fontsize=8)
    ax1.set_title("Node Spatial Layout & Patches")
    
    # Subplot 2: GT Coarse
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(gt_coarse, cmap='Blues')
    ax2.set_title("GT Coarse Graph (Patch)")
    ax2.set_xlabel("Effect Patch"); ax2.set_ylabel("Cause Patch")
    
    # Subplot 3: Est Coarse
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(est_coarse, cmap='Reds')
    ax3.set_title("Est Coarse Graph (Patch)")
    ax3.set_xlabel("Effect Patch"); ax3.set_ylabel("Cause Patch")

    # Subplot 4: GT Fine
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(gt_fine, cmap='Blues')
    ax4.set_title("GT Fine Graph (Node)")
    ax4.set_xlabel("Effect Node"); ax4.set_ylabel("Cause Node")
    
    # Subplot 5: Est Fine
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(est_fine, cmap='Reds')
    ax5.set_title("Est Fine Graph (Node)")
    ax5.set_xlabel("Effect Node"); ax5.set_ylabel("Cause Node")

    # Subplot 6: Mask (Verification)
    ax6 = fig.add_subplot(2, 3, 6)
    # Mask 是 (Target, Source) 还是 (Source, Target)? 
    # model.psi_mask 是 (C, Target, Source) -> mean -> (Target, Source)
    # 转置为 (Source, Target)
    mask_vis = model.psi_mask.mean(0).cpu().numpy().T
    ax6.imshow(mask_vis, cmap='Greens')
    ax6.set_title("Applied Spatial Mask")
    ax6.set_xlabel("Effect Node"); ax6.set_ylabel("Cause Node")
    
    plt.tight_layout()
    plt.savefig('st_uncle_demo_result.png')
    print("Results saved to 'st_uncle_demo_result.png'")
    plt.show()