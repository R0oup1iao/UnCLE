import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.cluster import KMeans

# --- Basic Components ---
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
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, 
            dropout=dropout, batch_first=True, norm_first=True
        )
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
        if not hasattr(self, 'causal_mask') or self.causal_mask.size(0) != T:
            mask = self._generate_square_subsequent_mask(T).to(x.device)
            self.register_buffer('causal_mask', mask, persistent=False)
        out = self.transformer(x, mask=self.causal_mask, is_causal=True)
        return self.output_proj(out)

# --- Causal Graph Layer ---
class CausalGraphLayer(nn.Module):
    def __init__(self, N, C):
        super().__init__()
        self.N = N
        self.C = C
        
        # Start dense and prune via L1
        self.adjacency = nn.Parameter(torch.ones(N, N) * 0.9 + torch.randn(N, N) * 0.05)
        self.weights = nn.Parameter(torch.randn(C, N, N) * 0.02)
        self.register_buffer('static_mask', torch.ones(N, N))

    def forward(self, z, dynamic_mask=None):
        """
        Args:
            z: (B, N, T, C) input features
            dynamic_mask: (N, N) optional dynamic mask
        Returns:
            z_out: (B, N, T, C) output features
        """
        mask = self.static_mask
        if dynamic_mask is not None:
            mask = mask * dynamic_mask

        # Compute effective weights
        eff_weights = self.weights * self.adjacency.unsqueeze(0) * mask.unsqueeze(0)
        
        # Graph propagation using einsum
        z_in = z.permute(0, 2, 3, 1)  # (B, T, C, N)
        z_out = torch.einsum('btcn,cnm->btcm', z_in, eff_weights)
        z_out = torch.tanh(z_out)
        
        return z_out.permute(0, 3, 1, 2).contiguous()

    def structural_l1_loss(self):
        return torch.sum(torch.abs(self.adjacency))
    
    def get_soft_graph(self):
        with torch.no_grad():
            w_mag = torch.mean(torch.abs(self.weights), dim=0)
            a_mag = torch.abs(self.adjacency)
            return w_mag * a_mag * self.static_mask

class CausalFormer(nn.Module):
    def __init__(self, N, latent_C=8, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.N = N
        self.C = latent_C
        self.unc_trans = CausalTransformerBlock(input_dim=1, output_dim=latent_C, 
                                                d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.rec_trans = CausalTransformerBlock(input_dim=latent_C, output_dim=1, 
                                                d_model=d_model, nhead=nhead, num_layers=num_layers)
        
        self.graph = CausalGraphLayer(N, latent_C)

    def forward(self, x, mask=None):
        B, N, T = x.shape
        z = self.unc_trans(x.view(B*N, T, 1)).view(B, N, T, self.C)
        x_recon = self.rec_trans(z.view(B*N, T, self.C)).view(B, N, T)
        
        # Predict dynamics with Mask
        zhat_next = self.graph(z, dynamic_mask=mask)
        
        x_pred = self.rec_trans(zhat_next[..., :-1, :].contiguous().view(B*N, T-1, self.C)).view(B, N, T-1)
        return x_recon, x_pred, z

# --- Learnable Spatial Pooler ---
class LearnableSpatialPooler(nn.Module):
    def __init__(self, seq_len, num_patches, d_model=64):
        super().__init__()
        self.num_patches = num_patches
        self.d_model = d_model
        self.initialized = False

        # Coordinate encoder
        self.coord_enc = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Time series encoder
        self.ts_enc = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(), 
            nn.Linear(d_model, d_model)
        )

        # Mixer for combining spatial and temporal features
        self.mixer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_patches) 
        )
        
        # Zero-initialize mixer output to prioritize spatial prior initially
        nn.init.zeros_(self.mixer[-1].weight)
        nn.init.zeros_(self.mixer[-1].bias)

        # Cluster centers and spatial gate
        self.cluster_centers = nn.Parameter(torch.randn(num_patches, 2))
        self.spatial_gate = nn.Parameter(torch.tensor(0.5))

    def init_with_kmeans(self, coords):
        """Initialize cluster centers using K-Means clustering"""
        if self.initialized: 
            return self.cluster_centers.detach()

        try:
            # Normalize coordinates for clustering
            c_mean = coords.mean(dim=0, keepdim=True)
            c_std = coords.std(dim=0, keepdim=True) + 1e-5
            coords_norm = (coords - c_mean) / c_std
            
            # Run K-Means
            coords_np = coords_norm.cpu().numpy()
            kmeans = KMeans(n_clusters=self.num_patches, n_init=10)
            kmeans.fit(coords_np)
            
            # Update cluster centers
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            with torch.no_grad():
                self.cluster_centers.copy_(centers.to(self.cluster_centers.device))
            
            self.initialized = True
            print(f"Pooler initialized with K-Means (K={self.num_patches})")
            
            # Return denormalized centers for next layer
            centers_raw = centers.to(coords.device) * c_std + c_mean
            return centers_raw

        except Exception as e:
            print(f"K-Means init failed: {e}. Using random initialization.")
            idx = torch.randperm(coords.size(0))[:self.num_patches]
            return coords[idx]

    def forward(self, x, coords, temperature=1.0):
        """Forward pass for spatial pooling"""
        B, N, T = x.shape
        
        # Normalize coordinates (consistent with init_with_kmeans)
        c_mean = coords.mean(dim=0, keepdim=True)
        c_std = coords.std(dim=0, keepdim=True) + 1e-5
        coords_norm = (coords - c_mean) / c_std 

        # Compute embeddings
        c_emb = self.coord_enc(coords_norm).unsqueeze(0).expand(B, -1, -1)
        t_emb = self.ts_enc(x)
        combined = torch.cat([t_emb, c_emb], dim=-1)
        
        # Compute semantic and spatial logits
        logits_sem = self.mixer(combined)
        dists = torch.cdist(coords_norm, self.cluster_centers)
        logits_spatial = -dists.pow(2).unsqueeze(0).expand(B, -1, -1)

        # Combine logits with learnable gate
        alpha = F.softplus(self.spatial_gate)
        final_logits = logits_sem + alpha * logits_spatial
        final_logits = torch.clamp(final_logits, min=-10.0, max=10.0)

        # Compute soft assignment matrix
        if self.training:
            S = F.gumbel_softmax(final_logits, tau=temperature, hard=False, dim=-1)
        else:
            S = F.softmax(final_logits / temperature, dim=-1)
            
        return S

# --- Unified Hierarchy Model ---
class ST_CausalFormer(nn.Module):
    def __init__(self, N, coords, hierarchy=[32, 8], latent_C=8, d_model=64, seq_len=100):
        super().__init__()
        self.dims = [N] + hierarchy
        self.num_levels = len(self.dims)
        
        # Ensure coords is a tensor
        if not torch.is_tensor(coords):
            coords = torch.tensor(coords).float()
        self.register_buffer('coords', coords)
        
        # Initialize layers and poolers
        self.layers = nn.ModuleList()
        self.poolers = nn.ModuleList()
        
        for i in range(self.num_levels - 1):
            self.register_buffer(f'structure_S_{i}', torch.zeros(self.dims[i], self.dims[i+1]))

        for i in range(self.num_levels):
            self.layers.append(CausalFormer(N=self.dims[i], latent_C=latent_C, d_model=d_model))
            if i < self.num_levels - 1:
                self.poolers.append(LearnableSpatialPooler(
                    seq_len=seq_len, 
                    num_patches=self.dims[i+1], 
                    d_model=d_model
                ))
        
        self._init_hierarchy_with_kmeans()
        
        self.last_S = [] 

    def _init_hierarchy_with_kmeans(self):
        """Initialize each pooler layer using K-Means clustering"""
        current_coords = self.coords
        
        for pooler in self.poolers:
            next_coords = pooler.init_with_kmeans(current_coords)
            current_coords = next_coords

    @property
    def fine_model(self):
        return self.layers[0]

    @property
    def patch_labels(self):
        """Get patch assignment labels"""
        if hasattr(self, 'structure_S_0'):
             return getattr(self, 'structure_S_0').argmax(dim=-1).detach().cpu().numpy()
        elif len(self.last_S) > 0:
            return self.last_S[0].mean(0).argmax(dim=-1).detach().cpu().numpy()
        return np.zeros(self.dims[0])

    def get_structural_l1_loss(self):
        return self.layers[-1].graph.structural_l1_loss()

    def forward(self, x, tau=1.0):
        B = x.shape[0]
        
        # Bottom-up pass: spatial pooling
        xs = [x]
        curr_coords = self.coords
        S_list = []
        
        for i in range(self.num_levels - 1):
            S = self.poolers[i](xs[-1], curr_coords, temperature=tau)
            S_list.append(S)
            
            with torch.no_grad():
                avg_S = S.mean(dim=0).detach()
                getattr(self, f'structure_S_{i}').copy_(avg_S)

            S_trans = S.transpose(1, 2)
            S_norm = S_trans / (S_trans.sum(dim=-1, keepdim=True) + 1e-6)
            
            next_x = torch.bmm(S_norm, xs[-1])
            xs.append(next_x)
            
            coords_batch = curr_coords.unsqueeze(0).repeat(B, 1, 1)
            next_coords_batch = torch.bmm(S_norm, coords_batch)
            curr_coords = next_coords_batch.mean(dim=0)
            
        self.last_S = [s.detach() for s in S_list]
        
        # Top-down pass: mask propagation
        masks = [None] * self.num_levels
        top_adj = self.layers[-1].graph.get_soft_graph() 
        current_mask = top_adj
        
        for i in range(self.num_levels - 2, -1, -1):
            S_avg = S_list[i].mean(dim=0) 
            projected_mask = torch.mm(torch.mm(S_avg, current_mask), S_avg.t())
            masks[i] = projected_mask
            current_mask = projected_mask

        # Parallel causal discovery at all levels
        results = []
        for i in range(self.num_levels):
            mask_in = masks[i] if i < self.num_levels - 1 else None
            x_rec, x_pred, z = self.layers[i](xs[i], mask=mask_in)
            
            results.append({
                'level': i,
                'x_rec': x_rec,
                'x_pred': x_pred,
                'x_target': xs[i],
                'S': S_list[i] if i < len(S_list) else None
            })
            
        return results
    def update_hard_mask(self):
        """
        Vectorized update of hard masks.
        Replaces O(N^2) loops with efficient Matrix Multiplication.
        """
        with torch.no_grad():
            # 1. Update Top Level
            top_graph = self.layers[-1].graph.get_soft_graph()
            # Dynamic thresholding
            vals = top_graph[~torch.eye(top_graph.shape[0], dtype=bool, device=top_graph.device)]
            thresh = vals.mean() + 0.5 * vals.std()
            curr_mask = (top_graph > thresh).float()
            
            # Ensure diagonal is 1 (self-loops allowed for dynamics)
            curr_mask.fill_diagonal_(1.0)
            self.layers[-1].graph.static_mask.copy_(curr_mask)
            
            # 2. Propagate Down (Vectorized)
            for i in range(self.num_levels - 2, -1, -1):
                S_avg = getattr(self, f'structure_S_{i}') # Shape: (N_fine, N_coarse)
                
                # Get hard assignments: (N_fine,)
                parents = S_avg.argmax(dim=1)
                n_fine, n_coarse = S_avg.shape
                
                # Create One-Hot Assignment Matrix P: (N_fine, N_coarse)
                P = F.one_hot(parents, num_classes=n_coarse).float()
                
                # Project Mask: M_fine = P @ M_coarse @ P.T
                # If coarse nodes i and j are connected, all their children should be connected.
                projected_mask = P @ curr_mask @ P.t()
                
                # Binarize
                next_mask = (projected_mask > 0).float()
                
                # Enforce diagonal (self-loops)
                next_mask.fill_diagonal_(1.0)
                
                self.layers[i].graph.static_mask.copy_(next_mask)
                curr_mask = next_mask