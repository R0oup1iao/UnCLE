import os
import argparse
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def generate_lorenz96_ring(p, T, F=10.0, seed=0, center=(0,0), radius=5.0):
    """生成单个 Lorenz96 环的数据和坐标"""
    if seed is not None:
        np.random.seed(seed)

    # 1. ODE
    def lorenz96_deriv(x, t):
        x_plus_1 = np.roll(x, -1)
        x_minus_1 = np.roll(x, 1)
        x_minus_2 = np.roll(x, 2)
        dxdt = (x_plus_1 - x_minus_2) * x_minus_1 - x + F
        return dxdt

    # 2. Integrate
    x0 = np.random.normal(scale=0.01, size=p) + F 
    # 丢弃前 1000 步作为 Burn-in
    burn_in = 1000
    t = np.linspace(0, (T + burn_in) * 0.1, T + burn_in)
    X_full = odeint(lorenz96_deriv, x0, t)
    X = X_full[burn_in:, :]
    
    # 3. GT Matrix (只在环内连接)
    gt = np.zeros((p, p), dtype=int)
    for i in range(p):
        gt[(i - 1) % p, i] = 1
        gt[(i - 2) % p, i] = 1
        gt[(i + 1) % p, i] = 1
        gt[i, i] = 1
        
    # 4. Coords (Ring around center)
    angles = np.linspace(0, 2 * np.pi, p, endpoint=False)
    # 加上 center 偏移
    coords = np.stack([radius * np.cos(angles) + center[0], 
                       radius * np.sin(angles) + center[1]], axis=1)
    coords += np.random.normal(0, 0.1, coords.shape) # Jitter

    return X, gt, coords

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='data/synthetic')
    parser.add_argument('--num_groups', type=int, default=4, help='Number of independent groups')
    parser.add_argument('--nodes_per_group', type=int, default=32)
    parser.add_argument('--T', type=int, default=2000)
    parser.add_argument('--num_replicas', type=int, default=1)
    args = parser.parse_args()

    dataset_name = "cluster_lorenz"
    base_path = os.path.join(args.output_path, dataset_name)
    os.makedirs(base_path, exist_ok=True)

    total_N = args.num_groups * args.nodes_per_group
    
    for r in range(args.num_replicas):
        all_data = []
        all_gt = np.zeros((total_N, total_N))
        all_coords = []
        
        # 空间布局：把 4 个环放在 2x2 的格子上
        centers = [(0,0), (20,0), (0,20), (20,20), (40,0), (40,20)] # 预留多一点
        
        for g in range(args.num_groups):
            seed = 42 + r * 100 + g
            center = centers[g]
            
            X, gt, coords = generate_lorenz96_ring(
                p=args.nodes_per_group, 
                T=args.T, 
                seed=seed, 
                center=center
            )
            
            all_data.append(X)
            all_coords.append(coords)
            
            # 填充大 GT 矩阵的对角块
            start_idx = g * args.nodes_per_group
            end_idx = (g + 1) * args.nodes_per_group
            all_gt[start_idx:end_idx, start_idx:end_idx] = gt
            
        # 合并
        final_data = np.concatenate(all_data, axis=1) # (T, Total_N)
        final_coords = np.concatenate(all_coords, axis=0) # (Total_N, 2)
        
        # 保存
        np.save(os.path.join(base_path, f'data_{r}.npy'), final_data)
        np.save(os.path.join(base_path, f'gt_{r}.npy'), all_gt)
        np.save(os.path.join(base_path, f'coords_{r}.npy'), final_coords)
        
        logging.info(f"Saved Replica {r}: Shape {final_data.shape}, Groups={args.num_groups}")
        
    # 画个示意图看看
    plt.figure(figsize=(8, 8))
    plt.scatter(final_coords[:, 0], final_coords[:, 1], c=np.arange(total_N) // args.nodes_per_group, cmap='tab10')
    plt.title(f"Spatial Layout: {args.num_groups} Independent Clusters")
    plt.savefig(os.path.join(base_path, "layout_preview.png"))
    logging.info(f"Layout preview saved to {base_path}/layout_preview.png")

if __name__ == "__main__":
    main()