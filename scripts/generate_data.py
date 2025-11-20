# script/generate_data.py
import os
import argparse
import numpy as np
from scipy.integrate import odeint
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_data(data, ground_truth, coords, base_path, dataset_name, replica_id):
    """
    Saves the generated data, ground truth, and coordinates to files.
    """
    data_dir = os.path.join(base_path, dataset_name)
    os.makedirs(data_dir, exist_ok=True)
    
    data_file = os.path.join(data_dir, f'data_{replica_id}.npy')
    gt_file = os.path.join(data_dir, f'gt_{replica_id}.npy')
    coords_file = os.path.join(data_dir, f'coords_{replica_id}.npy')
    
    np.save(data_file, data)
    np.save(gt_file, ground_truth)
    np.save(coords_file, coords)
    
    logging.info(f"Saved replica {replica_id} to {data_dir} | Data: {data.shape}, GT: {ground_truth.shape}, Coords: {coords.shape}")

def generate_lorenz96(p, T, F=10.0, seed=0, delta_t=0.1, burn_in=1000, noise_scale=0.1):
    """
    Generates Lorenz96 data + Ring Topology Coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    # --- 1. Define ODE ---
    def lorenz96_deriv(x, t):
        x_plus_1 = np.roll(x, -1)
        x_minus_1 = np.roll(x, 1)
        x_minus_2 = np.roll(x, 2)
        dxdt = (x_plus_1 - x_minus_2) * x_minus_1 - x + F
        return dxdt

    # --- 2. Integration ---
    x0 = np.random.normal(scale=0.01, size=p) + F 
    total_steps = T + burn_in
    t = np.linspace(0, total_steps * delta_t, total_steps)
    
    X_full = odeint(lorenz96_deriv, x0, t)
    X = X_full[burn_in:, :]
    
    if noise_scale > 0:
        X += np.random.normal(scale=noise_scale, size=X.shape)

    # --- 3. Construct GT ---
    gt = np.zeros((p, p), dtype=int)
    for i in range(p):
        gt[(i - 1) % p, i] = 1
        gt[(i - 2) % p, i] = 1
        gt[(i + 1) % p, i] = 1
        gt[i, i] = 1
        
    # --- 4. Construct Coordinates (Critical for ST-CausalFormer) ---
    # Lorenz96 is a cyclic system. We map nodes to a 2D Circle.
    # This ensures that node i is spatially close to i-1 and i+1.
    # KMeans will naturally slice this ring into arcs (patches).
    angles = np.linspace(0, 2 * np.pi, p, endpoint=False)
    radius = 10.0
    # Coords: (p, 2)
    coords = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    
    # Optional: Add slight jitter to prevent perfect symmetry (more realistic)
    coords += np.random.normal(0, 0.1, coords.shape)

    return X, gt, coords

def generate_tvsem(T, seed):
    """
    Generates TVSEM data + Random Coordinates.
    """
    np.random.seed(seed)
    N = 2
    data = np.zeros((T, N))
    gt = np.zeros((T, N, N))
    
    errors = np.random.normal(0, 0.1, (T, N))
    data[0] = np.random.normal(0, 1, N)

    for t in range(1, T):
        segment_index = (t - 1) // 400
        if segment_index % 2 == 0: 
            at, bt = 0.8, 0.1
            gt[t, 1, 0] = 1 
        else: 
            at, bt = 0.2, 0.7
            gt[t, 0, 1] = 1 

        data[t, 0] = at * data[t-1, 1] + errors[t, 0] 
        data[t, 1] = bt * data[t-1, 0] + errors[t, 1] 
        
    # Coordinates: Just 2 random points
    coords = np.random.rand(N, 2) * 10
        
    return data, gt, coords

def generate_nc8(T, seed):
    """
    Generates NC8 data + Random Coordinates.
    """
    np.random.seed(seed)
    N = 8
    data = np.zeros((T, N))
    gt = np.zeros((N, N))
    
    # Static GT Construction (Same as before)
    gt[0, 1] = 1; gt[1, 1] = 1
    gt[0, 2] = 1
    gt[2, 3] = 1
    gt[4, 5] = 1; gt[5, 5] = 1
    gt[4, 6] = 1; gt[0, 6] = 1
    gt[0, 7] = 1; gt[4, 7] = 1
    
    data[:16] = np.random.normal(0, 0.1, (16, N))
    errors = np.random.normal(0, 1, (T, N)) 
    
    for t in tqdm(range(16, T), desc=f"Generating NC8 replica (seed {seed})"):
        hist = data[t-1:t-17:-1].T 
        xt, yt, zt, wt, at, bt, ct, ot = hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7]
        
        data[t, 0] = 0.45 * np.sin(t / (4 * np.pi)) + 0.45 * np.sin(t / (9 * np.pi)) + 0.25 * np.sin(t / (3 * np.pi)) + 0.1 * errors[t, 0]
        data[t, 1] = 0.24*xt[0] - 0.28*xt[1] + 0.08*xt[2] + 0.2*xt[3] + 0.2*yt[0] - 0.12*yt[1] + 0.16*yt[2] + 0.04*yt[3] + 0.02*errors[t, 1]
        data[t, 2] = 3*(0.6*xt[0])**3 + 3*(0.4*xt[1])**3 + 3*(0.2*xt[2])**3 + 3*(0.5*xt[3])**3 + 0.02*errors[t, 2]
        data[t, 3] = 0.8*(0.4*zt[0])**3 + 0.8*(0.5*zt[1])**3 + 0.64*zt[2] + 0.48*zt[3] + 0.02*errors[t, 3]
        data[t, 4] = 0.15*np.sin(t/6) + 0.35*np.sin(t/80) + 0.65*np.sin(t/125) + 0.1*errors[t, 4]
        data[t, 5] = 0.54*at[12] - 0.63*at[13] + 0.18*at[14] + 0.45*at[15] + 0.36*bt[12] + 0.27*bt[13] - 0.36*bt[14] + 0.18*bt[15] + 0.02*errors[t, 5]
        data[t, 6] = max(0.24*at[12] + 0.3*at[13], -0.2) + 1.2*abs(0.2*at[14] + 0.5*xt[15]) + 0.02*errors[t, 6]
        data[t, 7] = 0.39*xt[12] - 0.65*xt[13] + 0.52*xt[14] + 0.13*xt[15] + 0.52*at[0] - 0.65*at[1] + 0.26*at[2] + 0.52*at[3] + 0.02*errors[t, 7]

    # Coordinates: 8 random points
    coords = np.random.rand(N, 2) * 10
    
    return data, gt, coords

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for UnCLe.")
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['lorenz96', 'tvsem', 'nc8'],
                        help='Name of the dataset to generate.')
    parser.add_argument('--num_replicas', type=int, default=5, help='Number of replicas to generate.')
    parser.add_argument('--output_path', type=str, default='data/synthetic', help='Base path to save the data.')
    parser.add_argument('--T', type=int, default=1000, help='Time steps.')
    parser.add_argument('--p', type=int, default=128, help='Variables for Lorenz96.')
    
    args = parser.parse_args()

    for i in range(args.num_replicas):
        seed = 42 + i * 100 
        logging.info(f"--- Generating {args.dataset}, Replica {i+1}/{args.num_replicas} (seed={seed}) ---")

        if args.dataset == 'lorenz96':
            # Lorenz: Ring topology
            data, gt, coords = generate_lorenz96(p=args.p, T=args.T, F=10, seed=seed)
        elif args.dataset == 'tvsem':
            data, gt, coords = generate_tvsem(T=args.T, seed=seed)
        elif args.dataset == 'nc8':
            data, gt, coords = generate_nc8(T=args.T, seed=seed)
        else:
            raise ValueError("Unknown dataset.")
            
        save_data(data, gt, coords, args.output_path, args.dataset, i)

if __name__ == '__main__':
    main()