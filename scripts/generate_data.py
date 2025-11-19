import os
import argparse
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_data(data, ground_truth, base_path, dataset_name, replica_id):
    """Saves the generated data and ground truth to files."""
    data_dir = os.path.join(base_path, dataset_name)
    os.makedirs(data_dir, exist_ok=True)
    
    data_file = os.path.join(data_dir, f'data_{replica_id}.npy')
    gt_file = os.path.join(data_dir, f'gt_{replica_id}.npy')
    
    np.save(data_file, data)
    np.save(gt_file, ground_truth)
    logging.info(f"Saved replica {replica_id} to {data_dir} | Data: {data.shape}, GT: {ground_truth.shape}")

def generate_lorenz96(p, T, F, seed):
    """
    Generates data from the Lorenz96 system.
    GT Convention: [Cause, Effect] i.e., A[i, j] = 1 means i -> j
    """
    np.random.seed(seed)
    
    # Ground truth: dx_i/dt depends on x_{i-1}, x_{i-2}, x_{i+1}
    # So Parents(i) = {i-1, i-2, i+1}
    # Matrix: Row=Cause, Col=Effect
    gt = np.zeros((p, p))
    for i in range(p):
        gt[(i - 1) % p, i] = 1  # i-1 -> i
        gt[(i - 2) % p, i] = 1  # i-2 -> i
        gt[(i + 1) % p, i] = 1  # i+1 -> i
        
    # ODE integration using Euler method
    dt = 0.01
    burn_in_steps = 1000
    total_steps = T * 10 + burn_in_steps # Sample every 10 steps
    
    x = np.random.normal(0, 1, p)
    
    data_full = []
    for _ in range(total_steps):
        dxdt = np.zeros(p)
        # Vectorized computation for speed
        # dx/dt = (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
        x_plus_1 = np.roll(x, -1)
        x_minus_2 = np.roll(x, 2)
        x_minus_1 = np.roll(x, 1)
        dxdt = (x_plus_1 - x_minus_2) * x_minus_1 - x + F
        
        x = x + dxdt * dt
        data_full.append(x.copy())
        
    data_full = np.array(data_full)
    data = data_full[burn_in_steps::10, :] # Subsample
    
    return data, gt

def generate_tvsem(T, seed):
    """
    Generates data from the Time-Varying Structural Equation Model (TVSEM).
    GT Convention: [Cause, Effect]
    """
    np.random.seed(seed)
    N = 2
    data = np.zeros((T, N))
    # Dynamic GT: (T, N, N)
    gt = np.zeros((T, N, N))
    
    errors = np.random.normal(0, 0.1, (T, N))
    
    # Init
    data[0] = np.random.normal(0, 1, N)

    for t in range(1, T):
        segment_index = (t - 1) // 400
        # Note: The paper describes segments 1,3,5 (even index in 0-based logic) as Y->X dominant
        
        if segment_index % 2 == 0: # Segments 1, 3, 5... (0-based index 0, 2, 4)
            # Paper: "When index is even... (at, bt) = (0.8, 0.1)... dominant Y->X"
            # Eq: Xt = at * Yt-1 ...
            at, bt = 0.8, 0.1
            gt[t, 1, 0] = 1 # Cause: Y(1) -> Effect: X(0)
        else: # Segments 2, 4...
            # Paper: "When index is odd... (at, bt) = (0.2, 0.7)... dominant X->Y"
            # Eq: Yt = bt * Xt-1 ...
            at, bt = 0.2, 0.7
            gt[t, 0, 1] = 1 # Cause: X(0) -> Effect: Y(1)

        # Eq 12: Y_t = bt * X_{t-1} + ...
        # Eq 13: X_t = at * Y_{t-1} + ...
        data[t, 0] = at * data[t-1, 1] + errors[t, 0] # X_t
        data[t, 1] = bt * data[t-1, 0] + errors[t, 1] # Y_t
        
    return data, gt

def generate_nc8(T, seed):
    """
    Generates data from the Non-linear Constant (NC8) dataset.
    Strictly follows Appendix G equations.
    Variables mapping: 0:x, 1:y, 2:z, 3:w, 4:a, 5:b, 6:c, 7:o (labeled y in paper typo)
    GT Convention: [Cause, Effect]
    """
    np.random.seed(seed)
    N = 8
    data = np.zeros((T, N))
    
    # Ground Truth Construction (Static)
    # Row = Cause, Col = Effect
    gt = np.zeros((N, N))
    
    # Based on Appendix G Equations:
    # 1. x_t: pure sine (Root node). No parents.
    # 2. y_t: depends on x_{t-k}, y_{t-k}. Parents: X, Y.
    gt[0, 1] = 1 # X -> Y
    gt[1, 1] = 1 # Y -> Y (Self-loop, optional depending on eval metric, usually kept in generation)
    
    # 3. z_t: depends on x_{t-k}. Parent: X.
    gt[0, 2] = 1 # X -> Z
    
    # 4. w_t: depends on z_{t-k}. Parent: Z.
    gt[2, 3] = 1 # Z -> W
    
    # 5. a_t: pure sine (Root node). No parents.
    
    # 6. b_t: depends on a_{t-k}, b_{t-k}. Parents: A, B.
    gt[4, 5] = 1 # A -> B
    gt[5, 5] = 1 # B -> B
    
    # 7. c_t: depends on a_{t-k}, x_{t-k}. Parents: A, X.
    gt[4, 6] = 1 # A -> C
    gt[0, 6] = 1 # X -> C
    
    # 8. o_t (typo y_t in paper): depends on x_{t-k}, a_{t-k}. Parents: X, A.
    gt[0, 7] = 1 # X -> O
    gt[4, 7] = 1 # A -> O
    
    # Initialization
    data[:16] = np.random.normal(0, 0.1, (16, N))
    errors = np.random.normal(0, 1, (T, N)) # Paper says N(0,1)
    
    for t in tqdm(range(16, T), desc=f"Generating NC8 replica (seed {seed})"):
        # Unpack previous values for clarity. 
        # xt represents x_{t-1} to x_{t-16}. xt[0] is t-1.
        # Shape of slice: (16, 8). Transpose to (8, 16) so [var_idx, lag_idx]
        hist = data[t-1:t-17:-1].T 
        xt, yt, zt, wt, at, bt, ct, ot = hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7]
        
        # Equations from Appendix G
        
        # Eq 1: x_t (Pure Exogenous - Root Node)
        # Note: Do NOT add dependencies to y here. The paper defines x as driven by time.
        data[t, 0] = 0.45 * np.sin(t / (4 * np.pi)) + 0.45 * np.sin(t / (9 * np.pi)) + 0.25 * np.sin(t / (3 * np.pi)) + 0.1 * errors[t, 0]
        
        # Eq 2: y_t (Depends on X and Y)
        data[t, 1] = 0.24*xt[0] - 0.28*xt[1] + 0.08*xt[2] + 0.2*xt[3] + \
                     0.2*yt[0] - 0.12*yt[1] + 0.16*yt[2] + 0.04*yt[3] + 0.02*errors[t, 1]
        
        # Eq 3: z_t (Depends on X)
        data[t, 2] = 3*(0.6*xt[0])**3 + 3*(0.4*xt[1])**3 + 3*(0.2*xt[2])**3 + 3*(0.5*xt[3])**3 + 0.02*errors[t, 2]
        
        # Eq 4: w_t (Depends on Z)
        data[t, 3] = 0.8*(0.4*zt[0])**3 + 0.8*(0.5*zt[1])**3 + 0.64*zt[2] + 0.48*zt[3] + 0.02*errors[t, 3]
        
        # Eq 5: a_t (Pure Exogenous - Root Node)
        data[t, 4] = 0.15*np.sin(t/6) + 0.35*np.sin(t/80) + 0.65*np.sin(t/125) + 0.1*errors[t, 4]
        
        # Eq 6: b_t (Depends on A and B)
        data[t, 5] = 0.54*at[12] - 0.63*at[13] + 0.18*at[14] + 0.45*at[15] + \
                     0.36*bt[12] + 0.27*bt[13] - 0.36*bt[14] + 0.18*bt[15] + 0.02*errors[t, 5]
        
        # Eq 7: c_t (Depends on A and X)
        # Note: Paper eq uses x_{t-16}, which is xt[15]
        data[t, 6] = max(0.24*at[12] + 0.3*at[13], -0.2) + 1.2*abs(0.2*at[14] + 0.5*xt[15]) + 0.02*errors[t, 6]
        
        # Eq 8: o_t (Paper typo labels this y_t, effectively o_t)
        # Depends on X and A
        data[t, 7] = 0.39*xt[12] - 0.65*xt[13] + 0.52*xt[14] + 0.13*xt[15] + \
                     0.52*at[0] - 0.65*at[1] + 0.26*at[2] + 0.52*at[3] + 0.02*errors[t, 7]

    return data, gt

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for UnCLe.")
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['lorenz96', 'tvsem', 'nc8'],
                        help='Name of the dataset to generate.')
    parser.add_argument('--num_replicas', type=int, default=5, help='Number of replicas to generate.')
    parser.add_argument('--output_path', type=str, default='data/synthetic', help='Base path to save the data.')
    args = parser.parse_args()

    for i in range(args.num_replicas):
        seed = 42 + i * 100 # Good practice to have a fixed base seed
        logging.info(f"--- Generating {args.dataset}, Replica {i+1}/{args.num_replicas} (seed={seed}) ---")

        if args.dataset == 'lorenz96':
            # Lorenz#1 config
            data, gt = generate_lorenz96(p=20, T=250, F=10, seed=seed)
        elif args.dataset == 'tvsem':
            data, gt = generate_tvsem(T=2000, seed=seed)
        elif args.dataset == 'nc8':
            data, gt = generate_nc8(T=2000, seed=seed)
        else:
            raise ValueError("Unknown dataset.")
            
        save_data(data, gt, args.output_path, args.dataset, i)

if __name__ == '__main__':
    main()