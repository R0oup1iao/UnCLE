import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CausalTimeSeriesDataset(Dataset):
    """
    专业的时序 Dataset，支持滑动窗口切片，低内存占用。
    """
    def __init__(self, data, window_size, stride=1, mode='train', split_ratio=0.8):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        
        # 1. 数据切分 (按时间轴切分训练/验证集)
        split_point = int(len(data) * split_ratio)
        if mode == 'train':
            self.data = data[:split_point]
        elif mode == 'val':
            self.data = data[split_point:]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 2. 计算样本总量
        if len(self.data) < window_size:
            self.n_samples = 0
        else:
            self.n_samples = (len(self.data) - window_size) // stride + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        sample = self.data[start:end]
        # (T, N) -> (N, T)
        sample_tensor = torch.from_numpy(sample).float().t() 
        return sample_tensor

def load_from_disk(base_path, dataset_name, replica_id):
    """
    读取磁盘文件 (Numpy格式)。
    [修改] 增加对真实数据的兼容性：GT 和 Coords 如果不存在，则返回默认值。
    """
    data_dir = os.path.join(base_path, dataset_name)
    data_path = os.path.join(data_dir, f'data_{replica_id}.npy')
    gt_path = os.path.join(data_dir, f'gt_{replica_id}.npy')
    coords_path = os.path.join(data_dir, f'coords_{replica_id}.npy')

    # 1. 必须要有 Data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Data not found: {data_path}")
    data_np = np.load(data_path) # Shape: (T, N)
    N = data_np.shape[1]

    # 2. GT 是可选的 (真实数据通常没有)
    if os.path.exists(gt_path):
        gt_np = np.load(gt_path) # Shape: (N, N)
    else:
        print(f"⚠️ Warning: Ground Truth not found at {gt_path}. Metrics will be skipped.")
        gt_np = None

    # 3. Coords 也是可选的 (如果没有，随机生成以适配 ST_CausalFormer)
    if os.path.exists(coords_path):
        coords_np = np.load(coords_path) # Shape: (N, 2)
    else:
        print(f"⚠️ Warning: Coords not found at {coords_path}. Using random coordinates for spatial clustering.")
        # 生成随机坐标 (N, 2)
        np.random.seed(42)
        coords_np = np.random.rand(N, 2)
    
    return data_np, gt_np, coords_np

def get_data_context(args):
    """
    工厂函数：返回 Train/Val Loaders 和 Meta
    """
    base_path = getattr(args, 'data_path', 'data/synthetic')
    dataset_name = getattr(args, 'dataset', 'lorenz96')
    replica_id = getattr(args, 'replica_id', 0)
    
    window_size = getattr(args, 'window_size', 100) 
    stride = getattr(args, 'stride', 10)
    batch_size = getattr(args, 'batch_size', 32)

    print(f"📂 Loading {dataset_name} (Replica {replica_id})...")
    
    # 加载数据 (兼容模式)
    data_np, gt_np, coords_np = load_from_disk(base_path, dataset_name, replica_id)

    # 1. 时序数据标准化 (Z-Score)
    # 这一步对 Transformer 训练稳定至关重要
    mean = data_np.mean(axis=0)
    std = data_np.std(axis=0) + 1e-5
    data_np = (data_np - mean) / std

    # 2. [关键修复] 坐标数据归一化 (Min-Max -> [-1, 1])
    # 防止坐标数值过大（如经纬度或米制坐标）主导 LearnableSpatialPooler 的线性层，导致梯度消失或模式坍塌。
    c_min = coords_np.min(axis=0)
    c_max = coords_np.max(axis=0)
    denom = c_max - c_min
    denom[denom == 0] = 1.0 # 防止除以0
    
    coords_np = 2 * (coords_np - c_min) / denom - 1.0
    print(f"📏 Coords Normalized to [-1, 1]. Original Shape: {coords_np.shape}")

    train_ds = CausalTimeSeriesDataset(data_np, window_size, stride, mode='train')
    val_ds = CausalTimeSeriesDataset(data_np, window_size, stride, mode='val')

    print(f"✅ Data Split: Train={len(train_ds)} samples, Val={len(val_ds)} samples")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    meta = {
        "coords": coords_np,
        "gt_fine": gt_np,   # 可能是 None
        "gt_coarse": None, 
        "patch_ids": None
    }
    
    return train_loader, val_loader, meta