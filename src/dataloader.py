import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CausalTimeSeriesDataset(Dataset):
    """
    专业的时序 Dataset，支持滑动窗口切片，低内存占用。
    """
    def __init__(self, data, window_size, stride=1, mode='train', split_ratio=0.8):
        """
        args:
            data: np.ndarray, shape (T_total, N)
            window_size: int, 时间窗口长度 (T)
            stride: int, 滑动步长
            mode: 'train' or 'val'
            split_ratio: 训练集占比
        """
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
        # 公式: (Total_Len - Window_Len) // Stride + 1
        if len(self.data) < window_size:
            self.n_samples = 0
        else:
            self.n_samples = (len(self.data) - window_size) // stride + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        核心：只在需要时切片，不占用额外内存
        """
        # 计算真实的切片索引
        start = idx * self.stride
        end = start + self.window_size
        
        # 切片: (T_window, N)
        sample = self.data[start:end]
        
        # 转换: Numpy -> Tensor
        # 形状变换: (T, N) -> (N, T) 以适配你的 ST_CausalFormer 输入 (Batch, N, T)
        sample_tensor = torch.from_numpy(sample).float().t() 
        
        # 注意：这里不需要 return target，因为因果发现通常是自监督的 (reconstruction)
        # 如果你需要 target (比如 next step prediction)，可以在这里由 sample 切分出来
        return sample_tensor

def load_from_disk(base_path, dataset_name, replica_id):
    """读取磁盘文件 (Numpy格式)"""
    data_dir = os.path.join(base_path, dataset_name)
    data_path = os.path.join(data_dir, f'data_{replica_id}.npy')
    gt_path = os.path.join(data_dir, f'gt_{replica_id}.npy')
    coords_path = os.path.join(data_dir, f'coords_{replica_id}.npy')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Data not found: {data_path}")

    data_np = np.load(data_path)     # Shape: (T, N)
    gt_np = np.load(gt_path)         # Shape: (N, N)
    coords_np = np.load(coords_path) # Shape: (N, 2)
    
    return data_np, gt_np, coords_np

def get_data_context(args):
    """
    工厂函数：返回 Train/Val Loaders 和 Meta
    """
    base_path = getattr(args, 'data_path', 'data/synthetic')
    dataset_name = getattr(args, 'dataset', 'lorenz96')
    replica_id = getattr(args, 'replica_id', 0)
    
    # 1. 参数配置
    # 真实数据往往很长，我们不能把整个 T=10000 塞进模型
    # 我们切成小窗口，比如 T_window=100
    window_size = getattr(args, 'window_size', 100) 
    stride = getattr(args, 'stride', 10) # 步长，越小数据越多
    batch_size = getattr(args, 'batch_size', 32)

    print(f"📂 Loading {dataset_name} (Replica {replica_id})...")
    
    # 2. 加载原始大矩阵
    data_np, gt_np, coords_np = load_from_disk(base_path, dataset_name, replica_id)

    # 3. 标准化 (Z-Score)
    mean = data_np.mean(axis=0)
    std = data_np.std(axis=0) + 1e-5
    data_np = (data_np - mean) / std

    # 4. 实例化 Dataset (Train / Val)
    train_ds = CausalTimeSeriesDataset(data_np, window_size, stride, mode='train')
    val_ds = CausalTimeSeriesDataset(data_np, window_size, stride, mode='val')

    print(f"✅ Data Split: Train={len(train_ds)} samples, Val={len(val_ds)} samples")
    print(f"   Window Size: {window_size}, Stride: {stride}")

    # 5. 构造 Loader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 6. Meta 信息
    meta = {
        "coords": coords_np,
        "gt_fine": gt_np,
        "gt_coarse": None, 
        "patch_ids": None
    }
    
    return train_loader, val_loader, meta