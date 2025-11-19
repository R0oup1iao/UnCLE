import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

class CausalDataset(Dataset):
    def __init__(self, data_dir, mode='train', normalize=True):
        """
        data_dir: 包含 data_*.npy 和 gt_*.npy 的目录
        mode: 'train' 或 'test' (此处简单处理，通常会将 replica 分开)
        """
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('data_') and f.endswith('.npy')])
        self.gt_files = sorted([f for f in os.listdir(data_dir) if f.startswith('gt_') and f.endswith('.npy')])
        self.data_dir = data_dir
        self.normalize = normalize
        
        # 简单划分：假设所有文件都用于训练（因为是无监督因果发现）
        # 如果需要验证集，可以在这里切分 self.data_files
        
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        # 加载数据 (T, N)
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        gt_path = os.path.join(self.data_dir, self.gt_files[idx])
        
        x = np.load(data_path).astype(np.float32)
        gt = np.load(gt_path).astype(np.float32)
        
        # 标准化 (对时间维度标准化，即每个变量均值0方差1)
        if self.normalize:
            scaler = StandardScaler()
            x = scaler.fit_transform(x)
            
        # 转换维度 (T, N) -> (N, T) 以适配 UnCLe 输入
        x = x.T
        
        # 转 Tensor
        x_tensor = torch.from_numpy(x)
        gt_tensor = torch.from_numpy(gt)
        
        return x_tensor, gt_tensor

def get_dataloader(data_path, batch_size=4, shuffle=True):
    dataset = CausalDataset(data_path)
    # DataLoader 会自动把单个 (N, T) 堆叠成 Batch (B, N, T)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)