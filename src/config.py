class BaseConfig:
    """基础配置类，定义默认参数"""
    def __init__(self):
        # --- 数据相关 ---
        self.n_vars = 20          # 变量数量 (N)
        self.batch_size = 8       # Batch Size
        self.seq_len = 200        # 序列长度 (T) (仅参考，实际由数据决定)
        
        # --- 模型结构 ---
        self.latent_dim = 32      # 潜在空间维度 (C)
        self.tcn_levels = 4       # TCN 层数
        self.tcn_hidden = 64      # TCN 隐藏层通道数
        self.kernel_size = 3      # 卷积核大小
        self.dropout = 0.2        # Dropout
        
        # --- 训练超参 ---
        self.lr = 1e-3            # 学习率
        self.alpha = 1.0          # Prediction Loss 权重
        self.lambda_l1 = 1e-3     # L1 正则化权重
        self.recon_epochs = 500   # 第一阶段 Epochs
        self.joint_epochs = 1000  # 第二阶段 Epochs
        self.seed = 42            # 随机种子

    def update(self, **kwargs):
        """允许通过 kwargs 更新配置"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                print(f"Warning: New config key '{k}' added.")
                setattr(self, k, v)
        return self

    def __repr__(self):
        return str(self.__dict__)

# --- 预设配置 (Presets) ---

class Lorenz96Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.n_vars = 128
        self.recon_epochs = 50
        self.joint_epochs = 100
        self.batch_size = 256

class NC8Config(BaseConfig):
    def __init__(self):
        super().__init__()
        self.n_vars = 8
        self.latent_dim = 8
        self.tcn_levels = 3
        self.tcn_hidden = 32
        self.recon_epochs = 500
        self.joint_epochs = 1000

class TVSEMConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.n_vars = 2
        self.latent_dim = 4
        self.tcn_levels = 3
        self.tcn_hidden = 32
        self.batch_size = 32 # 变量少，Batch可以大点

# 映射字典，方便脚本调用
CONFIG_MAP = {
    'lorenz96': Lorenz96Config,
    'nc8': NC8Config,
    'tvsem': TVSEMConfig,
    'default': BaseConfig
}

def get_config(name):
    return CONFIG_MAP.get(name, BaseConfig)()