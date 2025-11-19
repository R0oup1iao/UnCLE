import torch
import numpy as np
import random
import logging
import os

def set_seed(seed: int):
    """固定随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_logger(log_dir, name="UnCLe"):
    """获取配置好的 Logger"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加 handler
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File Handler
        file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Stream Handler (Console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    return logger