# tests/test_model_components.py
import torch
import torch.nn as nn
import pytest
import sys
import os

# 添加src路径到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import UnCLe
from tcn import TemporalConvNet, TemporalBlock, Chomp1d


class TestChomp1d:
    """测试Chomp1d模块"""
    
    def test_chomp1d_forward(self):
        """测试Chomp1d的前向传播"""
        chomp_size = 2
        chomp = Chomp1d(chomp_size)
        
        # 创建测试输入: (batch, channels, length)
        x = torch.randn(4, 8, 10)  # batch=4, channels=8, length=10
        
        output = chomp(x)
        
        # 输出长度应该减少chomp_size
        assert output.shape == (4, 8, 8), f"Expected shape (4, 8, 8), got {output.shape}"


class TestTemporalBlock:
    """测试TemporalBlock模块"""
    
    def test_temporal_block_same_channels(self):
        """测试输入输出通道数相同的TemporalBlock"""
        n_inputs = 8
        n_outputs = 8
        kernel_size = 3
        dilation = 2
        padding = (kernel_size - 1) * dilation
        
        block = TemporalBlock(n_inputs, n_outputs, kernel_size, stride=1, 
                             dilation=dilation, padding=padding, dropout=0.1)
        
        # 测试输入
        x = torch.randn(4, n_inputs, 20)  # batch=4, channels=8, length=20
        
        output = block(x)
        
        # 输出形状应该与输入相同
        assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
        
    def test_temporal_block_different_channels(self):
        """测试输入输出通道数不同的TemporalBlock"""
        n_inputs = 8
        n_outputs = 16
        kernel_size = 3
        dilation = 2
        padding = (kernel_size - 1) * dilation
        
        block = TemporalBlock(n_inputs, n_outputs, kernel_size, stride=1, 
                             dilation=dilation, padding=padding, dropout=0.1)
        
        # 测试输入
        x = torch.randn(4, n_inputs, 20)  # batch=4, channels=8, length=20
        
        output = block(x)
        
        # 输出通道数应该改变，但batch和length不变
        expected_shape = (4, n_outputs, 20)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"


class TestTemporalConvNet:
    """测试TemporalConvNet模块"""
    
    def test_tcn_forward_shape(self):
        """测试TCN的前向传播形状"""
        num_inputs = 1
        num_channels = [16, 32, 16]  # 多层TCN
        kernel_size = 3
        
        tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout=0.1)
        
        # 测试输入: (batch, channels, length)
        x = torch.randn(8, num_inputs, 50)  # batch=8, channels=1, length=50
        
        output = tcn(x)
        
        # 输出通道数应该等于最后一层的通道数，长度应该保持不变
        expected_shape = (8, num_channels[-1], 50)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        
    def test_tcn_multichannel_input(self):
        """测试多通道输入的TCN"""
        num_inputs = 8
        num_channels = [16, 32, 16]
        kernel_size = 3
        
        tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout=0.1)
        
        # 测试输入: (batch, channels, length)
        x = torch.randn(4, num_inputs, 30)  # batch=4, channels=8, length=30
        
        output = tcn(x)
        
        expected_shape = (4, num_channels[-1], 30)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"


class TestUnCLe:
    """测试UnCLe主模型"""
    
    def test_uncle_initialization(self):
        """测试UnCLe模型初始化"""
        num_variables = 5
        num_channels = 3
        tcn_channels = [16, 32]
        kernel_size = 3
        dropout = 0.1
        
        model = UnCLe(num_variables, num_channels, tcn_channels, kernel_size, dropout)
        
        # 检查模型组件是否正确初始化
        assert hasattr(model, 'uncoupler'), "Uncoupler should be initialized"
        assert hasattr(model, 'recoupler'), "Recoupler should be initialized"
        assert hasattr(model, 'dependency_matrices'), "Dependency matrices should be initialized"
        
        # 检查依赖矩阵的形状
        expected_dep_shape = (num_channels, num_variables, num_variables)
        assert model.dependency_matrices.shape == expected_dep_shape, \
            f"Expected dependency matrices shape {expected_dep_shape}, got {model.dependency_matrices.shape}"
    
    def test_uncle_forward_shape(self):
        """测试UnCLe前向传播的形状"""
        num_variables = 5
        num_channels = 3
        tcn_channels = [16, 32]
        kernel_size = 3
        dropout = 0.1
        
        model = UnCLe(num_variables, num_channels, tcn_channels, kernel_size, dropout)
        
        # 创建测试输入: (N, 1, T) 其中N=num_variables
        batch_size = num_variables  # 每个变量作为一个batch item
        seq_length = 100
        x = torch.randn(batch_size, 1, seq_length)
        
        x_recon, x_pred, dependency_matrices = model(x)
        
        # 检查重建输出的形状
        expected_recon_shape = (batch_size, 1, seq_length)
        assert x_recon.shape == expected_recon_shape, \
            f"Expected reconstruction shape {expected_recon_shape}, got {x_recon.shape}"
        
        # 检查预测输出的形状 (应该比输入少一个时间步)
        expected_pred_shape = (batch_size, 1, seq_length - 1)
        assert x_pred.shape == expected_pred_shape, \
            f"Expected prediction shape {expected_pred_shape}, got {x_pred.shape}"
        
        # 检查依赖矩阵的形状
        expected_dep_shape = (num_channels, num_variables, num_variables)
        assert dependency_matrices.shape == expected_dep_shape, \
            f"Expected dependency matrices shape {expected_dep_shape}, got {dependency_matrices.shape}"
    
    def test_uncle_different_sequence_lengths(self):
        """测试UnCLe对不同序列长度的处理"""
        num_variables = 4
        num_channels = 2
        tcn_channels = [8, 16]
        kernel_size = 3
        dropout = 0.1
        
        model = UnCLe(num_variables, num_channels, tcn_channels, kernel_size, dropout)
        
        # 测试不同序列长度
        sequence_lengths = [50, 100, 200]
        
        for seq_len in sequence_lengths:
            x = torch.randn(num_variables, 1, seq_len)
            
            x_recon, x_pred, _ = model(x)
            
            # 检查输出形状
            assert x_recon.shape == (num_variables, 1, seq_len)
            assert x_pred.shape == (num_variables, 1, seq_len - 1)




if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v"])