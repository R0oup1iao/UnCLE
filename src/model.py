# src/model.py
import torch
import torch.nn as nn
from src.tcn import TemporalConvNet

class UnCLe(nn.Module):
    """
    The main UnCLe model for scalable dynamic causal discovery.
    It consists of a shared Uncoupler-Recoupler TCN pair and auto-regressive Dependency Matrices.
    """
    def __init__(self, num_variables, num_channels, tcn_channels, kernel_size, dropout):
        """
        Args:
            num_variables (int): Number of variables in the time series (N).
            num_channels (int): Number of semantic channels for disentanglement (C).
            tcn_channels (list of int): Number of filters in each TCN layer.
            kernel_size (int): Kernel size for TCN convolutions.
            dropout (float): Dropout rate.
        """
        super(UnCLe, self).__init__()
        self.num_variables = num_variables
        self.num_channels = num_channels

        # Uncoupler: Maps each time series into multi-channel semantic representations
        # Input: (Batch, 1, Time_Length) -> Output: (Batch, C, Time_Length')
        self.uncoupler = TemporalConvNet(num_inputs=1, num_channels=tcn_channels + [num_channels], 
                                         kernel_size=kernel_size, dropout=dropout)

        # Recoupler: Reconstructs/predicts time series from semantic representations
        # Input: (Batch, C, Time_Length') -> Output: (Batch, 1, Time_Length)
        self.recoupler = TemporalConvNet(num_inputs=num_channels, num_channels=tcn_channels[::-1] + [1], 
                                         kernel_size=kernel_size, dropout=dropout)

        # Dependency Matrices (Psi): Models inter-variable dependencies in the latent space
        # Shape: (C, N, N)
        self.dependency_matrices = nn.Parameter(torch.randn(num_channels, num_variables, num_variables) * 0.1)
        nn.init.xavier_uniform_(self.dependency_matrices)

    def forward(self, x):
        """
        Forward pass for the UnCLe model during training.
        
        Args:
            x (torch.Tensor): Input multivariate time series. Shape: (Batch=N, 1, T).
                               Note: We treat each variable as a batch item for parameter sharing.

        Returns:
            tuple: A tuple containing:
                - x_recon (torch.Tensor): Reconstructed time series. Shape: (N, 1, T).
                - x_pred (torch.Tensor): Predicted time series. Shape: (N, 1, T-1).
                - self.dependency_matrices (torch.Tensor): For L1 regularization.
        """
        # 1. Disentanglement (Uncoupler)
        # x: (N, 1, T) -> z: (N, C, T')
        # T' might be different from T if TCN changes sequence length, but our TCN preserves it.
        z = self.uncoupler(x)

        # 2. Reconstruction
        # z: (N, C, T) -> x_recon: (N, 1, T)
        x_recon = self.recoupler(z)

        # 3. Auto-regressive Prediction in Latent Space
        # z_t: (N, C, T-1) is used to predict z_{t+1}: (N, C, T-1)
        z_current = z[:, :, :-1]  # Shape: (N, C, T-1)

        # Reshape for matrix multiplication: (C, N, T-1)
        z_current_permuted = z_current.permute(1, 0, 2)

        # Apply dependency matrices: z_pred_permuted = Psi @ z_current_permuted
        # (C, N, N) @ (C, N, T-1) -> (C, N, T-1) using einsum for batched matrix multiplication over C
        z_pred_permuted = torch.einsum('cij,cjk->cik', self.dependency_matrices, z_current_permuted)
        
        # Permute back to (N, C, T-1)
        z_pred = z_pred_permuted.permute(1, 0, 2)
        
        # The paper mentions an activation function sigma, let's use ReLU as in TCN
        z_pred = torch.relu(z_pred)

        # 4. Map prediction back to original space (Recoupler)
        # z_pred: (N, C, T-1) -> x_pred: (N, 1, T-1)
        x_pred = self.recoupler(z_pred)

        return x_recon, x_pred, self.dependency_matrices