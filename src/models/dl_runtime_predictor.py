"""
dl_runtime_predictor.py

Deep Learning Architectures for Runtime Prediction

This module defines PyTorch-based neural network architectures for GPU job
runtime prediction. It includes 1D-CNN, LSTM, and a hybrid CNN-LSTM model,
all optimized for tabular workload data where each row is treated as a
single time step.

Key Components
--------------
RuntimePredictorCNN
    1D convolutional network for spatial feature extraction.
RuntimePredictorLSTM
    Recurrent network for sequence-aware prediction.
RuntimePredictorCNNLSTM
    Hybrid architecture combining CNN feature extraction with LSTM sequencing.

# For tabular data, jobs are now grouped into sequences using a sliding window.
# Temporal features are dynamically extracted over the sequence length.
"""

from __future__ import annotations

import torch
import torch.nn as nn


__all__ = [
    "RuntimePredictorCNN",
    "RuntimePredictorLSTM",
    "RuntimePredictorCNNLSTM",
]


class RuntimePredictorLSTM(nn.Module):
    """
    LSTM-based regressor for GPU job runtime prediction.

    Parameters
    ----------
    input_size : int
        Number of input features per time step.
    hidden_size : int
        Number of hidden units in the LSTM cell.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability for regularization layers.

    Notes
    -----
    When used with a sequence data loader (e.g., seq_len > 1), it leverages
    the temporal relationships of consecutive jobs to make better runtime predictions.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2) -> None:
        super().__init__()
        # PyTorch LSTM's internal dropout only applies between layers (if num_layers > 1)
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

        # Scientific Initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, seq_len, input_size)

        Returns
        -------
        torch.Tensor of shape (batch,)
        """
        out, _ = self.lstm(x)
        out = self.ln(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        return out.view(-1)


class RuntimePredictorCNN(nn.Module):
    """
    1D-Convolutional regressor for GPU job runtime prediction.

    For sequence data (seq_len > 1), ``Conv1d`` slides a kernel across the 
    time steps to extract local temporal patterns, which are then pooled 
    via Global Average Pooling.

    Parameters
    ----------
    input_features : int
        Number of input feature columns.
    num_filters : int
        Number of output channels (filters) in the convolutional layer.
    kernel_size : int
        Convolution kernel size (e.g., 3) for temporal sliding window.
    dropout : float
        Dropout probability.
    """
    def __init__(self, input_features: int, num_filters: int, kernel_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_features, out_channels=num_filters, kernel_size=kernel_size
        )
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters, 32)
        self.fc2 = nn.Linear(32, 1)
        
        # He/Kaiming Initialization (Standard for ReLU networks)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, seq_len, input_features)

        Returns
        -------
        torch.Tensor of shape (batch,)
        """
        # (batch, seq_len, features) → (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = torch.mean(out, dim=2)  # Global average pool → (batch, filters)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.view(-1)


class RuntimePredictorCNNLSTM(nn.Module):
    """
    Hybrid CNN + LSTM regressor for GPU job runtime prediction.

    The CNN layer acts as a local temporal feature extractor over the sequence
    of jobs, whose output is then processed by an LSTM layer to capture longer
    temporal dependencies.

    Parameters
    ----------
    input_features : int
        Number of input feature columns.
    num_filters : int
        Number of filters (output channels) in the CNN layer.
    kernel_size : int
        Convolution kernel size (e.g., 3).
    lstm_hidden : int
        Number of hidden units in the LSTM.
    lstm_layers : int
        Number of stacked LSTM layers.
    dropout : float
    """
    def __init__(
        self,
        input_features: int,
        num_filters: int,
        kernel_size: int,
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_features, out_channels=num_filters, kernel_size=kernel_size
        )
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        self.ln = nn.LayerNorm(lstm_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden, 32)
        self.fc2 = nn.Linear(32, 1)

        # Standard Initializations
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        for name, param in self.lstm.named_parameters():
            if 'weight' in name: nn.init.orthogonal_(param.data)
            elif 'bias' in name: nn.init.constant_(param.data, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, seq_len, input_features)

        Returns
        -------
        torch.Tensor of shape (batch,)
        """
        x = x.permute(0, 2, 1)                   # (batch, features, seq_len)
        cnn_out = self.relu(self.bn1(self.conv1(x)))
        cnn_out = cnn_out.permute(0, 2, 1)        # (batch, seq_len, filters)
        
        lstm_out, _ = self.lstm(cnn_out)
        last_out = self.ln(lstm_out[:, -1, :])
        out = self.dropout(self.relu(self.fc1(last_out)))
        out = self.fc2(out)
        return out.view(-1)
