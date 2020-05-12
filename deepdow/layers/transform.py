"""Collection of layers focusing on transforming tensors while keeping the number of dimensions constant."""

import torch
import torch.nn as nn


class Conv(nn.Module):
    """Convolutional layer.

    Parameters
    ----------
    n_input_channels : int
        Number of input channels.

    n_output_channels : int
        Number of output channels.

    kernel_size : int
        Size of the kernel.

    method : str, {'2D, '1D'}
        What type of convolution is used in the background.
    """

    def __init__(self, n_input_channels, n_output_channels, kernel_size=3, method='2D'):
        super().__init__()

        self.method = method

        if method == '2D':
            self.conv = nn.Conv2d(n_input_channels,
                                  n_output_channels,
                                  kernel_size=kernel_size,
                                  padding=(kernel_size - 1) // 2)
        elif method == '1D':
            self.conv = nn.Conv1d(n_input_channels,
                                  n_output_channels,
                                  kernel_size=kernel_size,
                                  padding=(kernel_size - 1) // 2)
        else:
            raise ValueError("Invalid method {}, only supports '1D' or '2D'.".format(method))

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_input_channels, lookback, n_assets) if `self.method='2D'`. Otherwise
            `(n_samples, n_input_channels, lookback)`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, n_output_channels, lookback, n_assets)` if `self.method='2D'`. Otherwise
            `(n_samples, n_output_channels, lookback)`.

        """
        return self.conv(x)


class RNN(nn.Module):
    """Recurrent neural network layer.

    Parameters
    ----------
    n_channels : int
        Number of input channels.

    hidden_size : int
        Hidden state size. Alternatively one can see it as number of output channels.

    cell_type : str, {'LSTM', 'RNN'}
        Type of the recurrent cell.

    bidirectional : bool
        If True, then bidirectional. Note that `hidden_size` already takes this parameter into account.

    n_layers : int
        Number of stacked layers.

    """

    def __init__(self, n_channels, hidden_size, cell_type='LSTM', bidirectional=True, n_layers=1):
        """Construct."""
        super().__init__()

        if hidden_size % 2 != 0 and bidirectional:
            raise ValueError('Hidden size needs to be divisible by two for bidirectional RNNs.')

        hidden_size_one_direction = int(hidden_size // (1 + int(bidirectional)))  # only will work out for

        if cell_type == 'RNN':
            self.cell = torch.nn.RNN(n_channels, hidden_size_one_direction, bidirectional=bidirectional,
                                     num_layers=n_layers)

        elif cell_type == 'LSTM':
            self.cell = torch.nn.LSTM(n_channels, hidden_size_one_direction, bidirectional=bidirectional,
                                      num_layers=n_layers)

        else:
            raise ValueError('Unsupported cell_type {}'.format(cell_type))

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.


        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, self.hidden_size, lookback, n_assets)`.

        """
        n_samples, n_channels, lookback, n_assets = x.shape
        x_swapped = x.permute(0, 2, 3, 1)  # n_samples, lookback, n_assets, n_channels
        res = []

        for i in range(n_samples):
            all_hidden_ = self.cell(x_swapped[i])[0]  # lookback, n_assets, hidden_size
            res.append(all_hidden_.permute(2, 0, 1))  # hidden_size, lookback, n_assets

        return torch.stack(res)
