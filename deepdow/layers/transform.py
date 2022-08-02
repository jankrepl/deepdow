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

    def __init__(
        self, n_input_channels, n_output_channels, kernel_size=3, method="2D"
    ):
        super().__init__()

        self.method = method

        if method == "2D":
            self.conv = nn.Conv2d(
                n_input_channels,
                n_output_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            )
        elif method == "1D":
            self.conv = nn.Conv1d(
                n_input_channels,
                n_output_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            )
        else:
            raise ValueError(
                "Invalid method {}, only supports '1D' or '2D'.".format(method)
            )

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

    def __init__(
        self,
        n_channels,
        hidden_size,
        cell_type="LSTM",
        bidirectional=True,
        n_layers=1,
    ):
        """Construct."""
        super().__init__()

        if hidden_size % 2 != 0 and bidirectional:
            raise ValueError(
                "Hidden size needs to be divisible by two for bidirectional RNNs."
            )

        hidden_size_one_direction = int(
            hidden_size // (1 + int(bidirectional))
        )  # only will work out for

        if cell_type == "RNN":
            self.cell = torch.nn.RNN(
                n_channels,
                hidden_size_one_direction,
                bidirectional=bidirectional,
                num_layers=n_layers,
            )

        elif cell_type == "LSTM":
            self.cell = torch.nn.LSTM(
                n_channels,
                hidden_size_one_direction,
                bidirectional=bidirectional,
                num_layers=n_layers,
            )

        else:
            raise ValueError("Unsupported cell_type {}".format(cell_type))

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
        x_swapped = x.permute(
            0, 2, 3, 1
        )  # n_samples, lookback, n_assets, n_channels
        res = []

        for i in range(n_samples):
            all_hidden_ = self.cell(x_swapped[i])[
                0
            ]  # lookback, n_assets, hidden_size
            res.append(
                all_hidden_.permute(2, 0, 1)
            )  # hidden_size, lookback, n_assets

        return torch.stack(res)


class Warp(torch.nn.Module):
    """Custom warping layer."""

    def __init__(self, mode="bilinear", padding_mode="reflection"):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, x, tform):
        """Warp the tensor `x` with `tform` along the time dimension.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.

        tform : torch.Tensor
            Tensor of shape `(n_samples, lookback)` or `(n_samples, lookback, n_assets)`.
            Note that in the first case the same transformation is going to be used over all
            assets. To prevent folding the transformation should be increasing along the
            time dimension. It should range from -1 (beginning of the series) to 1 (end of
            the series).

        Returns
        -------
        x_warped : torch.Tensor
            Warped version of input `x` with transformation `tform`. The shape is the same
            as the input shape - `(n_samples, n_channels, lookback, n_assets)`.

        """
        n_samples, n_channels, lookback, n_assets = x.shape
        dtype, device = x.dtype, x.device

        if tform.ndim == 3:
            ty = tform
        elif tform.ndim == 2:
            ty = torch.stack(
                n_assets * [tform], dim=-1
            )  # (n_samples, lookback, n_assets)
        else:
            raise ValueError(
                "The tform tensor needs to be either 2 or 3 dimensional."
            )

        tx = torch.ones(
            n_samples, lookback, n_assets, dtype=dtype, device=device
        )
        tx *= torch.linspace(
            -1, 1, steps=n_assets, device=device, dtype=dtype
        )[None, None, :]

        grid = torch.stack([tx, ty], dim=-1)

        x_warped = nn.functional.grid_sample(
            x,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=True,
        )

        return x_warped


class Zoom(torch.nn.Module):
    """Zoom in and out.

    It can dynamically zoom into more recent timesteps and disregard older ones. Conversely,
    it can collapse more timesteps into one. Based on Spatial Transformer Network.

    Parameters
    ----------
    mode : str, {'bilinear', 'nearest'}
        What interpolation to perform.

    padding_mode : str, {'zeros', 'border', 'reflection'}
        How to fill in values that fall outisde of the grid. Relevant in the case when we
        zoom out.

    References
    ----------
    [1] Jaderberg, Max, Karen Simonyan, and Andrew Zisserman. "Spatial transformer networks."
        Advances in neural information processing systems. 2015.

    """

    def __init__(self, mode="bilinear", padding_mode="reflection"):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, x, scale):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.

        scale : torch.Tensor
            Tensor of shape `(n_samples,)` representing how much to zoom in (`scale < 1`) or
            zoom out (`scale > 1`).

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)` that is a zoomed
            version of the input. Note that the shape is identical to the input.

        """
        translate = 1 - scale

        theta = torch.stack(
            [
                torch.tensor([[1, 0, 0], [0, s, t]])
                for s, t in zip(scale, translate)
            ],
            dim=0,
        )
        theta = theta.to(device=x.device, dtype=x.dtype)

        grid = nn.functional.affine_grid(theta, x.shape, align_corners=True)
        x_zoomed = nn.functional.grid_sample(
            x,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=True,
        )

        return x_zoomed
