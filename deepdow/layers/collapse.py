"""Collection of layers that decrease the number of dimensions."""

import torch
import torch.nn as nn


class AttentionCollapse(nn.Module):
    """Collapsing over the channels with attention.

    Parameters
    ----------
    n_channels : int
        Number of input channels.

    Attributes
    ----------
    affine : nn.Module
        Fully connected layer performing linear mapping.

    context_vector : nn.Module
        Fully connected layer encoding direction importance.
    """

    def __init__(self, n_channels):
        super().__init__()

        self.affine = nn.Linear(n_channels, n_channels)
        self.context_vector = nn.Linear(n_channels, 1, bias=False)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, n_channels, n_assets)`.

        """
        n_samples, n_channels, lookback, n_assets = x.shape

        res_list = []
        for i in range(n_samples):
            inp_single = x[i].permute(2, 1, 0)  # n_assets, lookback, n_channels
            tformed = self.affine(inp_single)  # n_assets, lookback, n_channels
            w = self.context_vector(tformed)  # n_assets, lookback, 1
            scaled_w = torch.nn.functional.softmax(w, dim=1)  # n_assets, lookback, 1
            weighted_sum = (inp_single * scaled_w).mean(dim=1)  # n_assets, n_channels
            res_list.append(weighted_sum.permute(1, 0))  # n_channels, n_assets

        return torch.stack(res_list, dim=0)


class AverageCollapse(nn.Module):
    """Global average collapsing over a specified dimension."""

    def __init__(self, collapse_dim=2):
        super().__init__()
        self.collapse_dim = collapse_dim

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            N-dimensional tensor of shape (d_0, d_1, ..., d_{N-1}).

        Returns
        -------
        torch.Tensor
            {N-1}-dimensional tensor of shape (d_0, ..., d_{collapse_dim - 1}, d_{collapse_dim + 1}, ..., d_{N-1}).
            Average over the removeed dimension.
        """
        return x.mean(dim=self.collapse_dim)


class ElementCollapse(nn.Module):
    """Single element over a specified dimension."""

    def __init__(self, collapse_dim=2, element_ix=-1):
        super().__init__()
        self.collapse_dim = collapse_dim
        self.element_ix = element_ix

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            N-dimensional tensor of shape (d_0, d_1, ..., d_{N-1}).

        Returns
        -------
        torch.Tensor
            {N-1}-dimensional tensor of shape (d_0, ..., d_{collapse_dim - 1}, d_{collapse_dim + 1}, ..., d_{N-1}).
            Taking the `self.element_ix` element of the removed dimension.
        """
        return x.unbind(self.collapse_dim)[self.element_ix]


class MaxCollapse(nn.Module):
    """Global max collapsing over a specified dimension."""

    def __init__(self, collapse_dim=2):
        super().__init__()
        self.collapse_dim = collapse_dim

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            N-dimensional tensor of shape (d_0, d_1, ..., d_{N-1}).

        Returns
        -------
        torch.Tensor
            {N-1}-dimensional tensor of shape (d_0, ..., d_{collapse_dim - 1}, d_{collapse_dim + 1}, ..., d_{N-1}).
            Maximum over the removed dimension.
        """
        return x.max(dim=self.collapse_dim)[0]


class SumCollapse(nn.Module):
    """Global sum collapsing over a specified dimension."""

    def __init__(self, collapse_dim=2):
        super().__init__()
        self.collapse_dim = collapse_dim

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            N-dimensional tensor of shape (d_0, d_1, ..., d_{N-1}).

        Returns
        -------
        torch.Tensor
            {N-1}-dimensional tensor of shape (d_0, ..., d_{collapse_dim - 1}, d_{collapse_dim + 1}, ..., d_{N-1}).
            Sum over the removed dimension.
        """
        return x.sum(dim=self.collapse_dim)
