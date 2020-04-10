"""miscellaneous layers."""

import torch
import torch.nn as nn


class CovarianceMatrix(nn.Module):
    """Convariance matrix or its square root.

    Attributes
    ----------
    sqrt : bool
        If True, then returning the square root.

    shrinkage_strategy : None or {'diagonal', 'identity', 'scaled_identity'}
        Strategy of combining the sample covariance matrix with some more stable matrix.

    shrinkage_coef : float
        A float in the range [0, 1] representing the weight of the linear combination. If `shrinkage_coef=1` then
        using purely the sample covariance matrix. If `shrinkage_coef=0` then using purely the stable matrix.
    """

    def __init__(self, sqrt=True, shrinkage_strategy='diagonal', shrinkage_coef=0.5):
        """Construct."""
        super().__init__()

        self.sqrt = sqrt

        if shrinkage_strategy is not None:
            if shrinkage_strategy not in {'diagonal', 'identity', 'scaled_identity'}:
                raise ValueError('Unrecognized shrinkage strategy {}'.format(shrinkage_strategy))

        self.shrinkage_strategy = shrinkage_strategy
        self.shrinkage_coef = shrinkage_coef

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, n_assets).

        Returns
        -------
        covmat : torch.Tensor
            Of shape (n_samples, n_assets, n_assets).

        """
        n_samples = x.shape[0]

        wrapper = self.compute_sqrt if self.sqrt else lambda h: h

        return torch.stack([wrapper(self.compute_covariance(x[i].T.clone(),
                                                            shrinkage_strategy=self.shrinkage_strategy,
                                                            shrinkage_coef=self.shrinkage_coef))
                            for i in range(n_samples)], dim=0)

    @staticmethod
    def compute_covariance(m, shrinkage_strategy=None, shrinkage_coef=0.5):
        """Compute covariance matrix for a single sample.

        Parameters
        ----------
        m : torch.Tensor
            Of shape (n_assets, n_channels).

        shrinkage_strategy : None or {'diagonal', 'identity', 'scaled_identity'}
            Strategy of combining the sample covariance matrix with some more stable matrix.

        shrinkage_coef : float
            A float in the range [0, 1] representing the weight of the linear combination. If `shrinkage_coef=1` then
            using purely the sample covariance matrix. If `shrinkage_coef=0` then using purely the stable matrix.

        Returns
        -------
        covmat_single : torch.Tensor
            Covariance matrix of shape (n_assets, n_assets).

        """
        fact = 1.0 / (m.size(1) - 1)
        m -= torch.mean(m, dim=1, keepdim=True)  # !!!!!!!!!!! INPLACE
        mt = m.t()

        s = fact * m.matmul(mt)  # sample covariance matrix
        s += torch.eye(len(s), dtype=s.dtype, device=s.device) * 0.001  # prevent numerical issues

        if shrinkage_strategy is None:
            return s

        elif shrinkage_strategy == 'identity':
            identity = torch.eye(len(s), device=s.device, dtype=s.dtype)

            return shrinkage_coef * s + (1 - shrinkage_coef) * identity

        elif shrinkage_strategy == 'scaled_identity':
            identity = torch.eye(len(s), device=s.device, dtype=s.dtype)
            scaled_identity = identity * torch.diag(s).mean()

            return shrinkage_coef * s + (1 - shrinkage_coef) * scaled_identity

        elif shrinkage_strategy == 'diagonal':
            diagonal = torch.diag(torch.diag(s))

            return shrinkage_coef * s + (1 - shrinkage_coef) * diagonal

    @staticmethod
    def compute_sqrt(m):
        """Compute the square root of a single positive definite matrix.

        Parameters
        ----------
        m : torch.Tensor
            Tensor of shape `(n_assets, n_assets)` representing the covariance matrix - needs to be PSD.

        Returns
        -------
        m_sqrt : torch.Tensor
            Tensor of shape `(n_assets, n_assets)` representing the square root of the covariance matrix.

        """
        _, s, v = m.svd()

        good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
        components = good.sum(-1)
        common = components.max()
        unbalanced = common != components.min()
        if common < s.size(-1):
            s = s[..., :common]  # pragma: no cover
            v = v[..., :common]  # pragma: no cover
            if unbalanced:  # pragma: no cover
                good = good[..., :common]  # pragma: no cover
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))  # pragma: no cover

        return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


class MultiplyByConstant(torch.nn.Module):
    """Multiplying constant.

    Parameters
    ----------
    dim_size : int
        Number of input channels. We learn one constant per channel. Therefore `dim_size=n_trainable_parameters`.

    dim_ix : int
        Which dimension to apply the multiplication to.
    """

    def __init__(self, dim_size=1, dim_ix=1):
        super().__init__()

        self.dim_size = dim_size
        self.dim_ix = dim_ix
        self.constant = torch.nn.Parameter(torch.ones(self.dim_size), requires_grad=True)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            N-dimensional tensor of shape (d_0, d_1, ..., d_{N-1})

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (d_0, d_1, ..., d_{N-1}).

        """
        if self.dim_size != x.shape[self.dim_ix]:
            raise ValueError('The size of dimension {} is {} which is different than {}'.format(self.dim_ix,
                                                                                                x.shape[self.dim_ix],
                                                                                                self.dim_size))
        view = [self.dim_size if i == self.dim_ix else 1 for i in range(x.ndim)]
        return x * self.constant.view(view)


class SoftmaxAllocator(torch.nn.Module):
    """Dummy portfolio creation by computing a softmax over the asset dimension."""

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_assets`).

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets`).

        """
        return nn.functional.softmax(x, dim=1)
