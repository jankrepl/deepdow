"""Module containing neural networks."""
import torch

from .benchmarks import Benchmark
from .layers import AttentionCollapse, AverageCollapse, CovarianceMatrix, Markowitz, MultiplyByConstant, RNN


class DummyNet(torch.nn.Module, Benchmark):
    """Minimal trainable network achieving the task.

    Parameters
    ----------
    n_channels : int
        Number of input channels. We learn one constant per channel. Therefore `n_channels=n_trainable_parameters`.
    """

    def __init__(self, n_channels=1):
        self._hparams = locals().copy()
        super().__init__()

        self.n_channels = n_channels
        self.mbc = MultiplyByConstant(dim_size=n_channels, dim_ix=1)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, lookback, n_assets).

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        temp = self.mbc(x)
        means = torch.abs(temp).mean(dim=[1, 2]) + 1e-6

        return means / (means.sum(dim=1, keepdim=True))

    @property
    def hparams(self):
        return {k: v if isinstance(v, (int, float, str)) else str(v) for k, v in self._hparams.items() if k != 'self'}


class BachelierNet(torch.nn.Module, Benchmark):
    def __init__(self, n_input_channels, n_assets, hidden_size=32, max_weight=1, shrinkage_strategy='diagonal'):
        self._hparams = locals().copy()
        super().__init__()
        self.norm_layer = torch.nn.InstanceNorm2d(n_input_channels, affine=True)
        self.transform_layer = RNN(n_input_channels, hidden_size=hidden_size)
        self.time_collapse_layer = AttentionCollapse(n_channels=hidden_size)
        self.covariance_layer = CovarianceMatrix(sqrt=False, shrinkage_strategy=shrinkage_strategy)
        self.channel_collapse_layer = AverageCollapse(collapse_dim=1)
        self.portfolio_opt_layer = Markowitz(n_assets, max_weight=max_weight)
        self.gamma = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        # Normalize
        x = self.norm_layer(x)

        # Covmat
        rets = x[:, 0, :, :]
        covmat = self.covariance_layer(rets)

        # expected returns
        x = self.transform_layer(x)
        x = self.time_collapse_layer(x)
        exp_rets = self.channel_collapse_layer(x)

        # gamma
        gamma_all = torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.gamma

        # weights
        weights = self.portfolio_opt_layer(exp_rets, covmat, gamma_all)

        return weights

    @property
    def hparams(self):
        return {k: v if isinstance(v, (int, float, str)) else str(v) for k, v in self._hparams.items() if k != 'self'}


class ThorpNet(torch.nn.Module, Benchmark):
    def __init__(self, n_assets, max_weight=1, force_symmetric=False):
        self._hparams = locals().copy()
        super().__init__()

        self.force_symmetric = force_symmetric
        self.matrix = torch.nn.Parameter(torch.eye(n_assets), requires_grad=True)
        self.exp_returns = torch.nn.Parameter(torch.zeros(n_assets), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones(1), requires_grad=True)

        self.portfolio_opt_layer = Markowitz(n_assets, max_weight=max_weight)

    def forward(self, x):
        n = len(x)

        covariance = torch.mm(self.matrix, torch.t(self.matrix)) if self.force_symmetric else self.matrix

        exp_returns_all = torch.repeat_interleave(self.exp_returns[None, ...], repeats=n, dim=0)
        covariance_all = torch.repeat_interleave(covariance[None, ...], repeats=n, dim=0)
        gamma_all = torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.gamma

        weights = self.portfolio_opt_layer(exp_returns_all, covariance_all, gamma_all)

        return weights

    @property
    def hparams(self):
        return {k: v if isinstance(v, (int, float, str)) else str(v) for k, v in self._hparams.items() if k != 'self'}
