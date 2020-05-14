"""Module containing neural networks."""
import torch

from .benchmarks import Benchmark
from .layers import (AttentionCollapse, AverageCollapse, CovarianceMatrix, Conv, NumericalMarkowitz, MultiplyByConstant,
                     RNN, SoftmaxAllocator)


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
        """Hyperparamters relevant to construction of the model."""
        return {k: v if isinstance(v, (int, float, str)) else str(v) for k, v in self._hparams.items() if k != 'self'}


class BachelierNet(torch.nn.Module, Benchmark):
    """Combination of recurrent neural networks and convex optimization.

    Parameters
    ----------
    n_input_channels : int
        Number of input channels of the dataset.

    n_assets : int
        Number of assets in our dataset. Note that this network is shuffle invariant along this dimension.

    hidden_size : int
        Hidden state size. Alternatively one can see it as number of output channels.

    max_weight : float
        Maximum weight for a single asset.

    shrinkage_strategy : str, {'diagonal', 'identity', 'scaled_identity'}
        Strategy of estimating the covariance matrix.

    p : float
        Dropout rate - probability of an element to be zeroed during dropout.

    Attributes
    ----------
    norm_layer : torch.nn.Module
        Instance normalization (per channel).

    transform_layer : deepdow.layers.RNN
        RNN layer that transforms `(n_samples, n_channels, lookback, n_assets)` to
        `(n_samples, hidden_size, lookback, n_assets)` where the first (sample) and the last dimension (assets) is
        shuffle invariant.

    time_collapse_layer : deepdow.layers.AttentionCollapse
        Attention pooling layer that turns  `(n_samples, hidden_size, lookback, n_assets)` into
        `(n_samples, hidden_size, n_assets)` by assigning each timestep in the lookback dimension a weight and
        then performing a weighted average.

    dropout_layer : torch.nn.Module
        Dropout layer where the probability is controled by the parameter `p`.

    covariance_layer : deepdow.layers.CovarianceMatrix
        Estimate square root of a covariance metric for the optimization. Turns `(n_samples, lookback, n_assets)` to
        `(n_samples, n_assets, n_assets)`.

    channel_collapse_layer : deepdow.layers.AverageCollapse
        Averaging layer turning `(n_samples, hidden_size, n_assets)` to `(n_samples, n_assets)` where the output
        serves as estimate of expected returns in the optimization.

    gamma : torch.nn.Parameter
        A single learnable parameter that will be used for all samples. It represents the tradoff between risk and
        return. If equal to zero only expected returns are considered.

    alpha : torch.nn.Parameter
        A single learnable parameter that will be used for all samples. It represents the regularization strength of
        portfolio weights. If zero then no effect if high then encourages weights to be closer to zero.

    portfolio_opt_layer : deepdow.layers.NumericalMarkowitz
        Markowitz optimizer that inputs expected returns, square root of a covariance matrix and a gamma

    """

    def __init__(self, n_input_channels, n_assets, hidden_size=32, max_weight=1, shrinkage_strategy='diagonal', p=0.5):
        self._hparams = locals().copy()
        super().__init__()
        self.norm_layer = torch.nn.InstanceNorm2d(n_input_channels, affine=True)
        self.transform_layer = RNN(n_input_channels, hidden_size=hidden_size)
        self.dropout_layer = torch.nn.Dropout(p=p)
        self.time_collapse_layer = AttentionCollapse(n_channels=hidden_size)
        self.covariance_layer = CovarianceMatrix(sqrt=False, shrinkage_strategy=shrinkage_strategy)
        self.channel_collapse_layer = AverageCollapse(collapse_dim=1)
        self.portfolio_opt_layer = NumericalMarkowitz(n_assets, max_weight=max_weight)
        self.gamma = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)

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
        # Normalize
        x = self.norm_layer(x)

        # Covmat
        rets = x[:, 0, :, :]
        covmat = self.covariance_layer(rets)

        # expected returns
        x = self.transform_layer(x)
        x = self.dropout_layer(x)
        x = self.time_collapse_layer(x)
        exp_rets = self.channel_collapse_layer(x)

        # gamma
        gamma_all = torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.gamma
        alpha_all = torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.alpha

        # weights
        weights = self.portfolio_opt_layer(exp_rets, covmat, gamma_all, alpha_all)

        return weights

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {k: v if isinstance(v, (int, float, str)) else str(v) for k, v in self._hparams.items() if k != 'self'}


class LinearNet(torch.nn.Module, Benchmark):
    """Network with one layer.

    Parameters
    ----------
    n_channels : int
        Number of channels, needs to be fixed for each input tensor.

    lookback : int
        Lookback, needs to be fixed for each input tensor.

    n_assets : int
        Number of assets, needs to be fixed for each input tensor.

    p : float
        Dropout probability.

    Attributes
    ----------
    norm_layer : torch.nn.InstanceNorm2d
        Instance normalization (per channel) with learnable paramters.

    dropout_layer : torch.nn.Dropout
        Dropout layer with probability `p`.

    linear : torch.nn.Linear
        One dense layer with `n_assets` outputs and the flattened input tensor `(n_channels, lookback, n_assets)`.

    temperature : torch.Parameter
        Learnable parameter for representing the final softmax allocator temperature.

    allocate_layer : SoftmaxAllocator
        Softmax allocator with a per sample temperature.

    """

    def __init__(self, n_channels, lookback, n_assets, p=0.5):
        self._hparams = locals().copy()
        super().__init__()

        self.n_channels = n_channels
        self.lookback = lookback
        self.n_assets = n_assets

        n_features = self.n_channels * self.lookback * self.n_assets

        self.norm_layer = torch.nn.InstanceNorm2d(self.n_channels, affine=True)
        self.dropout_layer = torch.nn.Dropout(p=p)
        self.linear = torch.nn.Linear(n_features, n_assets, bias=True)

        self.temperature = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.allocate_layer = SoftmaxAllocator(temperature=None)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, n_channels, lookback, n_assets). The last 3 dimensions need to be of the same
            size as specified in the constructor. They cannot vary.

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        if x.shape[1:] != (self.n_channels, self.lookback, self.n_assets):
            raise ValueError('Input x has incorrect shape {}'.format(x.shape))

        n_samples, _, _, _ = x.shape

        # Normalize
        x = self.norm_layer(x)
        x = self.dropout_layer(x)
        x = x.view(n_samples, -1)  # flatten
        x = self.linear(x)

        temperatures = torch.ones(n_samples).to(device=x.device, dtype=x.dtype) * self.temperature
        weights = self.allocate_layer(x, temperatures)

        return weights

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {k: v if isinstance(v, (int, float, str)) else str(v) for k, v in self._hparams.items() if k != 'self'}


class ThorpNet(torch.nn.Module, Benchmark):
    """All inputs of convex optimization are learnable but do not depend on the input.

    Parameters
    ----------
    n_assets : int
        Number of assets in our dataset. Note that this network is shuffle invariant along this dimension.

    force_symmetric : bool
        If True, then the square root of the covariance matrix will be always by construction symmetric.
        The resulting array will be :math:`M^T M` where :math:`M` is the learnable parameter. If `False` then
        no guarantee of the matrix being symmetric.

    max_weight : float
        Maximum weight for a single asset.


    Attributes
    ----------
    matrix : torch.nn.Parameter
        A learnable matrix of shape `(n_assets, n_assets)`.

    exp_returns : torch.nn.Parameter
        A learnable vector of shape `(n_assets,)`.

    gamma : torch.nn.Parameter
        A single learnable parameter that will be used for all samples. It represents the tradoff between risk and
        return. If equal to zero only expected returns are considered.

    alpha : torch.nn.Parameter
        A single learnable parameter that will be used for all samples. It represents the regularization strength of
        portfolio weights. If zero then no effect if high then encourages weights to be closer to zero.

    """

    def __init__(self, n_assets, max_weight=1, force_symmetric=False):
        self._hparams = locals().copy()
        super().__init__()

        self.force_symmetric = force_symmetric
        self.matrix = torch.nn.Parameter(torch.eye(n_assets), requires_grad=True)
        self.exp_returns = torch.nn.Parameter(torch.zeros(n_assets), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=True)

        self.portfolio_opt_layer = NumericalMarkowitz(n_assets, max_weight=max_weight)

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
        n = len(x)

        covariance = torch.mm(self.matrix, torch.t(self.matrix)) if self.force_symmetric else self.matrix

        exp_returns_all = torch.repeat_interleave(self.exp_returns[None, ...], repeats=n, dim=0)
        covariance_all = torch.repeat_interleave(covariance[None, ...], repeats=n, dim=0)
        gamma_all = torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.gamma
        alpha_all = torch.ones(len(x)).to(device=x.device, dtype=x.dtype) * self.alpha

        weights = self.portfolio_opt_layer(exp_returns_all, covariance_all, gamma_all, alpha_all)

        return weights

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {k: v if isinstance(v, (int, float, str)) else str(v) for k, v in self._hparams.items() if k != 'self'}
