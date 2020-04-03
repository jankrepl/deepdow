"""Module containing neural networks."""
import torch
import torch.nn as nn

from .benchmarks import Benchmark
from .layers import (AttentionPool, ConvOneByOne, ConvTime, CovarianceMatrix, GammaOneByOne, MultiplyByConstant,
                     PoolTime, PortfolioOptimization, TimeCollapseRNN,)


class DummyNetwork(torch.nn.Module, Benchmark):
    """Minimal trainable network achieving the task.

    Parameters
    ----------
    n_channels : int
        Number of input channels. We learn one constant per channel. Therefore `n_channels=n_trainable_parameters`.
    """

    def __init__(self, n_channels=1):
        super().__init__()

        self.n_channels = n_channels
        self.mbc = MultiplyByConstant(n_channels=n_channels)

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
        return self.mbc(x)


class Whatever(nn.Module, Benchmark):
    def __init__(self, hidden_size, num_layers=1, fix_gamma=True, time_collapse='RNN', channel_collapse='att',
                 max_weight=1, channel_collapse_kwargs=None, time_collapse_kwargs=None, n_assets=None,
                 n_input_channels=1, shrinkage_strategy='diagonal', shrinkage_coef=0.5):
        """Construct."""
        self._mlflow_params = locals().copy()
        del self._mlflow_params['self']

        super().__init__()

        self.max_weight = max_weight
        self.n_input_channels = n_input_channels

        self.feature_extraction_layer = TimeCollapseRNN(n_input_channels,
                                                        hidden_size=hidden_size,
                                                        num_layers=num_layers,
                                                        hidden_strategy='many2many')
        self.covariance_layer = CovarianceMatrix(sqrt=True,
                                                 shrinkage_strategy=shrinkage_strategy,
                                                 shrinkage_coef=shrinkage_coef)

        if channel_collapse == 'avg':
            self.channel_collapse_layer = lambda x: x.mean(dim=1, keepdim=False)

        elif channel_collapse == '1b1':
            self.channel_collapse_layer = ConvOneByOne(hidden_size)

        elif channel_collapse == 'att':
            self.channel_collapse_layer = AttentionPool(hidden_size)

        else:
            raise ValueError('Unrecognized channel collapsing strategy:{}'.format(channel_collapse))

        if time_collapse == 'RNN':
            self.time_collapse_layer = TimeCollapseRNN(hidden_size, hidden_size, **(time_collapse_kwargs or {}))

        elif time_collapse == 'avg':
            self.time_collapse_layer = lambda x: x.mean(dim=2, keepdim=False)

        elif time_collapse == 'att':
            self.time_collapse_layer = AttentionPool(hidden_size)

        else:
            raise ValueError('Unrecognized time collapsing strategy:{}'.format(time_collapse))

        if not fix_gamma:
            self.gamma_layer = GammaOneByOne(hidden_size)
        else:
            self.gamma_layer = lambda x: (torch.ones(len(x)).to(x.device).to(x.dtype) * fix_gamma)  # no parameters

        if n_assets is not None:
            self.n_assets = n_assets
            self.portolioopt = PortfolioOptimization(n_assets, max_weight=self.max_weight)

        else:
            self.n_assets = None
            self.portolioopt = None

    def forward(self, x, debug_mode=False):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of samples of shape `(n_samples, self.n_input_channels, lookback, n_assets)`. Note that in different
            calls one can alter the `lookback` without problems. If `self.n_assets=None` then we can also dynamically
            change the number of assets. Otherwise it needs to be `self.n_assets`.

        debug_mode : bool
            If True returning multiple different objects. If False then only the weights.

        Returns
        -------
        weights : torch.Tensor
            Final weights of shape `(n_samples, n_assets)` that are solution to the convex optimization with parameters
            being determined by the extracted feature tensor by the CNN.

        """
        n_samples, n_input_channels, lookback, n_assets = x.shape

        x_inp = x.clone()

        # Checks
        if self.n_input_channels != n_input_channels:
            raise ValueError('Incorrect number of input channels: {}, expected: {}'.format(n_input_channels,
                                                                                           self.n_input_channels))

        # Setup convex optimization layer
        if self.portolioopt is not None:
            if self.n_assets != n_assets:
                raise ValueError('Incorrect number of assets: {}, expected: {}'.format(n_assets, self.n_assets))

            portolioopt = self.portolioopt
        else:
            # overhead
            portolioopt = PortfolioOptimization(n_assets, max_weight=self.max_weight)

        x = self.feature_extraction_layer(x)
        # time collapsing
        tc_features = self.time_collapse_layer(x)  # (n_samples, n_channels, n_assets)

        gamma = self.gamma_layer(tc_features)

        # rets = self.channel_collapse_layer(tc_features) + x_inp[:, 0, ...].mean(dim=1)
        rets = self.channel_collapse_layer(tc_features)
        # covmat_sqrt = self.covariance_layer(tc_features)
        covmat_sqrt = self.covariance_layer(x_inp[:, 0, ...])

        weights = portolioopt(rets, covmat_sqrt, gamma)

        return weights

    @property
    def n_parameters(self):
        """Compute number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def mlflow_params(self):
        return self._mlflow_params


class DowNet(nn.Module):
    """End to end model.

    This model expects a tensor of shape `(n_samples, n_input_channels, lookback, n_assets)` and performs the
    following steps:

    1. 1D convolutions over the lookback of each asset to accumulate new channels (technical indicators)
    2. Time collapsing resulting in `(n_samples, n_channels, n_assets)` tensor
    3. Channel collapsing resulting in `(n_samples, n_assets`) tensor used as expected return in optimization
    4. Solving convex optimization problem and outputing optimal weights of shape `(n_samples, n_assets)`


    Parameters
    ----------
    channels : tuple
        Tuple where the length represents the number of convolutions and the actual values are the channel sizes.

    kernel_size : int or tuple
        Kernel size in the time dimension during initial convolutions.

    pool : int
        Kernel size for pooling. Note that `pool=1` results in no pooling.

    fix_gamma : bool or float
        If True or nonzero float, then gamma from the portfolio optimization is fixed. Otherwise it is extracted
        via the one by one convolution and averaging from the feature tensor.

    channel_collapse : str, {'avg', '1b1', 'att'}
        A scheme how to get from the `(n_samples, n_channels, n_assets)` to `(n_samples, n_assets)`.

    time_collapse : str, {'avg', 'RNN'}
        A scheme how to get from the `(n_samples, n_channels, lookback, n_assets)` to
        `(n_samples, n_channels, n_assets)`.

    max_weight : int
        Maximum weight per asset in the portfolio optimization.

    shrinkage_strategy : None or {'diagonal', 'identity', 'scaled_identity'}
        Strategy of combining the sample covariance matrix with some more stable matrix.

    shrinkage_coef : float
        A float in the range [0, 1] representing the weight of the linear combination. If `shrinkage_coef=1` then
        using purely the sample covariance matrix. If `shrinkage_coef=0` then using purely the stable matrix.

    time_collapse_kwargs : None or dict
        Additional parameters to be passed into the time collapsing layer.

    channel_collapse_kwargs : None or dict
        Additional parameters to be passed into the channel collapsing layer.

    n_assets : None or int
        If specifed the network will always have to be provided with `n_assets` of assets in the forward pass. This
        way one can achieve major speedups since the optimization problem is canonicalized only once in the constructor.
        However, when `n_assets` is None the optimization problem is canonizalized before each forward pass - overhead.

    n_input_channels : int
        By default equal to 1 assuming the inputs are just one dimensional (i.e. returns from close prices). However,
        can be of arbitrary size (volumes, technical indicators, etc).

    Attributes
    ----------
    convolutions : nn.ModuleList
        Contains `ConvTime` layers representing consecutive 1D convolutions over the time dimension. In general
        transforms `(n_samples, n_input_channels, lookback, n_assets)` to
        `(n_samples, n_output_channels, lookback, n_assets)`.

    pool_layer : PoolTime
        Pooling layer over the time dimension. Transforms `(n_samples, n_input_channels, lookback_input, n_assets)` to
        `(n_samples, n_output_channels, lookback_output, n_assets)`.

    covariance_layer : CovarianceMatrix
        Covariance matrix layer.

    time_collapse_layer : TimeCollapseRNN or callable
        Layer transforming `(n_samples, n_channels, lookback, n_assets)` tensors to `(n_samples, n_channels, n_assets)`.
        Assuming that it cannot alter the `n_channels`.

    channel_collapse_layer : ConvOneByOne or AttentionPool or callable
        Layer transforming `(n_samples, n_channels, n_assets)` tensors to `(n_samples, n_assets)`.

    gamma_layer : ConvOneByOne or callable
        Layer transforming `(n_samples, n_channels, n_assets)` tensors to `(n_samples,)`.

    """

    def __init__(self, channels, kernel_size=3, pool=1, fix_gamma=True, time_collapse='RNN', channel_collapse='att',
                 max_weight=1, channel_collapse_kwargs=None, time_collapse_kwargs=None, n_assets=None,
                 n_input_channels=1, shrinkage_strategy='diagonal', shrinkage_coef=0.5):
        """Construct."""
        self._mlflow_params = locals()

        super().__init__()

        self.max_weight = max_weight
        self.n_input_channels = n_input_channels

        channels_ = [self.n_input_channels] + list(channels)
        kernel_size_ = kernel_size if isinstance(kernel_size, (tuple, list)) else len(channels) * [kernel_size]

        self.convolutions = nn.ModuleList([ConvTime(channels_[i], channels_[i + 1], kernel_size=kernel_size_[i])
                                           for i in range(len(channels))])

        self.covariance_layer = CovarianceMatrix(sqrt=True,
                                                 shrinkage_strategy=shrinkage_strategy,
                                                 shrinkage_coef=shrinkage_coef)

        self.pool_layer = PoolTime(pool)

        if channel_collapse == 'avg':
            self.channel_collapse_layer = lambda x: x.mean(dim=1, keepdim=False)

        elif channel_collapse == '1b1':
            self.channel_collapse_layer = ConvOneByOne(channels_[-1])

        elif channel_collapse == 'att':
            self.channel_collapse_layer = AttentionPool(channels_[-1])

        else:
            raise ValueError('Unrecognized channel collapsing strategy:{}'.format(channel_collapse))

        if time_collapse == 'RNN':
            self.time_collapse_layer = TimeCollapseRNN(channels_[-1], channels_[-1], **(time_collapse_kwargs or {}))

        elif time_collapse == 'avg':
            self.time_collapse_layer = lambda x: x.mean(dim=2, keepdim=False)

        elif time_collapse == 'att':
            self.time_collapse_layer = RealAttentionPool(channels_[-1])

        else:
            raise ValueError('Unrecognized time collapsing strategy:{}'.format(time_collapse))

        if not fix_gamma:
            self.gamma_layer = GammaOneByOne(channels_[-1])
        else:
            self.gamma_layer = lambda x: (torch.ones(len(x)).to(x.device).to(x.dtype) * fix_gamma)  # no parameters

        if n_assets is not None:
            self.n_assets = n_assets
            self.portolioopt = PortfolioOptimization(n_assets, max_weight=self.max_weight)

        else:
            self.n_assets = None
            self.portolioopt = None

    def forward(self, x, debug_mode=False):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of samples of shape `(n_samples, self.n_input_channels, lookback, n_assets)`. Note that in different
            calls one can alter the `lookback` without problems. If `self.n_assets=None` then we can also dynamically
            change the number of assets. Otherwise it needs to be `self.n_assets`.

        debug_mode : bool
            If True returning multiple different objects. If False then only the weights.

        Returns
        -------
        weights : torch.Tensor
            Final weights of shape `(n_samples, n_assets)` that are solution to the convex optimization with parameters
            being determined by the extracted feature tensor by the CNN.

        features : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)` representing extracted features after initial
            1D convolutions. Note that the `lookback` might be different if `kernel_size` not odd due to symmetric
            padding. Only returned if `self.debug_mode=True`.

        time_collapsed_features : torch.Tensor
            Tensor of shape `(n_samples, n_channels, n_assets)` representing the time collapsed features. Only returned
            if `self.debug_mode=True`.

        rets : torch.Tensor
            Expected returns of shape `(n_samples, n_assets)`. Fully determined by the CNN feature extractor. Only
            returned if `self.debug_mode=True`.

        covmat_sqrt : torch.Tensor
            Square root of the covariance matrix of shape `(n_samples, n_assets, n_assets)`. Fully determined by the CNN
            feature extractor. Only returned if `self.debug_mode=True`.

        gamma : torch.Tensor
            Risk and return tradeoff of shape `(n_samples,)`. If `fix_gamma` was False, then determined by the CNN
            feature extractor. Otherwise a user defined constant.  Only returned if `self.debug_mode=True`.

        """
        n_samples, n_input_channels, lookback, n_assets = x.shape

        x_inp = x.clone()

        # Checks
        if self.n_input_channels != n_input_channels:
            raise ValueError('Incorrent number of input channels: {}, expected: {}'.format(n_input_channels,
                                                                                           self.n_input_channels))

        # Setup convex optimization layer
        if self.portolioopt is not None:
            if self.n_assets != n_assets:
                raise ValueError('Incorrect number of assets: {}, expected: {}'.format(n_assets, self.n_assets))

            portolioopt = self.portolioopt
        else:
            # overhead
            portolioopt = PortfolioOptimization(n_assets, max_weight=self.max_weight)

        for i, conv in enumerate(self.convolutions):
            x = conv(x)
            if i != len(self.convolutions) - 1:
                x = torch.nn.functional.relu(x)
                x = self.pool_layer(x)

        # time collapsing
        tc_features = self.time_collapse_layer(x)  # (n_samples, n_channels, n_assets)

        gamma = self.gamma_layer(tc_features)

        rets = self.channel_collapse_layer(tc_features) + x_inp[:, 0, ...].mean(dim=1)

        # covmat_sqrt = self.covariance_layer(tc_features)
        covmat_sqrt = self.covariance_layer(x_inp[:, 0, ...])

        weights = portolioopt(rets, covmat_sqrt, gamma)

        if debug_mode:
            return weights, x, tc_features, rets, covmat_sqrt, gamma
        else:
            return weights

    @property
    def n_parameters(self):
        """Compute number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def lookback_invariant(self):
        """Determine whether invariant to lookback size."""
        return False

    @property
    def deterministic(self):
        return True
