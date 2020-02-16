"""Module containing neural networks."""
import torch
import torch.nn as nn

from .benchmarks import Benchmark
from .layers import (AttentionPool, ConvOneByOne, ConvTime, CovarianceMatrix, GammaOneByOne,
                     PortfolioOptimization, TimeCollapseRNN)


class DowNet(nn.Module, Benchmark):
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

    sqrt : int
        If True, then computing matrix square root after covariance computation. Note that the portfolio optimizer
        expects a square root of the covariance matrix however empirically there are some numerical issues.

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

    covariance : CovarianceMatrix
        Covariance matrix layer.

    time_collapse_layer : TimeCollapseRNN or callable
        Layer transforming `(n_samples, n_channels, lookback, n_assets)` tensors to `(n_samples, n_channels, n_assets)`.
        Assuming that it cannot alter the `n_channels`.

    channel_collapse_layer : ConvOneByOne or AttentionPool or callable
        Layer transforming `(n_samples, n_channels, n_assets)` tensors to `(n_samples, n_assets)`.

    gamma_layer : ConvOneByOne or callable
        Layer transforming `(n_samples, n_channels, n_assets)` tensors to `(n_samples,)`.

    """

    def __init__(self, channels, kernel_size=3, fix_gamma=True, time_collapse='RNN', channel_collapse='att',
                 max_weight=1, channel_collapse_kwargs=None, time_collapse_kwargs=None, sqrt=True, n_assets=None,
                 n_input_channels=1):
        """Construct."""
        self.mlflow_params = locals()
        del self.mlflow_params['self']

        super(DowNet, self).__init__()

        self.max_weight = max_weight
        self.n_input_channels = n_input_channels

        channels_ = [self.n_input_channels] + list(channels)
        kernel_size_ = kernel_size if isinstance(kernel_size, (tuple, list)) else len(channels) * [kernel_size]

        self.convolutions = nn.ModuleList([ConvTime(channels_[i], channels_[i + 1], kernel_size=kernel_size_[i])
                                           for i in range(len(channels))])

        self.covariance = CovarianceMatrix(sqrt=sqrt)

        if channel_collapse == 'avg':
            self.channel_collapse_layer = lambda x: x.mean(dim=1, keepdim=False)

        elif channel_collapse == '1b1':
            self.channel_collapse_layer = ConvOneByOne(channels[-1])

        elif channel_collapse == 'att':
            self.channel_collapse_layer = AttentionPool(channels[-1])

        else:
            raise ValueError('Unrecognized channel collapsing strategy:{}'.format(channel_collapse))

        if time_collapse == 'RNN':
            self.time_collapse_layer = TimeCollapseRNN(channels[-1], channels[-1], **(time_collapse_kwargs or {}))

        elif time_collapse == 'avg':
            self.time_collapse_layer = lambda x: x.mean(dim=2, keepdim=False)

        else:
            raise ValueError('Unrecognized time collapsing strategy:{}'.format(time_collapse))

        if not fix_gamma:
            self.gamma_layer = GammaOneByOne(channels[-1])
        else:
            self.gamma_layer = lambda x: (torch.ones(len(x)).to(x.device) * fix_gamma)  # no parameters

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
                x = torch.tanh(x)  # to be customized

        # time collapsing
        tc_features = self.time_collapse_layer(x)  # (n_samples, n_channels, n_assets)

        gamma = self.gamma_layer(tc_features)

        rets = self.channel_collapse_layer(tc_features)

        covmat_sqrt = self.covariance(tc_features)
        covmat_sqrt += torch.stack([torch.eye(n_assets).to(covmat_sqrt.device) for _ in range(n_samples)], dim=0)
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
