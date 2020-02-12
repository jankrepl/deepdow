"""Module containing neural networks."""
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Pooling over the channels with attention.

    Parameters
    ----------
    n_channels : int
        Number of input channels.

    Attributes
    ----------
    linear : nn.Module
        Fully connected layer assigning each channel a weight.

    """

    def __init__(self, n_channels):
        """Construct."""
        super(AttentionPool, self).__init__()
        self.linear = nn.Linear(n_channels, n_channels)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, n_assets)`.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, n_assets)`.

        """
        n_samples, n_channels, n_assets = x.shape

        res = (torch.stack([self.linear(x[..., i]) for i in range(n_assets)], dim=2) * x).mean(dim=1, keepdim=False)

        return res


class ConvOneByOne(nn.Module):
    """One by one convolution iterated over all assets.

    This layer is mainly conceived to help us go from the extracted per asset features of shape
    (n_samples, n_channels, n_assets) to an expected return (conceptually) tensor of shape (n_samples, n_assets).

    Parameters
    ----------
    n_channels : int
        Number of channels of the previous tensor.

    Attributes
    ----------
    linear : tensor.nn.Linear
        Linear model to be applied to each asset over the channel dimension.

    """

    def __init__(self, n_channels):
        """Construct."""
        super(ConvOneByOne, self).__init__()

        self.linear = nn.Linear(n_channels, 1)

    def forward(self, x):
        """Forward.

        Parameters
        ----------
        x : torch.Tensor
            Of shape `(n_samples, n_channels, n_assets)`. It should represent all relevant features
            (technical indicators) for each asset.

        Returns
        -------
        torch.Tensor
            Of shape `(n_samples, n_assets)`. It should represent one overall indicator (expected return) for each
            asset.
        """
        n_samples, n_channels, n_assets = x.shape

        res = torch.stack([torch.cat([self.linear(x[i, :, a]) for a in range(n_assets)]) for i in range(n_samples)])

        return res


class ConvTime(nn.Module):
    """Convolution over the time dimension shared across assets.

    Parameters
    ----------
    n_input_channels : int
        Number of input channels (number of channels of the previous layer). Note that if applied to the
        original input then will be one.

    n_output_channels : int
        Number of output channels.

    Notes
    -----
    We basically iterate the same 1D convolution over all assets. Therefore the number of assets is irrelevant.
    """

    def __init__(self, n_input_channels, n_output_channels, kernel_size=3):
        """Construct.

        Notes
        -----
        Note that we would like to use padding='same' in tensorflow parlance. It is however only possible with odd
        kernel sizes. This is because torch does only symmetric padding.
        """
        super(ConvTime, self).__init__()
        self.conv1d = nn.Conv1d(n_input_channels, n_output_channels, kernel_size=kernel_size,
                                padding=(kernel_size - 1) // 2)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape `(n_samples, n_input_channels, lookback, n_assets)`.

        Returns
        -------
        torch.Tensor
            Of shape `(n_samples, n_output_channels, lookback, n_assets)`.
        """
        _, _, _, n_assets = x.shape

        return torch.stack([self.conv1d(x[..., i]) for i in range(n_assets)], dim=-1)


class CovarianceMatrix(nn.Module):
    """Convariance matrix or its sqaure root.

    It helps us go from the extracted features tensor of shape (n_samples, n_channels, n_assets) to their covariance
    of shape (n_samples, n_assets, n_assets). Is positive semi-definite by construction. Currently it is used
    as a square root of the risk model in the convex optimization.

    Attributes
    ----------
    sqrt : bool
        If True, then returning the square root.
    """

    def __init__(self, sqrt=True):
        """Construct."""
        super(CovarianceMatrix, self).__init__()

        self.sqrt = sqrt

    def compute_sqrt(self, m):
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
            s = s[..., :common]
            v = v[..., :common]
            if unbalanced:
                good = good[..., :common]  # pragma: no cover
        if unbalanced:
            s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))  # pragma: no cover

        return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)

    def single_sample(self, m):
        """Compute covariance matrix for a single sample.

        Parameters
        ----------
        m : torch.Tensor
            Of shape (n_assets, n_channels).

        Returns
        -------
        covmat_single : torch.Tensor
            Covariance matrix of shape (n_assets, n_assets).

        """
        fact = 1.0 / (m.size(1) - 1)
        m -= torch.mean(m, dim=1, keepdim=True)  # !!!!!!!!!!! INPLACE
        mt = m.t()

        return fact * m.matmul(mt)

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

        return torch.stack([wrapper(self.single_sample(x[i].T.clone())) for i in range(n_samples)], dim=0)


class GammaOneByOne(nn.Module):
    """Utility layer for Gamma.

    Parameters
    ----------
    n_channels : int
        Number of input channels.

    min_value : float
        Minimum allowed value for gamma - used to avoid problems in portfolio optimization.

    Attributes
    ----------
    onebyone : ConvOneByOne
        One by one convolution removing the channel dimension.
    """

    def __init__(self, n_channels, min_value=0.01):
        """Construct."""
        super(GammaOneByOne, self).__init__()

        self.onebyone = ConvOneByOne(n_channels)
        self.min_value = min_value

    def forward(self, x):
        """Forward.

        Parameters
        ----------
        x : torch.Tensor
            Of shape `(n_samples, n_channels, n_assets)`. It should represent all relevant features
            (technical indicators) for each asset.

        Returns
        -------
        torch.Tensor
            Of shape `(n_samples,)`. That represents the gamma (risk return trade-off) in portfolio optimization.
        """
        n_samples, _, _ = x.shape

        return torch.max(torch.ones(n_samples) * self.min_value, (torch.mean(self.onebyone(x), dim=-1)))


class PortfolioOptimization(nn.Module):
    """Convex optimization layer stylized into portfolio optimization problem.

    Parameters
    ----------
    n_assets : int
        Number of assets.

    Attributes
    ----------
    cvxpylayer : CvxpyLayer
        Custom layer used by a third party package called cvxpylayers.

    Notes
    -----
    The idea is to reinstantiate this layer inside of a forward pass of the main network. This way we can dynamically
    determine `n_assets` and reuse existing tensors as parameters of the optimization problem.

    References
    ----------
    [1] https://github.com/cvxgrp/cvxpylayers

    """

    def __init__(self, n_assets, max_weight=1):
        """Construct."""
        super(PortfolioOptimization, self).__init__()
        covmat_sqrt = cp.Parameter((n_assets, n_assets))
        rets = cp.Parameter(n_assets)

        w = cp.Variable(n_assets)
        ret = rets @ w
        risk = cp.sum_squares(covmat_sqrt @ w)

        prob = cp.Problem(cp.Maximize(ret - risk),
                          [cp.sum(w) == 1,
                           w >= 0,
                           w <= max_weight
                           ])

        assert prob.is_dpp()

        self.cvxpylayer = CvxpyLayer(prob, parameters=[rets, covmat_sqrt], variables=[w])

    def forward(self, rets, covmat_sqrt, gamma):
        """Perform forward pass.

        Parameters
        ----------
        rets : torch.Tensor
            Of shape (n_samples, n_assets) representing expected returns (or whatever the feature extractor decided
            to encode).

        covmat_sqrt : torch.Tensor
            Of shape (n_samples, n_assets, n_assets) representing the square of the covariance matrix. Fully determined
            by the feature extractor by in standard Markowitz setup it represents the risk model.

        gamma : torch.Tensor
            Of shape (n_samples,) representing the tradeoff between risk and return - where on efficient frontier
            we are.

        Returns
        -------
        weights : torch.Tensor
            Of shape (n_samples, n_assets) representing the optimal weights as determined by the convex optimizer.

        """
        n_samples = rets.shape[0]

        results_list = [self.cvxpylayer(rets[i], gamma[i] * covmat_sqrt[i])[0] for i in range(n_samples)]

        return torch.stack(results_list, dim=0)


class TimeCollapseRNN_(nn.Module):
    """Many to one architecture."""

    def __init__(self, n_channels, hidden_size, hidden_strategy='many2many'):
        """Construct."""
        super(TimeCollapseRNN_, self).__init__()

        self.cell = torch.nn.RNNCell(n_channels, hidden_size)
        self.hidden_strategy = hidden_strategy

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
        x_swapped = x.permute(0, 2, 3, 1)  # n_samples, lookback, n_assets, n_channels
        res = []  # [(n_assets, hidden_size), (n_assets, hidden_s

        for i in range(n_samples):
            all_hidden = []
            for j in range(lookback):
                all_hidden.append(self.cell(x_swapped[i, j]).T)  # (hidden_size, n_assets)

            if self.hidden_strategy == 'many2one':
                res.append(all_hidden[-1])

            elif self.hidden_strategy == 'many2many':
                res.append(torch.mean(torch.stack(all_hidden, dim=0), dim=0))

            else:
                raise ValueError('Unrecognized hidden strategy {}'.format(self.hidden_strategy))

        return torch.stack(res)


class TimeCollapseRNN(nn.Module):
    """Time collapsing RNN."""

    def __init__(self, n_channels, hidden_size, hidden_strategy='many2many', cell_type='LSTM', bidirectional=True):
        """Construct."""
        super(TimeCollapseRNN, self).__init__()

        if hidden_size % 2 != 0 and bidirectional:
            raise ValueError('Hidden size needs to be divisible by two for bidirectional RNNs.')

        hidden_size_one_direction = int(hidden_size // (1 + int(bidirectional)))  # only will work out for

        if cell_type == 'RNN':
            self.cell = torch.nn.RNN(n_channels, hidden_size_one_direction, bidirectional=bidirectional)

        elif cell_type == 'LSTM':
            self.cell = torch.nn.LSTM(n_channels, hidden_size_one_direction, bidirectional=bidirectional)

        else:
            raise ValueError('Unsupported cell_type {}'.format(cell_type))

        self.hidden_strategy = hidden_strategy

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.


        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples, self.hidden_size, n_assets)`.

        """
        # lookback, n_assets, n_channels
        n_samples, n_channels, lookback, n_assets = x.shape
        x_swapped = x.permute(0, 2, 3, 1)  # n_samples, lookback, n_assets, n_channels
        res = []  # [(n_assets, hidden_size), (n_assets, hidden_s

        for i in range(n_samples):
            all_hidden_ = self.cell(x_swapped[i])[0]  # lookback, n_assets, hidden_size
            all_hidden = all_hidden_.permute(0, 2, 1)

            if self.hidden_strategy == 'many2one':
                res.append(all_hidden[-1])

            elif self.hidden_strategy == 'many2many':
                res.append(all_hidden.mean(dim=0))

            else:
                raise ValueError('Unrecognized hidden strategy {}'.format(self.hidden_strategy))

        return torch.stack(res)


class DowNet(nn.Module):
    """End to end model.

    This model expects a tensor of shape `(n_samples, 1, lookback, n_assets)` and performs the following steps:

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
                 max_weight=1, channel_collapse_kwargs=None, time_collapse_kwargs=None, sqrt=True):
        """Construct."""
        self.mlflow_params = locals()
        del self.mlflow_params['self']

        super(DowNet, self).__init__()

        self.max_weight = max_weight

        channels_ = [1] + list(channels)
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

    def forward(self, x, debug_mode=True):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of samples of shape `(n_samples, 1, lookback, n_assets)`. Note that in different calls one can
            alter the `lookback` and `n_assets` without problems. This is how we designed the forward pass and the
            constituent operations.

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
        n_samples, input_channels, lookback, n_assets = x.shape
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
