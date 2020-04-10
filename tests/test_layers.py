import pytest
import torch

from deepdow.layers import AverageCollapse, AttentionCollapse, ElementCollapse, MaxCollapse, SumCollapse
from deepdow.layers import Markowitz
from deepdow.layers import CovarianceMatrix, MultiplyByConstant, SoftmaxAllocator
from deepdow.layers import Conv, RNN

ALL_COLLAPSE = [AverageCollapse, AttentionCollapse, ElementCollapse, MaxCollapse, SumCollapse]
ALL_TRANSFORM = [Conv]


class TestCollapse:
    @pytest.mark.parametrize('layer', ALL_COLLAPSE)
    def test_default(self, layer, Xy_dummy):
        X, _, _, _ = Xy_dummy

        n_samples, n_channels, lookback, n_assets = X.shape

        try:
            layer_inst = layer(n_channels=n_channels)
        except TypeError:
            layer_inst = layer()

        layer_inst.to(device=X.device, dtype=X.dtype)

        res = layer_inst(X)

        assert torch.is_tensor(res)
        assert X.ndim == res.ndim + 1
        assert X.device == res.device
        assert X.dtype == res.dtype
        assert res.shape == (*X.shape[:2], X.shape[-1])


class TestConv:

    def test_wrong_method(self):
        with pytest.raises(ValueError):
            Conv(1, 2, method='FAKE')

    @pytest.mark.parametrize('n_output_channels', [1, 5], ids=['n_output_channels_1', 'n_output_channels_5'])
    @pytest.mark.parametrize('kernel_size', [1, 3], ids=['kernel_size_1', 'kernel_size_3'])
    @pytest.mark.parametrize('method', ['1D', '2D'])
    def test_default(self, Xy_dummy, method, kernel_size, n_output_channels):
        X, _, _, _ = Xy_dummy
        n_samples, n_channels, lookback, n_assets = X.shape

        if method == '1D':
            X = X.mean(dim=2)

        layer_inst = Conv(n_channels,
                          n_output_channels,
                          kernel_size=kernel_size,
                          method=method)

        layer_inst.to(device=X.device, dtype=X.dtype)

        res = layer_inst(X)

        assert torch.is_tensor(res)
        assert X.ndim == res.ndim
        assert X.device == res.device
        assert X.dtype == res.dtype

        assert X.shape[0] == res.shape[0]
        assert res.shape[1] == n_output_channels
        assert X.shape[2:] == res.shape[2:]


class TestCovarianceMatrix:

    def test_wrong_construction(self):
        with pytest.raises(ValueError):
            CovarianceMatrix(shrinkage_strategy='fake')

    @pytest.mark.parametrize('shrinkage_strategy', ['diagonal', 'identity', 'scaled_identity', None])
    @pytest.mark.parametrize('sqrt', [True, False], ids=['sqrt', 'nosqrt'])
    def test_basic(self, Xy_dummy, sqrt, shrinkage_strategy):
        X, _, _, _ = Xy_dummy
        n_samples, n_channels, lookback, n_assets = X.shape

        X_ = X.mean(dim=1)

        if n_channels == 1:
            with pytest.raises(ZeroDivisionError):
                CovarianceMatrix(sqrt, shrinkage_strategy=shrinkage_strategy)(X_)
        else:
            out = CovarianceMatrix(sqrt, shrinkage_strategy=shrinkage_strategy)(X_)

            assert out.shape == (n_samples, n_assets, n_assets)

    @pytest.mark.parametrize('sqrt', [True, False], ids=['sqrt', 'nosqrt'])
    def test_n_parameters(self, sqrt):
        layer = CovarianceMatrix()

        n_parameters = sum(p.numel() for p in layer.parameters() if p.requires_grad)

        assert n_parameters == 0

    def test_sqrt_works(self):
        n_samples = 3
        n_channels = 4
        n_assets = 5

        x = torch.rand((n_samples, n_channels, n_assets)) * 100

        cov = CovarianceMatrix(sqrt=False)(x)
        cov_sqrt = CovarianceMatrix(sqrt=True)(x)

        assert (n_samples, n_assets, n_assets) == cov.shape == cov_sqrt.shape

        for i in range(n_samples):
            assert torch.allclose(cov[i], cov_sqrt[i] @ cov_sqrt[i], atol=1e-2)


class TestMarkowitz:

    def test_basic(self, Xy_dummy):
        X, _, _, _ = Xy_dummy
        device, dtype = X.device, X.dtype
        n_samples, n_channels, lookback, n_assets = X.shape

        popt = Markowitz(n_assets)

        rets = X.mean(dim=(1, 2))

        covmat_sqrt__ = torch.rand((n_assets, n_assets)).to(device=X.device, dtype=X.dtype)
        covmat_sqrt_ = covmat_sqrt__ @ covmat_sqrt__
        covmat_sqrt_.add_(torch.eye(n_assets, dtype=dtype, device=device))

        covmat_sqrt = torch.stack(n_samples * [covmat_sqrt_])

        gamma = (torch.rand(n_samples) * 5 + 0.1).to(device=X.device, dtype=X.dtype)

        weights = popt(rets, covmat_sqrt, gamma)

        assert weights.shape == (n_samples, n_assets)
        assert weights.dtype == X.dtype
        assert weights.device == X.device


class TestMultiplyByConstant:

    def test_error(self):
        with pytest.raises(ValueError):
            MultiplyByConstant(dim_ix=1, dim_size=2)(torch.ones((2, 3)))

    @pytest.mark.parametrize('dim_ix', [1, 2, 3])
    def test_basic(self, Xy_dummy, dim_ix):
        X, _, _, _ = Xy_dummy

        layer_inst = MultiplyByConstant(dim_ix=dim_ix, dim_size=X.shape[dim_ix])

        layer_inst.to(device=X.device, dtype=X.dtype)

        res = layer_inst(X)

        assert torch.is_tensor(res)
        assert X.device == res.device
        assert X.dtype == res.dtype
        assert res.shape == X.shape


class TestRNN:
    @pytest.mark.parametrize('bidirectional', [True, False], ids=['bidir', 'onedir'])
    @pytest.mark.parametrize('cell_type', ['LSTM', 'RNN'])
    @pytest.mark.parametrize('hidden_size', [4, 6])
    @pytest.mark.parametrize('n_layers', [1, 2])
    def test_basic(self, Xy_dummy, hidden_size, bidirectional, cell_type, n_layers):
        X, _, _, _ = Xy_dummy
        n_samples, n_channels, lookback, n_assets = X.shape

        layer_inst = RNN(n_channels,
                         hidden_size,
                         n_layers=n_layers,
                         bidirectional=bidirectional,
                         cell_type=cell_type)

        layer_inst.to(device=X.device, dtype=X.dtype)

        res = layer_inst(X)

        assert torch.is_tensor(res)
        assert X.ndim == res.ndim
        assert X.device == res.device
        assert X.dtype == res.dtype

        assert X.shape[0] == res.shape[0]
        assert res.shape[1] == hidden_size
        assert X.shape[2:] == res.shape[2:]

    @pytest.mark.parametrize('bidirectional', [True, False], ids=['bidir', 'onedir'])
    @pytest.mark.parametrize('cell_type', ['LSTM', 'RNN'])
    @pytest.mark.parametrize('hidden_size', [4, 6])
    @pytest.mark.parametrize('n_channels', [1, 4])
    def test_n_parameters(self, n_channels, hidden_size, cell_type, bidirectional):
        layer = RNN(n_channels, hidden_size, bidirectional=bidirectional, cell_type=cell_type)

        n_parameters = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        n_dir = (1 + int(bidirectional))
        hidden_size_a = int(hidden_size // n_dir)

        if cell_type == 'RNN':
            assert n_parameters == n_dir * (
                    (n_channels * hidden_size_a) + (hidden_size_a * hidden_size_a) + 2 * hidden_size_a)

        else:
            assert n_parameters == n_dir * 4 * (
                    (n_channels * hidden_size_a) + (hidden_size_a * hidden_size_a) + 2 * hidden_size_a)

    def test_error(self):

        with pytest.raises(ValueError):
            RNN(2, 4, cell_type='FAKE')

        with pytest.raises(ValueError):
            RNN(3, 3, cell_type='LSTM', bidirectional=True)


class TestSoftmax:
    def test_basic(self, Xy_dummy):
        eps = 1e-5
        X, _, _, _ = Xy_dummy
        dtype, device = X.dtype, X.device
        n_samples, n_channels, lookback, n_assets = X.shape

        rets = X.mean(dim=(1, 2))

        weights = SoftmaxAllocator()(rets)

        assert weights.shape == (n_samples, n_assets)
        assert weights.dtype == X.dtype
        assert weights.device == X.device
        assert torch.all(-eps <= weights) and torch.all(weights <= 1 + eps)
        assert torch.allclose(weights.sum(dim=1), torch.ones(n_samples).to(dtype=dtype, device=device), atol=eps)
