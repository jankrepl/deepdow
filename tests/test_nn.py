"""Tests focused on the nn module."""
import pytest
import torch

from deepdow.nn import (AttentionPool, DowNet, ConvOneByOne, ConvTime, CovarianceMatrix, GammaOneByOne,
                        PortfolioOptimization, TimeCollapseRNN, TimeCollapseRNN_)


class TestAttentionPool:
    def test_basic(self, feature_notime_tensor):
        n_samples, n_channels, n_assets = feature_notime_tensor.shape

        out = AttentionPool(n_channels)(feature_notime_tensor)

        assert out.shape == (n_samples, n_assets)

    @pytest.mark.parametrize('n_channels', [1, 3, 5])
    def test_n_parameters(self, n_channels):
        layer = AttentionPool(n_channels)

        n_parameters = sum(p.numel() for p in layer.parameters() if p.requires_grad)

        assert n_parameters == n_channels * n_channels + n_channels


class TestConvTime:

    @pytest.mark.parametrize('n_output_channels', [2, 10], ids=['output_channels_2', 'output_channels_10'])
    def test_basic(self, feature_tensor, n_output_channels):
        n_samples, n_input_channels, lookback, n_assets = feature_tensor.shape

        out = ConvTime(n_input_channels, n_output_channels)(feature_tensor)

        assert out.shape == (n_samples, n_output_channels, lookback, n_assets)

        with pytest.raises(RuntimeError):
            ConvTime(n_input_channels + 1, n_output_channels)(feature_tensor)

    @pytest.mark.parametrize('n_input_channels', [1, 4])
    @pytest.mark.parametrize('n_output_channels', [2, 3])
    @pytest.mark.parametrize('kernel_size', [4, 8])
    def test_n_parameters(self, n_input_channels, n_output_channels, kernel_size):
        layer = ConvTime(n_input_channels, n_output_channels, kernel_size=kernel_size)

        n_parameters = sum(p.numel() for p in layer.parameters() if p.requires_grad)

        assert n_parameters == n_input_channels * n_output_channels * kernel_size + n_output_channels

    @pytest.mark.parametrize('kernel_size', [3, 5, 10])
    @pytest.mark.parametrize('lookback', [4, 6, 9])
    @pytest.mark.parametrize('n_input_channels', [1, 2, 5])
    @pytest.mark.parametrize('n_output_channels', [1, 3, 4])
    def test_lookback_unchanged(self, lookback, kernel_size, n_input_channels, n_output_channels):
        n_samples = 2
        n_assets = 4

        x = torch.rand((n_samples, n_input_channels, lookback, n_assets))

        out = ConvTime(n_input_channels=n_input_channels, n_output_channels=n_output_channels)(x)

        assert out.shape == (n_samples, n_output_channels, lookback, n_assets)


class TestConvOneByOne:

    def test_basic(self, feature_notime_tensor):
        n_samples, n_channels, n_assets = feature_notime_tensor.shape

        out = ConvOneByOne(n_channels)(feature_notime_tensor)

        assert out.shape == (n_samples, n_assets)

        with pytest.raises(RuntimeError):
            ConvOneByOne(n_channels + 1)(feature_notime_tensor)

    @pytest.mark.parametrize('n_channels', [1, 4])
    def test_n_parameters(self, n_channels):
        layer = ConvOneByOne(n_channels)

        n_parameters = sum(p.numel() for p in layer.parameters() if p.requires_grad)

        assert n_parameters == n_channels + 1


class TestCovarianceMatrix:

    @pytest.mark.parametrize('sqrt', [True, False], ids=['sqrt', 'nosqrt'])
    def test_basic(self, feature_notime_tensor, sqrt):
        n_samples, n_channels, n_assets = feature_notime_tensor.shape

        if n_channels == 1:
            with pytest.raises(ZeroDivisionError):
                CovarianceMatrix(sqrt)(feature_notime_tensor)
        else:
            out = CovarianceMatrix(sqrt)(feature_notime_tensor)

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


class TestDownNet:

    @pytest.mark.parametrize('fix_gamma', [True, False])
    @pytest.mark.parametrize('channel_collapse', ['avg', '1b1', 'att'])
    @pytest.mark.parametrize('time_collapse', ['RNN', 'avg'])
    @pytest.mark.parametrize('n_assets', [2, 5, 10])
    @pytest.mark.parametrize('lookback', [4, 6, 9])
    def test_n_assets_and_lookback_irrelevant_new_instance(self, n_assets, lookback, time_collapse, channel_collapse,
                                                           fix_gamma):
        """Make sure that the network processes 2+ assets with different instantiations.

        Note that lookback >= kernel_size.
        """
        n_samples = 2
        channels = (2, 4)

        x = torch.rand((n_samples, 1, lookback, n_assets))

        weights, features, time_collapsed_features, rets, covmat, gamma = DowNet(channels,
                                                                                 kernel_size=5,
                                                                                 time_collapse=time_collapse,
                                                                                 channel_collapse=channel_collapse,
                                                                                 fix_gamma=fix_gamma)(x,
                                                                                                      debug_mode=True)

        # actual_lookback = lookback if ker % 2 != 0 else

        assert weights.shape == (n_samples, n_assets)
        assert features.shape == (n_samples, channels[-1], lookback, n_assets)
        assert time_collapsed_features.shape == (n_samples, channels[-1], n_assets)
        assert rets.shape == (n_samples, n_assets)
        assert covmat.shape == (n_samples, n_assets, n_assets)
        assert gamma.shape == (n_samples,)

    def test_errors(self):
        with pytest.raises(ValueError):
            DowNet((2, 4), channel_collapse='FAKE')

        with pytest.raises(ValueError):
            DowNet((2, 4), time_collapse='FAKE')

    def test_n_parameters(self):
        net = DowNet((2,), kernel_size=(3,), time_collapse='avg', channel_collapse='avg', fix_gamma=2)

        assert net.n_parameters == 2 * (3 + 1)

    def test_not_debug_mode(self):
        res = DowNet((4,))(torch.ones(2, 1, 5, 19), debug_mode=False)

        assert isinstance(res, torch.Tensor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_device(self):
        """Make sure that casting that correctly run on both GPU and CPU."""
        net = DowNet((2,))
        x = torch.ones((3, 1, 4, 5))

        device_cpu = torch.device('cpu')
        device_gpu = torch.device('cuda:0')

        res_cpu = net.to(device_cpu)(x.to(device_cpu))
        res_gpu = net.to(device_gpu)(x.to(device_gpu))

        for t in res_cpu:
            assert t.device == device_cpu

        for t in res_gpu:
            assert t.device == device_gpu

    def test_shuffle_invariant(self):
        """Make sure shuffling assets has no effect."""
        raise NotImplementedError()


class TestGammaOneByOne:
    def test_basic(self, feature_notime_tensor):
        n_samples, n_channels, n_assets = feature_notime_tensor.shape

        out = GammaOneByOne(n_channels)(feature_notime_tensor)

        assert out.shape == (n_samples,)

        with pytest.raises(RuntimeError):
            GammaOneByOne(n_channels + 1)(feature_notime_tensor)

    @pytest.mark.parametrize('n_channels', [1, 4])
    def test_n_parameters(self, n_channels):
        layer = GammaOneByOne(n_channels)

        n_parameters = sum(p.numel() for p in layer.parameters() if p.requires_grad)

        assert n_parameters == n_channels + 1


class TestPortfolioOptimization:

    @pytest.mark.parametrize('n_assets', [3, 4, 5])
    @pytest.mark.parametrize('n_samples', [1, 2])
    def test_basic(self, n_assets, n_samples):
        popt = PortfolioOptimization(n_assets)

        rets = torch.rand((n_samples, n_assets))

        covmat_sqrt__ = torch.rand((n_assets, n_assets))
        covmat_sqrt_ = covmat_sqrt__ @ covmat_sqrt__
        covmat_sqrt_.add_(torch.eye(n_assets))

        covmat_sqrt = torch.stack(n_samples * [covmat_sqrt_])

        gamma = torch.rand(n_samples) * 5 + 0.1

        weights = popt(rets, covmat_sqrt, gamma)

        assert weights.shape == (n_samples, n_assets)


class TestTimeCollapseRNN_:

    @pytest.mark.parametrize('hidden_strategy', ['many2many', 'many2one'])
    @pytest.mark.parametrize('hidden_size', [3, 5])
    def test_basic(self, feature_tensor, hidden_size, hidden_strategy):
        n_samples, n_channels, lookback, n_assets = feature_tensor.shape

        layer = TimeCollapseRNN_(n_channels, hidden_size, hidden_strategy=hidden_strategy)

        out = layer(feature_tensor)

        assert out.shape == (n_samples, hidden_size, n_assets)

    @pytest.mark.parametrize('hidden_size', [3, 5])
    @pytest.mark.parametrize('n_channels', [1, 4])
    def test_n_parameters(self, n_channels, hidden_size):
        layer = TimeCollapseRNN_(n_channels, hidden_size)

        n_parameters = sum(p.numel() for p in layer.parameters() if p.requires_grad)

        assert n_parameters == (n_channels * hidden_size) + (hidden_size * hidden_size) + 2 * hidden_size

    def test_error(self, feature_tensor):
        n_samples, n_channels, lookback, n_assets = feature_tensor.shape

        layer = TimeCollapseRNN_(n_channels, 4, hidden_strategy='fake')

        with pytest.raises(ValueError):
            layer(feature_tensor)


class TestTimeCollapseRNN:

    @pytest.mark.parametrize('bidirectional', [True, False], ids=['bidir', 'onedir'])
    @pytest.mark.parametrize('cell_type', ['LSTM', 'RNN'])
    @pytest.mark.parametrize('hidden_strategy', ['many2many', 'many2one'])
    @pytest.mark.parametrize('hidden_size', [4, 6])
    def test_basic(self, feature_tensor, hidden_size, hidden_strategy, bidirectional, cell_type):
        n_samples, n_channels, lookback, n_assets = feature_tensor.shape

        layer = TimeCollapseRNN(n_channels,
                                hidden_size,
                                hidden_strategy=hidden_strategy,
                                bidirectional=bidirectional)

        out = layer(feature_tensor)

        assert out.shape == (n_samples, hidden_size, n_assets)

    @pytest.mark.parametrize('bidirectional', [True, False], ids=['bidir', 'onedir'])
    @pytest.mark.parametrize('cell_type', ['LSTM', 'RNN'])
    @pytest.mark.parametrize('hidden_size', [4, 6])
    @pytest.mark.parametrize('n_channels', [1, 4])
    def test_n_parameters(self, n_channels, hidden_size, cell_type, bidirectional):
        layer = TimeCollapseRNN(n_channels, hidden_size, bidirectional=bidirectional, cell_type=cell_type)

        n_parameters = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        n_dir = (1 + int(bidirectional))
        hidden_size_a = int(hidden_size // n_dir)

        if cell_type == 'RNN':
            assert n_parameters == n_dir * (
                    (n_channels * hidden_size_a) + (hidden_size_a * hidden_size_a) + 2 * hidden_size_a)

        else:
            assert n_parameters == n_dir * 4 * (
                    (n_channels * hidden_size_a) + (hidden_size_a * hidden_size_a) + 2 * hidden_size_a)

    def test_error(self, feature_tensor):
        n_samples, n_channels, lookback, n_assets = feature_tensor.shape

        with pytest.raises(ValueError):
            TimeCollapseRNN(n_channels, 4, hidden_strategy='many2many', cell_type='FAKE')

        with pytest.raises(ValueError):
            TimeCollapseRNN(n_channels, 3, hidden_strategy='many2many', cell_type='LSTM', bidirectional=True)

        layer = TimeCollapseRNN(n_channels, 4, hidden_strategy='fake')

        with pytest.raises(ValueError):
            layer(feature_tensor)
