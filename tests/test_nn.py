"""Tests focused on the nn module."""
import pytest
import torch

from deepdow.nn import BachelierNet, DummyNet, KeynesNet, LinearNet, ThorpNet


class TestDummyNetwork:
    def test_basic(self, Xy_dummy):
        X, _, _, _ = Xy_dummy
        n_samples, n_channels, lookback, n_assets = X.shape
        dtype = X.dtype
        device = X.device

        network = DummyNet(n_channels=n_channels)
        network.to(device=device, dtype=dtype)

        weights = network(X)

        assert torch.is_tensor(weights)
        assert weights.shape == (n_samples, n_assets)
        assert X.device == weights.device
        assert X.dtype == weights.dtype
        assert torch.allclose(weights.sum(dim=1), torch.ones(n_samples).to(dtype=dtype, device=device), atol=1e-4)


class TestBachelierNet:
    @pytest.mark.parametrize('max_weight', [0.25, 0.5, 1], ids=['max_weight_0.25', 'max_weight_0.5', 'max_weight_1'])
    def test_basic(self, Xy_dummy, max_weight):
        eps = 1e-4

        X, _, _, _ = Xy_dummy
        n_samples, n_channels, lookback, n_assets = X.shape
        dtype = X.dtype
        device = X.device

        network = BachelierNet(n_channels, n_assets, max_weight=max_weight)
        network.to(device=device, dtype=dtype)

        weights = network(X)

        assert isinstance(network.hparams, dict)
        assert network.hparams
        assert torch.is_tensor(weights)
        assert weights.shape == (n_samples, n_assets)
        assert X.device == weights.device
        assert X.dtype == weights.dtype
        assert torch.all(-eps <= weights) and torch.all(weights <= max_weight + eps)
        assert torch.allclose(weights.sum(dim=1), torch.ones(n_samples).to(dtype=dtype, device=device), atol=eps)


class TestKeynesNet:
    def test_error(self):
        with pytest.raises(ValueError):
            KeynesNet(2, transform_type='FAKE', hidden_size=10, n_groups=2)

        with pytest.raises(ValueError):
            KeynesNet(2, hidden_size=10, n_groups=3)

    @pytest.mark.parametrize('transform_type', ['Conv', 'RNN'])
    @pytest.mark.parametrize('hidden_size', [4, 6])
    def test_basic(self, Xy_dummy, transform_type, hidden_size):
        eps = 1e-4

        X, _, _, _ = Xy_dummy
        n_samples, n_channels, lookback, n_assets = X.shape
        dtype = X.dtype
        device = X.device

        network = KeynesNet(n_channels, hidden_size=hidden_size, transform_type=transform_type, n_groups=2)  #
        network.to(device=device, dtype=dtype)

        weights = network(X)

        assert isinstance(network.hparams, dict)
        assert network.hparams
        assert torch.is_tensor(weights)
        assert weights.shape == (n_samples, n_assets)
        assert X.device == weights.device
        assert X.dtype == weights.dtype
        assert torch.allclose(weights.sum(dim=1), torch.ones(n_samples).to(dtype=dtype, device=device), atol=eps)

    @pytest.mark.parametrize('n_input_channels', [4, 8])
    @pytest.mark.parametrize('hidden_size', [16, 32])
    @pytest.mark.parametrize('n_groups', [2, 4, 8])
    @pytest.mark.parametrize('transform_type', ['Conv', 'RNN'])
    def test_n_params(self, n_input_channels, hidden_size, n_groups, transform_type):
        network = KeynesNet(n_input_channels=n_input_channels,
                            hidden_size=hidden_size,
                            n_groups=n_groups,
                            transform_type=transform_type)

        expected = 0
        expected += n_input_channels * 2  # instance norm
        if transform_type == 'Conv':
            expected += n_input_channels * 3 * hidden_size + hidden_size
        else:
            expected += 4 * ((n_input_channels * hidden_size) + (hidden_size * hidden_size) + 2 * hidden_size)
        expected += 2 * hidden_size  # group_norm
        expected += 1  # temperature

        actual = sum(p.numel() for p in network.parameters() if p.requires_grad)

        assert expected == actual


class TestLinear:
    def test_basic(self, Xy_dummy):
        eps = 1e-4

        X, _, _, _ = Xy_dummy
        n_samples, n_channels, lookback, n_assets = X.shape
        dtype = X.dtype
        device = X.device

        network = LinearNet(n_channels, lookback, n_assets)
        network.to(device=device, dtype=dtype)

        with pytest.raises(ValueError):
            network(torch.ones(n_samples, n_channels + 1, lookback, n_assets, device=device, dtype=dtype))

        weights = network(X)

        assert isinstance(network.hparams, dict)
        assert network.hparams
        assert torch.is_tensor(weights)
        assert weights.shape == (n_samples, n_assets)
        assert X.device == weights.device
        assert X.dtype == weights.dtype
        assert torch.allclose(weights.sum(dim=1), torch.ones(n_samples).to(dtype=dtype, device=device), atol=eps)

    @pytest.mark.parametrize('n_channels', [1, 3])
    @pytest.mark.parametrize('lookback', [2, 10])
    @pytest.mark.parametrize('n_assets', [40, 4])
    def test_n_params(self, n_channels, lookback, n_assets):
        network = LinearNet(n_channels, lookback, n_assets)

        n_features = n_channels * lookback * n_assets
        expected = 0
        expected += n_features * 2  # batch norm
        expected += n_features * n_assets + n_assets  # dense
        expected += 1  # temperature

        actual = sum(p.numel() for p in network.parameters() if p.requires_grad)

        assert expected == actual


class TestThorpNet:
    @pytest.mark.parametrize('force_symmetric', [True, False], ids=['symmetric', 'asymetric'])
    @pytest.mark.parametrize('max_weight', [0.25, 0.5, 1], ids=['max_weight_0.25', 'max_weight_0.5', 'max_weight_1'])
    def test_basic(self, Xy_dummy, max_weight, force_symmetric):
        eps = 1e-4

        X, _, _, _ = Xy_dummy
        n_samples, n_channels, lookback, n_assets = X.shape
        dtype = X.dtype
        device = X.device

        network = ThorpNet(n_assets, max_weight=max_weight, force_symmetric=force_symmetric)
        network.to(device=device, dtype=dtype)

        weights = network(X)

        assert isinstance(network.hparams, dict)
        assert network.hparams
        assert torch.is_tensor(weights)
        assert weights.shape == (n_samples, n_assets)
        assert X.device == weights.device
        assert X.dtype == weights.dtype
        assert torch.all(-eps <= weights) and torch.all(weights <= max_weight + eps)
        assert torch.allclose(weights.sum(dim=1), torch.ones(n_samples).to(dtype=dtype, device=device), atol=eps)

    @pytest.mark.parametrize('force_symmetric', [True, False], ids=['symmetric', 'asymetric'])
    @pytest.mark.parametrize('n_assets', [3, 5, 6])
    def test_n_params(self, n_assets, force_symmetric):
        network = ThorpNet(n_assets, force_symmetric=force_symmetric)

        expected = n_assets * n_assets + n_assets + 1 + 1
        actual = sum(p.numel() for p in network.parameters() if p.requires_grad)

        assert expected == actual
