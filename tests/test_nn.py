"""Tests focused on the nn module."""
import pytest
import torch

from deepdow.nn import BachelierNet, DummyNet, ThorpNet


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

        expected = n_assets * n_assets + n_assets + 1
        actual = sum(p.numel() for p in network.parameters() if p.requires_grad)

        assert expected == actual
