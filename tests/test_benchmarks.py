"""Collection of tests focused on `benchmarks` module."""
import pytest
import torch

from deepdow.benchmarks import Benchmark, InverseVolatility, MaximumReturn, MinimumVariance, OneOverN, Random, Singleton


class TestBenchmark:
    def test_errors(self, Xy_dummy):
        X_dummy, _, _, _ = Xy_dummy
        with pytest.raises(TypeError):
            Benchmark()

        class TempBenchmarkWrong(Benchmark):
            pass

        class TempBenchmarkCorrect(Benchmark):
            def __call__(self, X):
                return X * 2

        with pytest.raises(TypeError):
            TempBenchmarkWrong()

        temp = TempBenchmarkCorrect()
        temp(X_dummy)

        assert isinstance(temp.hparams, dict)


class TestInverseVolatility:
    @pytest.mark.parametrize('use_std', [True, False], ids=['use_std', 'use_var'])
    def test_basic(self, Xy_dummy, use_std):
        X_dummy, _, _, _ = Xy_dummy
        n_samples, n_channels, lookback, n_assets = X_dummy.shape
        dtype = X_dummy.dtype
        device = X_dummy.device
        bm = InverseVolatility(use_std=use_std)

        weights = bm(X_dummy)

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (n_samples, n_assets)
        assert weights.dtype == dtype
        assert weights.device == device
        assert torch.allclose(weights.sum(dim=1), torch.ones(n_samples).to(dtype=dtype, device=device))
        assert torch.all(weights >= 0) and torch.all(weights <= 1)
        assert isinstance(bm.hparams, dict) and bm.hparams


class TestMaximumReturn:

    @pytest.mark.parametrize('max_weight', [1, 0.5], ids=['max_weight=1', 'max_weight=0.5'])
    @pytest.mark.parametrize('predefined_assets', [True, False], ids=['fixed_assets', 'nonfixed_assets'])
    def test_basic(self, Xy_dummy, predefined_assets, max_weight):
        X_dummy, _, _, _ = Xy_dummy
        eps = 1e-4
        n_samples, n_channels, lookback, n_assets = X_dummy.shape
        dtype = X_dummy.dtype
        device = X_dummy.device

        X_more_assets = torch.cat([X_dummy, X_dummy], dim=-1)

        bm = MaximumReturn(n_assets=n_assets if predefined_assets else None,
                           max_weight=max_weight)

        weights = bm(X_dummy)

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (n_samples, n_assets)
        assert weights.dtype == dtype
        assert weights.device == device
        assert torch.allclose(weights.sum(dim=1), torch.ones(n_samples).to(dtype=dtype, device=device), atol=1e-4)
        assert torch.all(-eps <= weights) and torch.all(weights <= max_weight + eps)
        assert isinstance(bm.hparams, dict) and bm.hparams

        if predefined_assets:
            with pytest.raises(ValueError):
                bm(X_more_assets)

            return
        else:
            bm(X_more_assets)


class TestMinimumVariance:
    @pytest.mark.parametrize('max_weight', [1, 0.5], ids=['max_weight=1', 'max_weight=0.5'])
    @pytest.mark.parametrize('predefined_assets', [True, False], ids=['fixed_assets', 'nonfixed_assets'])
    def test_basic(self, Xy_dummy, predefined_assets, max_weight):
        X_dummy, _, _, _ = Xy_dummy
        eps = 1e-4
        n_samples, n_channels, lookback, n_assets = X_dummy.shape
        dtype = X_dummy.dtype
        device = X_dummy.device

        X_more_assets = torch.cat([X_dummy, X_dummy], dim=-1)

        bm = MinimumVariance(n_assets=n_assets if predefined_assets else None,
                             max_weight=max_weight)

        weights = bm(X_dummy)

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (n_samples, n_assets)
        assert weights.dtype == dtype
        assert weights.device == device
        assert torch.allclose(weights.sum(dim=1), torch.ones(n_samples).to(dtype=dtype, device=device), atol=1e-4)
        assert torch.all(-eps <= weights) and torch.all(weights <= max_weight + eps)
        assert isinstance(bm.hparams, dict) and bm.hparams

        if predefined_assets:
            with pytest.raises(ValueError):
                bm(X_more_assets)

            return
        else:
            bm(X_more_assets)


class TestOneOverN:
    def test_basic(self, Xy_dummy):
        X_dummy, _, _, _ = Xy_dummy
        n_samples, n_channels, lookback, n_assets = X_dummy.shape
        dtype = X_dummy.dtype
        device = X_dummy.device

        bm = OneOverN()
        weights = bm(X_dummy)

        assert isinstance(weights, torch.Tensor)
        assert weights.dtype == dtype
        assert weights.device == device
        assert torch.allclose(weights.sum(dim=1), torch.ones(n_samples).to(dtype=dtype, device=device))
        assert len(torch.unique(weights)) == 1
        assert isinstance(bm.hparams, dict) and not bm.hparams


class TestRandom:
    def test_basic(self, Xy_dummy):
        X_dummy, _, _, _ = Xy_dummy
        n_samples, n_channels, lookback, n_assets = X_dummy.shape
        dtype = X_dummy.dtype
        device = X_dummy.device
        bm = Random()

        weights = bm(X_dummy)

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (n_samples, n_assets)
        assert weights.dtype == dtype
        assert weights.device == device
        assert torch.allclose(weights.sum(dim=1), torch.ones(n_samples).to(dtype=dtype, device=device))

        assert torch.all(weights >= 0) and torch.all(weights <= 1)
        assert isinstance(bm.hparams, dict) and not bm.hparams


class TestSingleton:

    @pytest.mark.parametrize('asset_ix', [0, 3])
    def test_basic(self, asset_ix, Xy_dummy):
        X_dummy, _, _, _ = Xy_dummy
        n_samples, n_channels, lookback, n_assets = X_dummy.shape
        dtype = X_dummy.dtype
        device = X_dummy.device
        bm = Singleton(asset_ix=asset_ix)
        weights = bm(X_dummy)

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (n_samples, n_assets)
        assert weights.dtype == dtype
        assert weights.device == device
        assert torch.allclose(weights.sum(dim=1), torch.ones(n_samples).to(dtype=dtype, device=device))

        assert torch.allclose(weights[:, asset_ix], torch.ones(n_samples).to(dtype=dtype, device=device))
        assert isinstance(bm.hparams, dict) and bm.hparams

    def test_error(self):
        with pytest.raises(IndexError):
            Singleton(asset_ix=3)(torch.ones(2, 1, 3, 2))
