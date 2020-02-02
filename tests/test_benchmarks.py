"""Collection of tests focused on `benchmarks` module."""
import pytest
import torch

from deepdow.benchmarks import Benchmark, OneOverN, Random, Singleton


class TestBenchmark:
    def test_errors(self):
        with pytest.raises(TypeError):
            Benchmark()

        class TempBenchmarkWrong(Benchmark):
            pass

        class TempBenchmarkCorrect(Benchmark):
            def __call__(self, X):
                return 'SOMETHING'

        with pytest.raises(TypeError):
            TempBenchmarkWrong()

        TempBenchmarkCorrect()('something')


class TestOneOverN:
    def test_basic(self):
        n_samples, n_channels, lookback, n_assets = (2, 1, 3, 4)
        X = torch.ones((n_samples, n_channels, lookback, n_assets))

        bm = OneOverN()
        bm.fit()  # no effect
        weights = bm(X)

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (n_samples, n_assets)
        assert torch.allclose(weights.sum(dim=1), torch.ones(n_samples))
        assert len(torch.unique(weights)) == 1


class TestRandom:
    def test_basic(self):
        n_samples, n_channels, lookback, n_assets = (2, 1, 3, 4)
        X = torch.ones((n_samples, n_channels, lookback, n_assets))

        bm = Random()
        bm.fit()  # no effect
        weights = bm(X)

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (n_samples, n_assets)
        assert torch.allclose(weights.sum(dim=1), torch.ones(n_samples))

        assert torch.all(weights >= 0) and torch.all(weights <= 1)


class TestSingleton:

    @pytest.mark.parametrize('asset_ix', [0, 3])
    def test_basic(self, asset_ix):
        n_samples, n_channels, lookback, n_assets = (2, 1, 3, 4)
        X = torch.ones((n_samples, n_channels, lookback, n_assets))

        bm = Singleton(asset_ix=asset_ix)
        weights = bm(X)

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (n_samples, n_assets)
        assert torch.allclose(weights.sum(dim=1), torch.ones(n_samples))

        assert torch.allclose(weights[:, asset_ix], torch.ones(n_samples))

    def test_error(self):
        with pytest.raises(IndexError):
            Singleton(asset_ix=3)(torch.ones(2, 1, 3, 2))
