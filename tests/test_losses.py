import pytest
import torch

from deepdow.losses import (log2simple, mean_returns, number_of_unused_assets, portfolio_cumulative_returns,
                            portfolio_returns, sharpe_ratio, simple2log, single_punish, sortino_ratio,
                            squared_weights, std, worst_return)


class TestHelpers:
    """Collection of tests focused on helper methods."""

    def test_return_conversion(self):
        shape = (2, 3)
        x = torch.rand(shape)

        assert torch.allclose(log2simple(simple2log(x)), x, atol=1e-6)
        assert torch.allclose(simple2log(log2simple(x)), x, atol=1e-6)


class TestPortfolioReturns:
    @pytest.mark.parametrize('input_type', ['log', 'simple'])
    @pytest.mark.parametrize('output_type', ['log', 'simple'])
    def test_shape(self, y_dummy, input_type, output_type):
        n_samples, horizon, n_assets = y_dummy.shape

        weights = torch.randint(1, 10, size=(n_samples, n_assets)).float()

        prets = portfolio_returns(weights, y_dummy, input_type=input_type, output_type=output_type)

        assert prets.shape == (n_samples, horizon)

    def test_errors(self):
        n_samples = 3
        n_assets = 2
        horizon = 4

        weights = torch.ones((n_samples, n_assets))
        y = torch.ones((n_samples, horizon, n_assets))

        with pytest.raises(ValueError):
            portfolio_returns(weights, y, input_type='fake')

        with pytest.raises(ValueError):
            portfolio_returns(weights, y, output_type='fake')


class TestCumulativePortfolioReturns:
    @pytest.mark.parametrize('input_type', ['log', 'simple'])
    @pytest.mark.parametrize('output_type', ['log', 'simple'])
    def test_shape(self, y_dummy, input_type, output_type):
        n_samples, horizon, n_assets = y_dummy.shape

        weights = torch.randint(1, 10, size=(n_samples, n_assets)).float()

        pcrets = portfolio_cumulative_returns(weights, y_dummy, input_type=input_type, output_type=output_type)

        assert pcrets.shape == (n_samples, horizon)

    def test_errors(self):
        n_samples = 3
        n_assets = 2
        horizon = 4

        weights = torch.ones((n_samples, n_assets))
        y = torch.ones((n_samples, horizon, n_assets))

        with pytest.raises(ValueError):
            portfolio_cumulative_returns(weights, y, input_type='fake')

        with pytest.raises(ValueError):
            portfolio_cumulative_returns(weights, y, output_type='fake')


class TestAllLosses:
    @pytest.mark.parametrize('loss', [mean_returns, sharpe_ratio, sortino_ratio, std, worst_return])
    def test_shapes(self, y_dummy, loss):
        n_samples, horizon, n_assets = y_dummy.shape

        weights = torch.ones((n_samples, n_assets)) * 1 / n_assets
        res = loss(weights, y_dummy)

        assert res.shape == (n_samples,)


class TestSharpeRatio:
    def test_shapes(self, y_dummy):
        n_samples, horizon, n_assets = y_dummy.shape

        weights = torch.ones((n_samples, n_assets)) * 1 / n_assets
        res = sharpe_ratio(weights, y_dummy)

        assert res.shape == (n_samples,)


class TestWeightOnlyLosses:

    @pytest.mark.parametrize('loss', [number_of_unused_assets, single_punish, squared_weights])
    def test_1overN_better(self, loss):
        n_assets = 10
        n_samples = 8

        winner = 3

        weights_1overN = torch.ones((n_samples, n_assets)) / n_assets
        weights_winner = torch.zeros((n_samples, n_assets))
        weights_winner[:, winner] = 1

        loss_1overN = loss(weights_1overN)
        loss_winner = loss(weights_winner)

        assert torch.all(loss_1overN < loss_winner)
