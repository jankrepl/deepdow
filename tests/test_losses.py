import pytest
import torch

from deepdow.losses import (LargestWeight, Loss, MeanReturns, SharpeRatio, Softmax, SortinoRatio, SquaredWeights,
                            StandardDeviation, TargetMeanReturn, TargetStandardDeviation, WorstReturn, log2simple,
                            portfolio_returns, portfolio_cumulative_returns, simple2log)

ALL_LOSSES = [LargestWeight, MeanReturns, SharpeRatio, Softmax, SortinoRatio, SquaredWeights, StandardDeviation,
              TargetMeanReturn, TargetStandardDeviation, WorstReturn]


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
    def test_shape(self, Xy_dummy, input_type, output_type):
        _, y_dummy, _, _ = Xy_dummy
        y_dummy = y_dummy.mean(dim=1)
        n_samples, horizon, n_assets = y_dummy.shape

        weights = torch.randint(1, 10, size=(n_samples, n_assets)).to(device=y_dummy.device, dtype=y_dummy.dtype)

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
    def test_shape(self, Xy_dummy, input_type, output_type):
        _, y_dummy, _, _ = Xy_dummy

        y_dummy = y_dummy.mean(dim=1)
        n_samples, horizon, n_assets = y_dummy.shape

        weights = torch.randint(1, 10, size=(n_samples, n_assets)).to(device=y_dummy.device, dtype=y_dummy.dtype)

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
    @pytest.mark.parametrize('loss_class', ALL_LOSSES, ids=[x.__name__ for x in ALL_LOSSES])
    def test_correct_output(self, loss_class, Xy_dummy):
        _, y_dummy, _, _ = Xy_dummy
        n_samples, n_channels, horizon, n_assets = y_dummy.shape

        weights = (torch.ones(n_samples, n_assets) / n_assets).to(device=y_dummy.device, dtype=y_dummy.dtype)

        loss_instance = loss_class()  # only defaults
        losses = loss_instance(weights, y_dummy)

        assert torch.is_tensor(losses)
        assert losses.shape == (n_samples,)
        assert losses.dtype == y_dummy.dtype
        assert losses.device == y_dummy.device

    @pytest.mark.parametrize('loss_class_l', ALL_LOSSES, ids=[x.__name__ for x in ALL_LOSSES])
    @pytest.mark.parametrize('loss_class_r', ALL_LOSSES + [3], ids=[x.__name__ for x in ALL_LOSSES] + ['constant'])
    @pytest.mark.parametrize('op', ['sum', 'div', 'mul', 'pow'])
    def test_arithmetic(self, loss_class_l, loss_class_r, op, Xy_dummy):
        _, y_dummy, _, _ = Xy_dummy
        n_samples, n_channels, horizon, n_assets = y_dummy.shape

        loss_instance_l = loss_class_l()
        loss_instance_r = loss_class_r() if not isinstance(loss_class_r, int) else loss_class_r

        if op == 'sum':
            mixed = loss_instance_l + loss_instance_r
            sign = '+'

        elif op == 'mul':
            mixed = loss_instance_l * loss_instance_r
            sign = '*'

        elif op == 'div':
            mixed = loss_instance_l / loss_instance_r
            sign = '/'

        elif op == 'pow':
            mixed = loss_instance_l ** 2
            sign = '**'

        else:
            raise ValueError('Unrecognized op')

        weights = (torch.ones(n_samples, n_assets) / n_assets).to(device=y_dummy.device, dtype=y_dummy.dtype)

        losses = mixed(weights, y_dummy)

        assert torch.is_tensor(losses)
        assert losses.shape == (n_samples,)
        assert losses.dtype == y_dummy.dtype
        assert losses.device == y_dummy.device
        assert sign in repr(mixed)

    def test_parent_undefined_methods(self):
        with pytest.raises(NotImplementedError):
            Loss()(None, None)

        with pytest.raises(NotImplementedError):
            repr(Loss())

    def test_invalid_types_on_ops(self):
        with pytest.raises(TypeError):
            Loss() + 'wrong'

        with pytest.raises(TypeError):
            'wrong' + Loss()

        with pytest.raises(TypeError):
            Loss() * 'wrong'

        with pytest.raises(TypeError):
            'wrong' * Loss()

        with pytest.raises(TypeError):
            Loss() / 'wrong'

        with pytest.raises(ZeroDivisionError):
            Loss() / 0

        with pytest.raises(TypeError):
            Loss() ** 'wrong'

    @pytest.mark.parametrize('loss_class', ALL_LOSSES, ids=[x.__name__ for x in ALL_LOSSES])
    def test_repr_single(self, loss_class):
        n_samples, n_assets, n_channels = 3, 4, 2

        weights = torch.rand(n_samples, n_assets)
        weights /= weights.sum(dim=1).view(n_samples, 1)
        y = torch.rand(n_samples, n_channels, 5, n_assets) - 1

        loss_instance_orig = loss_class()  # only defaults
        loss_instance_recreated = eval(repr(loss_instance_orig))

        losses_orig = loss_instance_orig(weights, y)
        losses_recreated = loss_instance_recreated(weights, y)

        assert torch.allclose(losses_orig, losses_recreated)

    @pytest.mark.parametrize('loss_class_l', ALL_LOSSES, ids=[x.__name__ for x in ALL_LOSSES])
    @pytest.mark.parametrize('loss_class_r', ALL_LOSSES + [3], ids=[x.__name__ for x in ALL_LOSSES] + ['constant'])
    @pytest.mark.parametrize('op', ['sum', 'div', 'mul', 'pow'])
    def test_repr_arithmetic(self, loss_class_l, loss_class_r, op):
        n_samples, n_assets, n_channels = 3, 4, 2

        weights = torch.rand(n_samples, n_assets)
        weights /= weights.sum(dim=1).view(n_samples, 1)
        y = torch.rand(n_samples, n_channels, 5, n_assets) - 1

        loss_instance_l = loss_class_l()
        loss_instance_r = loss_class_r() if not isinstance(loss_class_r, int) else loss_class_r

        if op == 'sum':
            mixed = loss_instance_l + loss_instance_r

        elif op == 'mul':
            mixed = loss_instance_l * loss_instance_r

        elif op == 'div':
            mixed = loss_instance_l / loss_instance_r

        elif op == 'pow':
            mixed = loss_instance_l ** 2

        else:
            raise ValueError('Unrecognized op')

        mixed_recreated = eval(repr(mixed))
        losses_orig = mixed(weights, y)
        losses_recreated = mixed_recreated(weights, y)

        assert torch.allclose(losses_orig, losses_recreated)
