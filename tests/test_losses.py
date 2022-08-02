import operator

import pytest
import torch

from deepdow.losses import (
    Alpha,
    CumulativeReturn,
    LargestWeight,
    Loss,
    MaximumDrawdown,
    MeanReturns,
    RiskParity,
    Quantile,
    SharpeRatio,
    Softmax,
    SortinoRatio,
    SquaredWeights,
    StandardDeviation,
    TargetMeanReturn,
    TargetStandardDeviation,
    WorstReturn,
    DownsideRisk,
    log2simple,
    portfolio_returns,
    portfolio_cumulative_returns,
    simple2log,
)

ALL_LOSSES = [
    Alpha,
    CumulativeReturn,
    LargestWeight,
    MaximumDrawdown,
    MeanReturns,
    RiskParity,
    Quantile,
    SharpeRatio,
    Softmax,
    SortinoRatio,
    SquaredWeights,
    StandardDeviation,
    TargetMeanReturn,
    TargetStandardDeviation,
    WorstReturn,
    DownsideRisk,
]


class TestHelpers:
    """Collection of tests focused on helper methods."""

    def test_return_conversion(self):
        shape = (2, 3)
        x = torch.rand(shape)

        assert torch.allclose(log2simple(simple2log(x)), x, atol=1e-6)
        assert torch.allclose(simple2log(log2simple(x)), x, atol=1e-6)


class TestPortfolioReturns:
    @pytest.mark.parametrize("input_type", ["log", "simple"])
    @pytest.mark.parametrize("output_type", ["log", "simple"])
    def test_shape(self, Xy_dummy, input_type, output_type):
        _, y_dummy, _, _ = Xy_dummy
        y_dummy = y_dummy[:, 0, ...]
        n_samples, horizon, n_assets = y_dummy.shape

        weights = torch.randint(1, 10, size=(n_samples, n_assets)).to(
            device=y_dummy.device, dtype=y_dummy.dtype
        )
        weights = weights / weights.sum(-1, keepdim=True)

        prets = portfolio_returns(
            weights, y_dummy, input_type=input_type, output_type=output_type
        )

        assert prets.shape == (n_samples, horizon)

    def test_errors(self):
        n_samples = 3
        n_assets = 2
        horizon = 4

        weights = torch.ones((n_samples, n_assets))
        y = torch.ones((n_samples, horizon, n_assets))

        with pytest.raises(ValueError):
            portfolio_returns(weights, y, input_type="fake")

        with pytest.raises(ValueError):
            portfolio_returns(weights, y, output_type="fake")

    @pytest.mark.parametrize("input_type", ["simple", "log"])
    def test_sanity_check(self, input_type):
        initial_wealth = 50
        y_1_ = torch.tensor(
            [[0.01, 0.02], [-0.05, 0.04]]
        )  # assets move in a different way
        y_2_ = torch.tensor(
            [[0.01, 0.01], [-0.05, -0.05]]
        )  # assets move in the same way

        w_a_ = torch.tensor([0.2, 0.8])

        y_1 = y_1_[None, ...]
        y_2 = y_2_[None, ...]
        w_a = w_a_[None, ...]

        # No need to rebalance when returns evolve in the same way
        assert torch.allclose(
            portfolio_returns(w_a, y_2, rebalance=True, input_type=input_type),
            portfolio_returns(
                w_a, y_2, rebalance=False, input_type=input_type
            ),
        )

        # rebalancing necessary when returns evolve differently
        assert not torch.allclose(
            portfolio_returns(w_a, y_1, rebalance=True, input_type=input_type),
            portfolio_returns(
                w_a, y_1, rebalance=False, input_type=input_type
            ),
        )

        # manually computed returns

        if input_type == "simple":
            h_per_asset_1 = (
                torch.tensor(
                    [
                        [w_a_[0], w_a_[1]],
                        [
                            w_a_[0] * (1 + y_1_[0, 0]),
                            w_a_[1] * (1 + y_1_[0, 1]),
                        ],
                        [
                            w_a_[0] * (1 + y_1_[0, 0]) * (1 + y_1_[1, 0]),
                            w_a_[1] * (1 + y_1_[0, 1]) * (1 + y_1_[1, 1]),
                        ],
                    ]
                )
                * initial_wealth
            )
        else:
            h_per_asset_1 = (
                torch.tensor(
                    [
                        [w_a_[0], w_a_[1]],
                        [
                            w_a_[0] * torch.exp(y_1_[0, 0]),
                            w_a_[1] * torch.exp(y_1_[0, 1]),
                        ],
                        [
                            w_a_[0]
                            * torch.exp(y_1_[0, 0])
                            * torch.exp(y_1_[1, 0]),
                            w_a_[1]
                            * torch.exp(y_1_[0, 1])
                            * torch.exp(y_1_[1, 1]),
                        ],
                    ]
                )
                * initial_wealth
            )

        h_1 = h_per_asset_1.sum(1)

        correct_simple_returns = torch.tensor(
            [(h_1[1] / h_1[0]) - 1, (h_1[2] / h_1[1]) - 1]
        )
        correct_log_returns = torch.tensor(
            [torch.log(h_1[1] / h_1[0]), torch.log(h_1[2] / h_1[1])]
        )
        correct_simple_creturns = torch.tensor(
            [(h_1[1] / h_1[0]) - 1, (h_1[2] / h_1[0]) - 1]
        )
        correct_log_creturns = torch.tensor(
            [torch.log(h_1[1] / h_1[0]), torch.log(h_1[2] / h_1[0])]
        )

        assert torch.allclose(
            portfolio_returns(
                w_a,
                y_1,
                input_type=input_type,
                output_type="simple",
                rebalance=False,
            )[0],
            correct_simple_returns,
        )

        assert torch.allclose(
            portfolio_returns(
                w_a,
                y_1,
                input_type=input_type,
                output_type="log",
                rebalance=False,
            )[0],
            correct_log_returns,
        )

        assert torch.allclose(
            portfolio_cumulative_returns(
                w_a,
                y_1,
                input_type=input_type,
                output_type="simple",
                rebalance=False,
            )[0],
            correct_simple_creturns,
        )

        assert torch.allclose(
            portfolio_cumulative_returns(
                w_a,
                y_1,
                input_type=input_type,
                output_type="log",
                rebalance=False,
            )[0],
            correct_log_creturns,
        )

    @pytest.mark.parametrize("input_type", ["log", "simple"])
    @pytest.mark.parametrize("output_type", ["log", "simple"])
    @pytest.mark.parametrize("rebalance", [True, False])
    def test_sample_independence(self, input_type, output_type, rebalance):
        y_a = torch.tensor(
            [[0.01, 0.02], [-0.05, 0.04]]
        )  # assets move in a different way
        y_b = torch.tensor(
            [[0.01, 0.03], [-0.01, -0.02]]
        )  # assets move in the same way
        w_a = torch.tensor([0.2, 0.8])
        w_b = torch.tensor([0.45, 0.55])

        w_1 = torch.stack([w_a, w_b], dim=0)
        y_1 = torch.stack([y_a, y_b], dim=0)
        w_2 = torch.stack([w_b, w_a], dim=0)
        y_2 = torch.stack([y_b, y_a], dim=0)

        res_1 = portfolio_returns(
            w_1,
            y_1,
            input_type=input_type,
            output_type=output_type,
            rebalance=rebalance,
        )
        res_2 = portfolio_returns(
            w_2,
            y_2,
            input_type=input_type,
            output_type=output_type,
            rebalance=rebalance,
        )

        assert torch.allclose(res_1[0], res_2[1])
        assert torch.allclose(res_1[1], res_2[0])

    @pytest.mark.parametrize("input_type", ["log", "simple"])
    @pytest.mark.parametrize("output_type", ["log", "simple"])
    def test_shape_cumulative(self, Xy_dummy, input_type, output_type):
        _, y_dummy, _, _ = Xy_dummy

        y_dummy = y_dummy.mean(dim=1)
        n_samples, horizon, n_assets = y_dummy.shape

        weights = torch.randint(1, 10, size=(n_samples, n_assets)).to(
            device=y_dummy.device, dtype=y_dummy.dtype
        )

        pcrets = portfolio_cumulative_returns(
            weights, y_dummy, input_type=input_type, output_type=output_type
        )

        assert pcrets.shape == (n_samples, horizon)

    def test_errors_cumulative(self):
        n_samples = 3
        n_assets = 2
        horizon = 4

        weights = torch.ones((n_samples, n_assets))
        y = torch.ones((n_samples, horizon, n_assets))

        with pytest.raises(ValueError):
            portfolio_cumulative_returns(weights, y, input_type="fake")

        with pytest.raises(ValueError):
            portfolio_cumulative_returns(weights, y, output_type="fake")


class TestAllLosses:
    @pytest.mark.parametrize(
        "loss_class", ALL_LOSSES, ids=[x.__name__ for x in ALL_LOSSES]
    )
    def test_correct_output(self, loss_class, Xy_dummy):
        _, y_dummy, _, _ = Xy_dummy
        n_samples, n_channels, horizon, n_assets = y_dummy.shape

        weights = (torch.ones(n_samples, n_assets) / n_assets).to(
            device=y_dummy.device, dtype=y_dummy.dtype
        )

        loss_instance = loss_class()  # only defaults
        losses = loss_instance(weights, y_dummy)

        assert torch.is_tensor(losses)
        assert losses.shape == (n_samples,)
        assert losses.dtype == y_dummy.dtype
        assert losses.device == y_dummy.device

    @pytest.mark.parametrize(
        "loss_class_r",
        ALL_LOSSES + [3],
        ids=[x.__name__ for x in ALL_LOSSES] + ["constant"],
    )
    @pytest.mark.parametrize("op", ["add", "truediv", "mul", "pow"])
    def test_arithmetic(self, loss_class_r, op, Xy_dummy):
        _, y_dummy, _, _ = Xy_dummy
        n_samples, n_channels, horizon, n_assets = y_dummy.shape

        loss_class_l = SharpeRatio
        r_is_constant = isinstance(loss_class_r, int)

        weights = (torch.ones(n_samples, n_assets) / n_assets).to(
            device=y_dummy.device, dtype=y_dummy.dtype
        )

        loss_instance_l = loss_class_l()
        loss_instance_r = loss_class_r() if not r_is_constant else loss_class_r

        python_operator = getattr(operator, op)

        if op == "pow" and not r_is_constant:
            with pytest.raises(TypeError):
                python_operator(loss_instance_l, loss_instance_r)

            return

        else:
            mixed_loss = python_operator(loss_instance_l, loss_instance_r)

        true_tensor = python_operator(
            loss_instance_l(weights, y_dummy),
            loss_instance_r(weights, y_dummy)
            if not r_is_constant
            else loss_class_r,
        )

        sign = {"add": "+", "truediv": "/", "mul": "*", "pow": "**"}[op]

        mixed_tensor = mixed_loss(weights, y_dummy)

        assert torch.is_tensor(mixed_tensor)
        assert torch.allclose(mixed_tensor, true_tensor)
        assert mixed_tensor.shape == (n_samples,)
        assert mixed_tensor.dtype == y_dummy.dtype
        assert mixed_tensor.device == y_dummy.device
        assert sign in repr(mixed_loss)

    def test_parent_undefined_methods(self):
        with pytest.raises(NotImplementedError):
            Loss()(None, None)

        with pytest.raises(NotImplementedError):
            repr(Loss())

    def test_invalid_types_on_ops(self):
        with pytest.raises(TypeError):
            Loss() + "wrong"

        with pytest.raises(TypeError):
            "wrong" + Loss()

        with pytest.raises(TypeError):
            Loss() * "wrong"

        with pytest.raises(TypeError):
            "wrong" * Loss()

        with pytest.raises(TypeError):
            Loss() / "wrong"

        with pytest.raises(ZeroDivisionError):
            Loss() / 0

        with pytest.raises(TypeError):
            Loss() ** "wrong"

    @pytest.mark.parametrize(
        "loss_class", ALL_LOSSES, ids=[x.__name__ for x in ALL_LOSSES]
    )
    def test_repr_single(self, loss_class):
        n_samples, n_assets, n_channels = 3, 4, 2

        weights = torch.rand(n_samples, n_assets)
        weights /= weights.sum(dim=1).view(n_samples, 1)
        y = (torch.rand(n_samples, n_channels, 5, n_assets) - 1) / 100

        loss_instance_orig = loss_class()  # only defaults
        loss_instance_recreated = eval(repr(loss_instance_orig))

        losses_orig = loss_instance_orig(weights, y)
        losses_recreated = loss_instance_recreated(weights, y)

        assert torch.allclose(losses_orig, losses_recreated)

    @pytest.mark.parametrize(
        "loss_class_l", ALL_LOSSES, ids=[x.__name__ for x in ALL_LOSSES]
    )
    @pytest.mark.parametrize(
        "loss_class_r",
        ALL_LOSSES + [3],
        ids=[x.__name__ for x in ALL_LOSSES] + ["constant"],
    )
    @pytest.mark.parametrize("op", ["sum", "div", "mul", "pow"])
    def test_repr_arithmetic(self, loss_class_l, loss_class_r, op):
        n_samples, n_assets, n_channels = 3, 4, 2

        weights = torch.rand(n_samples, n_assets)
        weights /= weights.sum(dim=1).view(n_samples, 1)
        y = (torch.rand(n_samples, n_channels, 5, n_assets) - 1) / 100

        loss_instance_l = loss_class_l()
        loss_instance_r = (
            loss_class_r()
            if not isinstance(loss_class_r, int)
            else loss_class_r
        )

        if op == "sum":
            mixed = loss_instance_l + loss_instance_r

        elif op == "mul":
            mixed = loss_instance_l * loss_instance_r

        elif op == "div":
            mixed = loss_instance_l / loss_instance_r

        elif op == "pow":
            mixed = loss_instance_l**2

        else:
            raise ValueError("Unrecognized op")

        mixed_recreated = eval(repr(mixed))
        losses_orig = mixed(weights, y)
        losses_recreated = mixed_recreated(weights, y)

        assert torch.allclose(losses_orig, losses_recreated)


class TestAlpha:
    def test_manual(self, Xy_dummy):
        X, y, _, _ = Xy_dummy

        device, dtype = y.device, y.dtype
        n_samples, n_channels, horizon, n_assets = y.shape

        benchmark_weights = torch.zeros(n_assets, dtype=dtype, device=device)
        benchmark_weights[0] = 1
        weights = (
            benchmark_weights[None, :]
            .repeat(n_samples, 1)
            .to(device=device, dtype=dtype)
        )

        loss = Alpha(benchmark_weights=benchmark_weights)

        loss = loss(weights, y)

        assert torch.allclose(
            loss, torch.zeros(n_samples, dtype=dtype, device=device), atol=1e-3
        )


class TestMaximumDrawdown:
    def test_no_drawdowns(self):
        y_ = torch.tensor(
            [
                [0.01, 0.02],
                [0.02, 0.02],
                [0.01, 0.01],
                [0, 0],
            ]
        )
        y = y_[None, None, ...]

        w_ = torch.tensor([0.4, 0.6])
        w = w_[None, ...]

        loss_inst = MaximumDrawdown()
        loss = loss_inst(w, y)[0]

        assert loss == 0


class TestRiskParity:
    @staticmethod
    def stupid_compute(w, y):
        """Straightforward implementation with list comprehensions."""
        from deepdow.layers import CovarianceMatrix

        n_samples, n_assets = w.shape
        covar = CovarianceMatrix(sqrt=False)(y[:, 0, ...])  # returns_channel=0

        var = torch.cat(
            [
                (w[[i]] @ covar[i]) @ w[[i]].permute(1, 0)
                for i in range(n_samples)
            ],
            dim=0,
        )
        vol = torch.sqrt(var)

        lhs = vol / n_assets
        rhs = torch.cat(
            [
                (1 / vol[i]) * w[[i]] * (w[[i]] @ covar[i])
                for i in range(n_samples)
            ],
            dim=0,
        )

        res = torch.tensor(
            [((lhs[i] - rhs[i]) ** 2).sum() for i in range(n_samples)]
        )

        return res

    def test_correct_fpass(self):
        device, dtype = torch.device("cpu"), torch.float32
        n_samples, n_channels, horizon, n_assets = 2, 3, 10, 5

        # Generate weights and targets
        torch.manual_seed(2)
        y = torch.rand(
            (n_samples, n_channels, horizon, n_assets),
            dtype=dtype,
            device=device,
        )

        weights = torch.rand((n_samples, n_assets), dtype=dtype, device=device)

        weights /= weights.sum(dim=-1, keepdim=True)

        res_stupid = self.stupid_compute(weights, y)
        res_actual = RiskParity()(weights, y)

        assert torch.allclose(res_stupid, res_actual)
