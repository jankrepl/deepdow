import pytest
import torch

from deepdow.layers import (
    AverageCollapse,
    AttentionCollapse,
    ElementCollapse,
    ExponentialCollapse,
    MaxCollapse,
    SumCollapse,
)
from deepdow.layers import (
    AnalyticalMarkowitz,
    NCO,
    NumericalMarkowitz,
    NumericalRiskBudgeting,
    Resample,
    SoftmaxAllocator,
    SparsemaxAllocator,
    WeightNorm,
)
from deepdow.layers import (
    Cov2Corr,
    CovarianceMatrix,
    KMeans,
    MultiplyByConstant,
)
from deepdow.layers import Conv, RNN, Warp, Zoom

ALL_COLLAPSE = [
    AverageCollapse,
    AttentionCollapse,
    ElementCollapse,
    ExponentialCollapse,
    MaxCollapse,
    SumCollapse,
]
ALL_TRANSFORM = [Conv]


class TestAnalyticalMarkowitz:
    @pytest.mark.parametrize(
        "use_rets", [True, False], ids=["use_rets", "dont_use_rets"]
    )
    def test_eye(self, dtype_device, use_rets):
        dtype, device = dtype_device

        n_samples = 2
        n_assets = 4

        covmat = torch.stack(
            [
                torch.eye(n_assets, n_assets, dtype=dtype, device=device)
                for _ in range(n_samples)
            ],
            dim=0,
        )
        rets = (
            torch.ones(n_samples, n_assets, dtype=dtype, device=device)
            if use_rets
            else None
        )
        true_weights = (
            torch.ones(n_samples, n_assets, dtype=dtype, device=device)
            / n_assets
        )

        pred_weights = AnalyticalMarkowitz()(covmat, rets=rets)

        assert torch.allclose(pred_weights, true_weights)
        assert true_weights.device == pred_weights.device
        assert true_weights.dtype == pred_weights.dtype

    def test_diagonal(self, dtype_device):
        dtype, device = dtype_device

        covmat = torch.tensor(
            [[[1 / 2, 0, 0], [0, 1 / 3, 0], [0, 0, 1 / 5]]],
            dtype=dtype,
            device=device,
        )
        true_weights = torch.tensor(
            [[0.2, 0.3, 0.5]], dtype=dtype, device=device
        )

        pred_weights = AnalyticalMarkowitz()(covmat)

        assert torch.allclose(pred_weights, true_weights)
        assert true_weights.device == pred_weights.device
        assert true_weights.dtype == pred_weights.dtype


class TestCollapse:
    @pytest.mark.parametrize("layer", ALL_COLLAPSE)
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
            Conv(1, 2, method="FAKE")

    @pytest.mark.parametrize(
        "n_output_channels",
        [1, 5],
        ids=["n_output_channels_1", "n_output_channels_5"],
    )
    @pytest.mark.parametrize(
        "kernel_size", [1, 3], ids=["kernel_size_1", "kernel_size_3"]
    )
    @pytest.mark.parametrize("method", ["1D", "2D"])
    def test_default(self, Xy_dummy, method, kernel_size, n_output_channels):
        X, _, _, _ = Xy_dummy
        n_samples, n_channels, lookback, n_assets = X.shape

        if method == "1D":
            X = X.mean(dim=2)

        layer_inst = Conv(
            n_channels,
            n_output_channels,
            kernel_size=kernel_size,
            method=method,
        )

        layer_inst.to(device=X.device, dtype=X.dtype)

        res = layer_inst(X)

        assert torch.is_tensor(res)
        assert X.ndim == res.ndim
        assert X.device == res.device
        assert X.dtype == res.dtype

        assert X.shape[0] == res.shape[0]
        assert res.shape[1] == n_output_channels
        assert X.shape[2:] == res.shape[2:]


class TestCov2Corr:
    def test_eye(self, dtype_device):
        (
            dtype,
            device,
        ) = dtype_device  # we just use X to steal the device and dtype

        n_samples = 2
        n_assets = 3
        covmat = torch.stack(
            [
                torch.eye(n_assets, device=device, dtype=dtype)
                for _ in range(n_samples)
            ],
            dim=0,
        )
        corrmat = Cov2Corr()(covmat)

        assert torch.allclose(covmat, corrmat)
        assert covmat.device == corrmat.device
        assert covmat.dtype == corrmat.dtype

    def test_diagonal(self, Xy_dummy):
        X, _, _, _ = Xy_dummy

        device, dtype = (
            X.device,
            X.dtype,
        )  # we just use X to steal the device and dtype

        covmat = (
            torch.Tensor([[4, 0, 0], [0, 9, 0], [0, 0, 16]])
            .to(device=device, dtype=dtype)
            .view(1, 3, 3)
        )
        corrmat_true = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3)

        corrmat_pred = Cov2Corr()(covmat)

        assert torch.allclose(corrmat_pred, corrmat_true)
        assert corrmat_pred.device == corrmat_true.device
        assert corrmat_pred.dtype == corrmat_true.dtype


class TestCovarianceMatrix:
    def test_wrong_construction(self):
        with pytest.raises(ValueError):
            CovarianceMatrix(shrinkage_strategy="fake")

        with pytest.raises(ValueError):
            layer = CovarianceMatrix(shrinkage_coef=None)
            layer(torch.ones(1, 5, 2))

    @pytest.mark.parametrize(
        "shrinkage_strategy", ["diagonal", "identity", "scaled_identity", None]
    )
    @pytest.mark.parametrize("sqrt", [True, False], ids=["sqrt", "nosqrt"])
    def test_basic(self, Xy_dummy, sqrt, shrinkage_strategy):
        X, _, _, _ = Xy_dummy
        n_samples, n_channels, lookback, n_assets = X.shape

        X_ = X.mean(dim=1)

        if n_channels == 1:
            with pytest.raises(ZeroDivisionError):
                CovarianceMatrix(sqrt, shrinkage_strategy=shrinkage_strategy)(
                    X_
                )
        else:
            out = CovarianceMatrix(
                sqrt, shrinkage_strategy=shrinkage_strategy
            )(X_)

            assert out.shape == (n_samples, n_assets, n_assets)

    @pytest.mark.parametrize("sqrt", [True, False], ids=["sqrt", "nosqrt"])
    def test_n_parameters(self, sqrt):
        layer = CovarianceMatrix()

        n_parameters = sum(
            p.numel() for p in layer.parameters() if p.requires_grad
        )

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

    def test_shrinkage_coef(self):
        """Dynamic vs at construction."""
        n_samples = 3
        n_channels = 4
        n_assets = 5

        x = torch.rand((n_samples, n_channels, n_assets)) * 100

        layer_1 = CovarianceMatrix(
            sqrt=False, shrinkage_strategy="diagonal", shrinkage_coef=0.3
        )
        layer_2 = CovarianceMatrix(
            sqrt=False, shrinkage_strategy="diagonal", shrinkage_coef=None
        )

        assert torch.allclose(
            layer_1(x), layer_2(x, 0.3 * torch.ones(n_samples, dtype=x.dtype))
        )

        x_ = torch.rand((n_channels, n_assets)) * 100
        x_stacked = torch.stack([x_, x_, x_], dim=0)

        res_1 = layer_1(x_stacked)
        res_2 = layer_2(x_stacked, torch.tensor([0.2, 0.3, 0.6]))

        assert torch.allclose(res_1[0], res_1[1])
        assert torch.allclose(res_1[1], res_1[2])
        assert torch.allclose(res_1[2], res_1[0])

        assert not torch.allclose(res_2[0], res_2[1])
        assert not torch.allclose(res_2[1], res_2[2])
        assert not torch.allclose(res_2[2], res_2[0])


class TestKMeans:
    def test_errors(self):
        with pytest.raises(ValueError):
            KMeans(init="fake")

        with pytest.raises(ValueError):
            KMeans(n_clusters=4)(torch.ones(3, 2))

    def test_compute_distances(self, dtype_device):
        dtype, device = dtype_device

        x = torch.tensor([[1, 0], [0, 1], [2, 2]]).to(
            dtype=dtype, device=device
        )
        cluster_centers = torch.tensor([[0, 0], [1, 1]]).to(
            dtype=dtype, device=device
        )

        correct_result = torch.tensor([[1, 1], [1, 1], [8, 2]]).to(
            dtype=dtype, device=device
        )

        assert torch.allclose(
            KMeans.compute_distances(x, cluster_centers), correct_result
        )

    def test_manual_init(self):
        n_samples = 10
        n_features = 3
        n_clusters = 2

        kmeans_layer = KMeans(init="manual", n_clusters=n_clusters)

        x = torch.rand((n_samples, n_features))
        manual_init = torch.rand((n_clusters, n_features))
        wrong_init_1 = torch.rand((n_clusters + 1, n_features))
        wrong_init_2 = torch.rand((n_clusters, n_features + 1))

        with pytest.raises(TypeError):
            kmeans_layer.initialize(x, manual_init=None)

        with pytest.raises(ValueError):
            kmeans_layer.initialize(x, manual_init=wrong_init_1)

        with pytest.raises(ValueError):
            kmeans_layer.initialize(x, manual_init=wrong_init_2)

        assert torch.allclose(
            manual_init, kmeans_layer.initialize(x, manual_init=manual_init)
        )

    @pytest.mark.parametrize("init", ["random", "k-means++"])
    def test_init_deterministic(self, init, dtype_device):
        dtype, device = dtype_device

        random_state = 2
        x = torch.rand((20, 5), dtype=dtype, device=device)

        kmeans_layer = KMeans(n_clusters=3, init=init)

        torch.manual_seed(random_state)
        cluster_centers_1 = kmeans_layer.initialize(x)
        torch.manual_seed(random_state)
        cluster_centers_2 = kmeans_layer.initialize(x)

        assert torch.allclose(cluster_centers_1, cluster_centers_2)

    @pytest.mark.parametrize("init", ["random", "k-means++"])
    def test_all_deterministic(self, init, dtype_device):
        dtype, device = dtype_device

        random_state = 2
        kmeans_layer = KMeans(
            n_clusters=3,
            init=init,
            random_state=random_state,
            n_init=2,
            verbose=True,
        )

        x = torch.rand((20, 5), dtype=dtype, device=device)

        cluster_ixs_1, cluster_centers_1 = kmeans_layer(x)
        cluster_ixs_2, cluster_centers_2 = kmeans_layer(x)

        assert torch.allclose(cluster_centers_1, cluster_centers_2)
        assert torch.allclose(cluster_ixs_1, cluster_ixs_2)

    @pytest.mark.parametrize("random_state", [None, 1, 2])
    def test_dummy(self, dtype_device, random_state):
        """Create a dummy feature matrix.

        Notes
        -----
        Copied from scikit-learn tests.

        """

        dtype, device = dtype_device
        x = torch.tensor([[0, 0], [0.5, 0], [0.5, 1], [1, 1]]).to(
            dtype=dtype, device=device
        )
        manual_init = torch.tensor([[0, 0], [1, 1]])

        kmeans_layer = KMeans(
            n_clusters=2,
            init="manual",
            random_state=random_state,
            n_init=1,
            verbose=True,
        )

        cluster_ixs, cluster_centers = kmeans_layer(x, manual_init=manual_init)

        assert torch.allclose(
            cluster_ixs, torch.tensor([0, 0, 1, 1]).to(device=device)
        )
        assert torch.allclose(
            cluster_centers,
            torch.tensor([[0.25, 0], [0.75, 1]]).to(
                dtype=dtype, device=device
            ),
        )


class TestNumericalMarkowitz:
    def test_basic(self, Xy_dummy):
        X, _, _, _ = Xy_dummy
        device, dtype = X.device, X.dtype
        n_samples, n_channels, lookback, n_assets = X.shape

        popt = NumericalMarkowitz(n_assets)

        rets = X.mean(dim=(1, 2))

        covmat_sqrt__ = torch.rand((n_assets, n_assets)).to(
            device=X.device, dtype=X.dtype
        )
        covmat_sqrt_ = covmat_sqrt__ @ covmat_sqrt__
        covmat_sqrt_.add_(torch.eye(n_assets, dtype=dtype, device=device))

        covmat_sqrt = torch.stack(n_samples * [covmat_sqrt_])

        gamma = (torch.rand(n_samples) * 5 + 0.1).to(
            device=X.device, dtype=X.dtype
        )
        alpha = torch.ones(n_samples).to(device=X.device, dtype=X.dtype)

        weights = popt(rets, covmat_sqrt, gamma, alpha)

        assert weights.shape == (n_samples, n_assets)
        assert torch.allclose(
            weights.sum(dim=-1),
            torch.ones(n_samples, device=device, dtype=dtype),
        )
        assert weights.dtype == X.dtype
        assert weights.device == X.device


class TestMultiplyByConstant:
    def test_error(self):
        with pytest.raises(ValueError):
            MultiplyByConstant(dim_ix=1, dim_size=2)(torch.ones((2, 3)))

    @pytest.mark.parametrize("dim_ix", [1, 2, 3])
    def test_basic(self, Xy_dummy, dim_ix):
        X, _, _, _ = Xy_dummy

        layer_inst = MultiplyByConstant(
            dim_ix=dim_ix, dim_size=X.shape[dim_ix]
        )

        layer_inst.to(device=X.device, dtype=X.dtype)

        res = layer_inst(X)

        assert torch.is_tensor(res)
        assert X.device == res.device
        assert X.dtype == res.dtype
        assert res.shape == X.shape


class TestNCO:
    @pytest.mark.parametrize(
        "use_rets", [True, False], ids=["use_rets", "dont_use_rets"]
    )
    @pytest.mark.parametrize(
        "edge_case", ["single_cluster", "n_clusters=n_samples"]
    )
    def test_edge_case(self, dtype_device, use_rets, edge_case):
        dtype, device = dtype_device

        n_samples = 2
        n_assets = 4
        n_clusters = 1 if edge_case == "single_cluster" else n_assets

        single_ = torch.rand(n_assets, n_assets, dtype=dtype, device=device)
        single = single_ @ single_.t()
        covmat = torch.stack([single for _ in range(n_samples)], dim=0)
        rets = (
            torch.rand(n_samples, n_assets, dtype=dtype, device=device)
            if use_rets
            else None
        )

        true_weights = AnalyticalMarkowitz()(covmat, rets=rets)
        pred_weights = NCO(n_clusters=n_clusters)(covmat, rets=rets)

        assert torch.allclose(pred_weights, true_weights, atol=1e-3)
        assert true_weights.device == pred_weights.device
        assert true_weights.dtype == pred_weights.dtype

    @pytest.mark.parametrize(
        "use_rets", [True, False], ids=["use_rets", "dont_use_rets"]
    )
    def test_reproducible(self, dtype_device, use_rets):
        dtype, device = dtype_device

        n_samples = 2
        n_assets = 6
        n_clusters = 3

        single_ = torch.rand(n_assets, n_assets, dtype=dtype, device=device)
        single = single_ @ single_.t()
        covmat = torch.stack([single for _ in range(n_samples)], dim=0)
        rets = (
            torch.rand(n_samples, n_assets, dtype=dtype, device=device)
            if use_rets
            else None
        )

        layer = NCO(n_clusters=n_clusters, random_state=2)
        weights_1 = layer(covmat, rets=rets)
        weights_2 = layer(covmat, rets=rets)

        assert torch.allclose(weights_1, weights_2)
        assert weights_1.device == weights_2.device
        assert weights_1.dtype == weights_2.dtype


class TestResample:
    def test_error(self):
        with pytest.raises(TypeError):
            Resample("wrong_type")

    @pytest.mark.parametrize(
        "random_state", [1, None], ids=["random", "not_random"]
    )
    @pytest.mark.parametrize(
        "allocator_class", [AnalyticalMarkowitz, NCO, NumericalMarkowitz]
    )
    def test_basic(self, dtype_device, allocator_class, random_state):
        dtype, device = dtype_device

        n_samples = 2
        n_assets = 3

        single_ = torch.rand(n_assets, n_assets, dtype=dtype, device=device)
        single = single_ @ single_.t()
        covmat = torch.stack([single for _ in range(n_samples)], dim=0)
        rets = torch.rand(
            n_samples, n_assets, dtype=dtype, device=device, requires_grad=True
        )

        if allocator_class.__name__ == "AnalyticalMarkowitz":
            allocator = allocator_class()
            kwargs = {}
        elif allocator_class.__name__ == "NCO":
            allocator = allocator_class(n_clusters=2)
            kwargs = {}

        elif allocator_class.__name__ == "NumericalMarkowitz":
            allocator = allocator_class(n_assets=n_assets)
            kwargs = {
                "gamma": torch.ones(n_samples, dtype=dtype, device=device),
                "alpha": torch.ones(n_samples, dtype=dtype, device=device),
            }

        resample_layer = Resample(
            allocator, n_portfolios=2, sqrt=False, random_state=random_state
        )

        weights_1 = resample_layer(covmat, rets=rets, **kwargs)
        weights_2 = resample_layer(covmat, rets=rets, **kwargs)

        assert weights_1.shape == (n_samples, n_assets)
        assert weights_1.device == device
        assert weights_1.dtype == dtype

        if random_state is None:
            assert not torch.allclose(weights_1, weights_2)
        else:
            assert torch.allclose(weights_1, weights_2)

        # Make sure one can run backward pass (just sum the weights to get a scalar)
        some_loss = weights_1.sum()

        assert rets.grad is None

        some_loss.backward()

        assert rets.grad is not None
        assert single.grad is None


class TestRNN:
    @pytest.mark.parametrize(
        "bidirectional", [True, False], ids=["bidir", "onedir"]
    )
    @pytest.mark.parametrize("cell_type", ["LSTM", "RNN"])
    @pytest.mark.parametrize("hidden_size", [4, 6])
    @pytest.mark.parametrize("n_layers", [1, 2])
    def test_basic(
        self, Xy_dummy, hidden_size, bidirectional, cell_type, n_layers
    ):
        X, _, _, _ = Xy_dummy
        n_samples, n_channels, lookback, n_assets = X.shape

        layer_inst = RNN(
            n_channels,
            hidden_size,
            n_layers=n_layers,
            bidirectional=bidirectional,
            cell_type=cell_type,
        )

        layer_inst.to(device=X.device, dtype=X.dtype)

        res = layer_inst(X)

        assert torch.is_tensor(res)
        assert X.ndim == res.ndim
        assert X.device == res.device
        assert X.dtype == res.dtype

        assert X.shape[0] == res.shape[0]
        assert res.shape[1] == hidden_size
        assert X.shape[2:] == res.shape[2:]

    @pytest.mark.parametrize(
        "bidirectional", [True, False], ids=["bidir", "onedir"]
    )
    @pytest.mark.parametrize("cell_type", ["LSTM", "RNN"])
    @pytest.mark.parametrize("hidden_size", [4, 6])
    @pytest.mark.parametrize("n_channels", [1, 4])
    def test_n_parameters(
        self, n_channels, hidden_size, cell_type, bidirectional
    ):
        layer = RNN(
            n_channels,
            hidden_size,
            bidirectional=bidirectional,
            cell_type=cell_type,
        )

        n_parameters = sum(
            p.numel() for p in layer.parameters() if p.requires_grad
        )
        n_dir = 1 + int(bidirectional)
        hidden_size_a = int(hidden_size // n_dir)

        if cell_type == "RNN":
            assert n_parameters == n_dir * (
                (n_channels * hidden_size_a)
                + (hidden_size_a * hidden_size_a)
                + 2 * hidden_size_a
            )

        else:
            assert n_parameters == n_dir * 4 * (
                (n_channels * hidden_size_a)
                + (hidden_size_a * hidden_size_a)
                + 2 * hidden_size_a
            )

    def test_error(self):

        with pytest.raises(ValueError):
            RNN(2, 4, cell_type="FAKE")

        with pytest.raises(ValueError):
            RNN(3, 3, cell_type="LSTM", bidirectional=True)


class TestSoftmax:
    def test_errors(self):
        with pytest.raises(ValueError):
            SoftmaxAllocator(formulation="wrong")

        with pytest.raises(ValueError):
            SoftmaxAllocator(formulation="variational", n_assets=None)

        with pytest.raises(ValueError):
            SoftmaxAllocator(formulation="analytical", max_weight=0.3)

        with pytest.raises(ValueError):
            SoftmaxAllocator(
                formulation="variational", max_weight=0.09, n_assets=10
            )

    @pytest.mark.parametrize("formulation", ["analytical", "variational"])
    def test_basic(self, Xy_dummy, formulation):
        eps = 1e-5
        X, _, _, _ = Xy_dummy
        dtype, device = X.dtype, X.device
        n_samples, n_channels, lookback, n_assets = X.shape

        rets = X.mean(dim=(1, 2))

        with pytest.raises(ValueError):
            SoftmaxAllocator(
                temperature=None, formulation=formulation, n_assets=n_assets
            )(rets, temperature=None)

        weights = SoftmaxAllocator(
            temperature=2, formulation=formulation, n_assets=n_assets
        )(rets)

        assert torch.allclose(
            weights,
            SoftmaxAllocator(
                temperature=None, formulation=formulation, n_assets=n_assets
            )(rets, 2 * torch.ones(n_samples, dtype=dtype, device=device)),
        )
        assert weights.shape == (n_samples, n_assets)
        assert weights.dtype == X.dtype
        assert weights.device == X.device
        assert torch.all(-eps <= weights) and torch.all(weights <= 1 + eps)
        assert torch.allclose(
            weights.sum(dim=1),
            torch.ones(n_samples).to(dtype=dtype, device=device),
            atol=eps,
        )

    def test_equality_formulations(self, dtype_device):
        dtype, device = dtype_device
        n_samples, n_assets = 3, 6

        layer_analytical = SoftmaxAllocator(formulation="analytical")
        layer_variational = SoftmaxAllocator(
            formulation="variational", n_assets=n_assets
        )

        torch.manual_seed(2)
        rets = torch.randint(10, size=(n_samples, n_assets)) + torch.rand(
            size=(n_samples, n_assets)
        )
        rets = rets.to(device=device, dtype=dtype)

        weights_analytical = layer_analytical(rets)
        weights_variational = layer_variational(rets)

        assert weights_analytical.shape == weights_variational.shape
        assert weights_analytical.dtype == weights_variational.dtype
        assert weights_analytical.device == weights_variational.device

        assert torch.allclose(
            weights_analytical, weights_variational, atol=1e-4
        )

    @pytest.mark.parametrize("formulation", ["analytical", "variational"])
    def test_uniform(self, formulation):
        rets = torch.ones(2, 5)
        weights = SoftmaxAllocator(
            formulation=formulation, n_assets=5, temperature=1
        )(rets)

        assert torch.allclose(weights, rets / 5, atol=1e-4)

    @pytest.mark.parametrize("max_weight", [0.2, 0.25, 0.34])
    def test_contstrained(self, max_weight):
        rets = torch.tensor([[1.7909, -2, -0.6818, -0.4972, 0.0333]])

        w_const = SoftmaxAllocator(
            n_assets=5,
            temperature=1,
            max_weight=max_weight,
            formulation="variational",
        )(rets)
        w_unconst = SoftmaxAllocator(
            n_assets=5, temperature=1, max_weight=1, formulation="variational"
        )(rets)

        assert not torch.allclose(w_const, w_unconst)
        assert w_const.max().item() == pytest.approx(max_weight, abs=1e-4)


class TestSparsemax:
    def test_errors(self):
        with pytest.raises(ValueError):
            SparsemaxAllocator(n_assets=2, max_weight=0.3)

    def test_basic(self, Xy_dummy):
        eps = 1e-5
        X, _, _, _ = Xy_dummy
        dtype, device = X.dtype, X.device
        n_samples, n_channels, lookback, n_assets = X.shape

        rets = X.mean(dim=(1, 2))

        with pytest.raises(ValueError):
            SparsemaxAllocator(n_assets, temperature=None)(
                rets, temperature=None
            )

        weights = SparsemaxAllocator(n_assets, temperature=2)(rets)

        assert torch.allclose(
            weights,
            SparsemaxAllocator(n_assets, temperature=None)(
                rets, 2 * torch.ones(n_samples, dtype=dtype, device=device)
            ),
        )
        assert weights.shape == (n_samples, n_assets)
        assert weights.dtype == X.dtype
        assert weights.device == X.device
        assert torch.all(-eps <= weights) and torch.all(weights <= 1 + eps)
        assert torch.allclose(
            weights.sum(dim=1),
            torch.ones(n_samples).to(dtype=dtype, device=device),
            atol=eps,
        )

    def test_uniform(self):
        rets = torch.ones(2, 5)
        weights = SparsemaxAllocator(5, temperature=1)(rets)

        assert torch.allclose(weights, rets / 5)

    def test_known(self):
        rets = torch.tensor(
            [
                [1.7909, 0.3637, -0.6818, -0.4972, 0.0333],
                [0.6655, -0.9960, 1.1463, 1.9849, -0.1662],
            ]
        )

        true_weights = torch.tensor(
            [
                [1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0807, 0.9193, 0.0000],
            ]
        )

        assert torch.allclose(
            SparsemaxAllocator(5, temperature=1)(rets), true_weights, atol=1e-3
        )

    @pytest.mark.parametrize("max_weight", [0.2, 0.25, 0.34])
    def test_contstrained(self, max_weight):
        rets = torch.tensor([[1.7909, -2, -0.6818, -0.4972, 0.0333]])

        w_const = SparsemaxAllocator(5, temperature=1, max_weight=max_weight)(
            rets
        )
        w_unconst = SparsemaxAllocator(5, temperature=1)(rets)

        assert not torch.allclose(w_const, w_unconst)
        assert w_const.max().item() == pytest.approx(max_weight, abs=1e-3)


class TestWarp:
    def test_error(self):
        with pytest.raises(ValueError):
            Warp()(
                torch.rand(1, 2, 3, 4),
                torch.ones(
                    4,
                ),
            )

    @pytest.mark.parametrize("mode", ["nearest", "bilinear"])
    @pytest.mark.parametrize("padding_mode", ["zeros", "reflection", "border"])
    def test_basic(self, Xy_dummy, mode, padding_mode):
        X, _, _, _ = Xy_dummy
        dtype, device = X.dtype, X.device
        n_samples, _, lookback, n_assets = X.shape
        layer = Warp(mode=mode, padding_mode=padding_mode)

        tform_ = torch.rand(n_samples, lookback, dtype=dtype, device=device)
        tform_cumsum = tform_.cumsum(dim=-1)
        tform = 2 * (
            tform_cumsum / tform_cumsum.max(dim=1, keepdim=True)[0] - 0.5
        )
        x_warped = layer(X, tform)

        assert torch.is_tensor(x_warped)
        assert x_warped.shape == X.shape
        assert x_warped.dtype == X.dtype
        assert x_warped.device == X.device

    @pytest.mark.parametrize("mode", ["nearest", "bilinear"])
    @pytest.mark.parametrize("padding_mode", ["zeros", "reflection", "border"])
    def test_no_change(self, mode, padding_mode):
        # scale=1
        n_samples, n_channels, lookback, n_assets = 2, 3, 4, 5
        X = torch.rand(n_samples, n_channels, lookback, n_assets)
        tform = torch.stack(
            n_samples * [torch.linspace(-1, end=1, steps=lookback)], dim=0
        )

        layer = Warp(mode=mode, padding_mode=padding_mode)

        x_warped = layer(X, tform)

        assert torch.allclose(x_warped, X, atol=1e-5)

    def test_asset_specific(self):
        n_samples, n_channels, lookback, n_assets = 2, 3, 4, 5
        X = torch.rand(n_samples, n_channels, lookback, n_assets)
        tform = torch.randn(n_samples, lookback, n_assets, dtype=X.dtype)

        layer = Warp()

        x_warped = layer(X, tform)

        assert X.shape == x_warped.shape

    def test_n_parameters(self):
        n_parameters = sum(
            p.numel() for p in Warp().parameters() if p.requires_grad
        )

        assert n_parameters == 0


class TestWeightNorm:
    def test_basic(self, Xy_dummy):
        eps = 1e-5
        X, _, _, _ = Xy_dummy
        dtype, device = X.dtype, X.device
        n_samples, _, _, n_assets = X.shape
        layer = WeightNorm(n_assets=n_assets)
        layer.to(dtype=dtype, device=device)

        weights = layer(X)

        assert torch.is_tensor(weights)
        assert weights.shape == (n_samples, n_assets)
        assert torch.allclose(
            weights.sum(dim=1),
            torch.ones(n_samples).to(dtype=dtype, device=device),
            atol=eps,
        )

    @pytest.mark.parametrize("n_assets", [2, 5, 10])
    def test_n_parameters(self, n_assets):
        n_parameters = sum(
            p.numel()
            for p in WeightNorm(n_assets).parameters()
            if p.requires_grad
        )

        assert n_parameters == n_assets


class TestZoom:
    @pytest.mark.parametrize("mode", ["nearest", "bilinear"])
    @pytest.mark.parametrize("padding_mode", ["zeros", "reflection", "border"])
    def test_basic(self, Xy_dummy, mode, padding_mode):
        X, _, _, _ = Xy_dummy
        dtype, device = X.dtype, X.device
        n_samples, _, _, n_assets = X.shape
        layer = Zoom(mode=mode, padding_mode=padding_mode)

        scale = torch.rand(n_samples, dtype=dtype, device=device)
        x_zoomed = layer(X, scale)

        assert torch.is_tensor(x_zoomed)
        assert x_zoomed.shape == X.shape
        assert x_zoomed.dtype == X.dtype
        assert x_zoomed.device == X.device

    @pytest.mark.parametrize("mode", ["nearest", "bilinear"])
    @pytest.mark.parametrize("padding_mode", ["zeros", "reflection", "border"])
    def test_no_change(self, mode, padding_mode):
        # scale=1
        n_samples, n_channels, lookback, n_assets = 2, 3, 4, 5
        X = torch.rand(n_samples, n_channels, lookback, n_assets)
        scale = torch.ones(n_samples, dtype=X.dtype)

        layer = Zoom(mode=mode, padding_mode=padding_mode)

        x_zoomed = layer(X, scale)

        assert torch.allclose(x_zoomed, X)

    def test_n_parameters(self):
        n_parameters = sum(
            p.numel() for p in Zoom().parameters() if p.requires_grad
        )

        assert n_parameters == 0

    def test_equality_with_warp(self):
        n_samples, n_channels, lookback, n_assets = 2, 3, 4, 5
        X = torch.rand(n_samples, n_channels, lookback, n_assets)
        scale = torch.ones(n_samples, dtype=X.dtype) * 0.5
        tform = torch.stack(
            n_samples * [torch.linspace(0, end=1, steps=lookback)], dim=0
        )

        layer_zoom = Zoom()
        layer_warp = Warp()

        x_zoomed = layer_zoom(X, scale)
        x_warped = layer_warp(X, tform)

        assert torch.allclose(x_zoomed, x_warped)


class TestNumericalRiskBudgeting:
    def test_basic(self, Xy_dummy):
        X, _, _, _ = Xy_dummy
        device, dtype = X.device, X.dtype
        n_samples, n_channels, lookback, n_assets = X.shape

        popt = NumericalRiskBudgeting(n_assets)

        covmat_sqrt__ = torch.rand((n_assets, n_assets)).to(
            device=X.device, dtype=X.dtype
        )
        covmat_sqrt_ = covmat_sqrt__ @ covmat_sqrt__
        covmat_sqrt_.add_(torch.eye(n_assets, dtype=dtype, device=device))

        covmat_sqrt = torch.stack(n_samples * [covmat_sqrt_])

        budgets = torch.rand((n_samples, n_assets), dtype=dtype, device=device)
        budgets /= budgets.sum(dim=-1, keepdim=True)

        weights = popt(covmat_sqrt, budgets)

        assert weights.shape == (n_samples, n_assets)
        assert torch.allclose(
            weights.sum(dim=-1),
            torch.ones(n_samples, device=device, dtype=dtype),
        )
        assert weights.dtype == X.dtype
        assert weights.device == X.device
