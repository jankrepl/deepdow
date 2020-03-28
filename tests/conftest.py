import numpy as np
import pandas as pd
import pytest
import torch

from deepdow.data import InRAMDataset, RigidDataLoader

GPU_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(scope='session')
def dataset_dummy():
    n_samples = 200
    n_channels = 2
    lookback = 20
    horizon = 32
    n_assets = 30

    X = np.random.normal(size=(n_samples, n_channels, lookback, n_assets))
    y = np.random.normal(size=(n_samples, n_channels, horizon, n_assets))

    timestamps = pd.date_range(start='31/01/2000', periods=n_samples, freq='M')
    asset_names = ['asset_{}'.format(i) for i in range(n_assets)]

    return InRAMDataset(X, y, timestamps=timestamps, asset_names=asset_names)


@pytest.fixture()
def dataloader_dummy(dataset_dummy):
    batch_size = 32
    return RigidDataLoader(dataset_dummy,
                           batch_size=batch_size)


@pytest.fixture(params=[
    pytest.param((torch.float32, torch.device('cpu')), id='float32_cpu'),
    pytest.param((torch.float64, torch.device('cpu')), id='float64_cpu'),
    pytest.param((torch.float32, torch.device('cuda:0')),
                 id='float32_gpu',
                 marks=None if GPU_AVAILABLE else pytest.mark.skip),
    pytest.param((torch.float64, torch.device('cuda:0')),
                 id='float64_gpu',
                 marks=None if GPU_AVAILABLE else pytest.mark.skip),
])
def X_dummy(request):
    n_samples = 5
    n_channels = 2
    lookback = 3
    n_assets = 4

    torch.manual_seed(1)
    dtype, device = request.param

    return torch.rand((n_samples, n_channels, lookback, n_assets)).to(dtype=dtype, device=device)


@pytest.fixture()
def y_dummy():
    pass


@pytest.fixture()
def returns_dummy():
    """Historical returns."""
    n_timesteps = 20
    n_assets = 5
    random_state = 3

    freq = 'M'

    index = pd.date_range(start='1/1/2000', periods=n_timesteps, freq=freq)
    columns = [chr(65 + i) for i in range(n_assets)]

    np.random.seed(random_state)
    return_mean = np.random.uniform(-0.1, 0.1, size=n_assets)
    return_covmat = np.eye(n_assets) * 0.001

    returns = np.random.multivariate_normal(return_mean, return_covmat, size=n_timesteps - 1)

    return pd.DataFrame(returns, index=index[1:], columns=columns)


@pytest.fixture()
def prices_dummy(returns_dummy):
    """Historical prices."""

    index = pd.DatetimeIndex([pd.date_range(end=returns_dummy.index[0], periods=2, freq=returns_dummy.index.freq)[
                                  0]] + returns_dummy.index.to_list(), freq=returns_dummy.index.freq)
    columns = returns_dummy.columns
    n_assets = len(columns)

    starting_prices = np.random.randint(1, 100, size=n_assets)

    cumreturns = (1 + returns_dummy).cumprod(axis=1)

    prices = pd.DataFrame(index=index, columns=columns)
    prices.iloc[0, :] = starting_prices
    prices.iloc[1:, :] = prices.iloc[[0], :].values * cumreturns

    prices = prices.applymap(lambda x: max(0, x))

    return prices


@pytest.fixture(params=[1, 3], ids=['input_channels=1', "input_channels=3"])
def feature_tensor(request):
    """Standard tensor to be process by ConvTime layers."""
    n_samples = 10
    n_input_channels = request.param
    n_assets = 4
    lookback = 5

    return torch.ones((n_samples, n_input_channels, lookback, n_assets))


@pytest.fixture()
def feature_notime_tensor(feature_tensor):
    """Tensor to be processed by ConvOneByOne not containing time information."""
    return torch.mean(feature_tensor, dim=2)


@pytest.fixture()
def y_dummy():
    n_samples = 3
    horizon = 5
    n_assets = 6

    return (torch.rand((n_samples, horizon, n_assets)) - 0.5) / 10
