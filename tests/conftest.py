import numpy as np
import pandas as pd
import pytest
import torch

from deepdow.data import InRAMDataset, RigidDataLoader
from deepdow.benchmarks import OneOverN
from deepdow.experiments import Run
from deepdow.losses import MeanReturns
from deepdow.nn import DummyNet

GPU_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(scope='session', params=['B', 'M'], ids=['true_freq=B', 'true_freq=M'])
def raw_data(request):
    """Could represent prices, volumes,... Only positive values are allowed.

    Returns
    -------
    df : pd.DataFrame
        2D arrays where where rows represent different time points. Columns are a `pd.MultiIndex` with first
        level being the assets and the second level being the indicator.

    n_missing_entries : int
        Number of missing entries that were intentionally dropped from otherwise regular timeseries.

    true_freq : str
        True frequency of the underlying timeseries.
    """
    np.random.seed(1)

    n_assets = 4
    n_indicators = 6
    n_timestamps = 30
    n_missing_entries = 3
    true_freq = request.param

    missing_ixs = np.random.choice(list(range(1, n_timestamps - 1)), replace=False, size=n_missing_entries)

    index_full = pd.date_range('1/1/2000', periods=n_timestamps, freq=true_freq)
    index = pd.DatetimeIndex([x for ix, x in enumerate(index_full) if ix not in missing_ixs])  # freq=None

    columns = pd.MultiIndex.from_product([['asset_{}'.format(i) for i in range(n_assets)],
                                          ['indicator_{}'.format(i) for i in range(n_indicators)]],
                                         names=['assets', 'indicators'])

    df = pd.DataFrame(np.random.randint(low=1,
                                        high=1000,
                                        size=(n_timestamps - n_missing_entries, n_assets * n_indicators)) / 100,
                      index=index,
                      columns=columns)

    return df, n_missing_entries, true_freq


@pytest.fixture(scope='session')
def dataset_dummy():
    """Minimal instance of ``InRAMDataset``.

    Returns
    -------
    InRAMDataset

    """
    n_samples = 200
    n_channels = 2
    lookback = 9
    horizon = 10
    n_assets = 6

    X = np.random.normal(size=(n_samples, n_channels, lookback, n_assets)) / 100
    y = np.random.normal(size=(n_samples, n_channels, horizon, n_assets)) / 100

    timestamps = pd.date_range(start='31/01/2000', periods=n_samples, freq='M')
    asset_names = ['asset_{}'.format(i) for i in range(n_assets)]

    return InRAMDataset(X, y, timestamps=timestamps, asset_names=asset_names)


@pytest.fixture()
def dataloader_dummy(dataset_dummy):
    """Minimal instance of ``RigidDataLoader``.

    Parameters
    ----------
    dataset_dummy : InRAMDataset
        Underlying dataset.


    Returns
    -------

    """
    batch_size = 4
    return RigidDataLoader(dataset_dummy,
                           batch_size=batch_size)


@pytest.fixture(params=[
    pytest.param((torch.float32, torch.device('cpu')), id='float32_cpu'),
    pytest.param((torch.float64, torch.device('cpu')), id='float64_cpu'),
    pytest.param((torch.float32, torch.device('cuda:0')),
                 id='float32_gpu',
                 marks=[] if GPU_AVAILABLE else pytest.mark.skip),
    pytest.param((torch.float64, torch.device('cuda:0')),
                 id='float64_gpu',
                 marks=[] if GPU_AVAILABLE else pytest.mark.skip),
])
def dtype_device(request):
    dtype, device = request.param
    return dtype, device


@pytest.fixture()
def Xy_dummy(dtype_device, dataloader_dummy):
    dtype, device = dtype_device
    X, y, timestamps, asset_names = next(iter(dataloader_dummy))

    return X.to(dtype=dtype, device=device), y.to(dtype=dtype, device=device), timestamps, asset_names


@pytest.fixture()
def network_dummy(dataset_dummy):
    return DummyNet(n_channels=dataset_dummy.n_channels)


@pytest.fixture
def run_dummy(dataloader_dummy, network_dummy, Xy_dummy):
    """"""
    X_batch, y_batch, timestamps, asset_names = Xy_dummy

    device = X_batch.device
    dtype = X_batch.dtype

    return Run(network_dummy, MeanReturns(), dataloader_dummy,
               val_dataloaders={'val': dataloader_dummy},
               benchmarks={'bm': OneOverN()},
               device=device,
               dtype=dtype)


@pytest.fixture
def metadata_dummy(Xy_dummy, network_dummy):
    X_batch, y_batch, timestamps, asset_names = Xy_dummy

    device = X_batch.device
    dtype = X_batch.dtype
    _, _, horizon, _ = y_batch.shape

    network_dummy.to(device=device, dtype=dtype)

    return {'asset_names': asset_names,
            'batch': 1,
            'batch_loss': 1.4,
            'epoch': 1,
            'exception': ValueError,
            'locals': {'a': 2},
            'n_epochs': 2,
            'timestamps': timestamps,
            'weights': network_dummy(X_batch),
            'X_batch': X_batch,
            'y_batch': y_batch}
