"""
=====================
Vector autoregression
=====================

This example demonstrates how one can validate :code:`deepdow` on synthetic data.
We choose to model our returns with the vector autoregression model (VAR).
This model links future returns to lagged returns with a linear
model. See [Lütkepohl2005]_ for more details. We use a stable VAR
process with 12 lags and 8 assets, that is

.. math::

    r_t = A_1 r_{t-1} + ... +  A_{12} r_{t-12}


For this specific task, we use the :code:`LinearNet` network. It is very similar to VAR since it tries to find a linear
model of all lagged variables. However, it also has purely deep learning components like dropout, batch
normalization and softmax allocator.

To put the performance of our network into context, we create a benchmark **VARTrue** that has access to the true
parameters of the VAR process. We create a simple investment rule of investing all resources into the asset with the
highest future returns. Additionally, we also consider other benchmarks

- equally weighted portfolio
- inverse volatility
- random allocation


References
----------
.. [Lütkepohl2005]
    Lütkepohl, Helmut. New introduction to multiple time series analysis. Springer Science & Business Media, 2005.


.. warning::

    Note that we are using the :code:`statsmodels` package to simulate the VAR process.

"""

import numpy as np
import torch

import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VARProcess, forecast

from deepdow.benchmarks import OneOverN, Benchmark, InverseVolatility, Random
from deepdow.callbacks import EarlyStoppingCallback
from deepdow.data import InRAMDataset, RigidDataLoader
from deepdow.losses import MeanReturns, SquaredWeights
from deepdow.nn import LinearNet
from deepdow.experiments import Run


class VARTrue(Benchmark):
    """Benchmark representing the ground truth return process.

    Parameters
    ----------
    process : statsmodels.tsa.vector_ar.var_model.VARProcess
        The ground truth VAR process that generates the returns.

    """

    def __init__(self, process):
        self.process = process

    def __call__(self, x):
        """Invest all money into the asset with the highest return over the horizon."""
        n_samples, n_channels, lookback, n_assets = x.shape

        assert n_channels == 1

        x_np = x.detach().numpy()  # (n_samples, n_channels, lookback, n_assets)
        weights_list = [forecast(x_np[i, 0], self.process.coefs, None, 1).argmax() for i in range(n_samples)]

        result = torch.zeros(n_samples, n_assets).to(x.dtype)

        for i, w_ix in enumerate(weights_list):
            result[i, w_ix] = 1

        return result


coefs = np.load('var_coefs.npy')  # (lookback, n_assets, n_assets) = (12, 8, 8)

# Parameters
lookback, _, n_assets = coefs.shape
gap, horizon = 0, 1
batch_size = 256

# Simulate returns
process = VARProcess(coefs, None, np.eye(n_assets) * 1e-5)
data = process.simulate_var(10000)
n_timesteps = len(data)

# Create features and targets
X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(data[i - lookback: i, :])
    y_list.append(data[i + gap: i + gap + horizon, :])

X = np.stack(X_list, axis=0)[:, None, ...]
y = np.stack(y_list, axis=0)[:, None, ...]

# Setup deepdow framework
dataset = InRAMDataset(X, y)

network = LinearNet(1, lookback, n_assets, p=0.5)
dataloader = RigidDataLoader(dataset,
                             indices=list(range(5000)),
                             batch_size=batch_size,
                             lookback=lookback)
val_dataloaders = {'train': dataloader,
                   'val': RigidDataLoader(dataset,
                                          indices=list(range(5020, 9800)),
                                          batch_size=batch_size,
                                          lookback=lookback)}

run = Run(network,
          100 * MeanReturns(),
          dataloader,
          val_dataloaders=val_dataloaders,
          metrics={'sqweights': SquaredWeights()},
          benchmarks={'1overN': OneOverN(),
                      'VAR': VARTrue(process),
                      'Random': Random(),
                      'InverseVol': InverseVolatility()},
          optimizer=torch.optim.Adam(network.parameters(), amsgrad=True),
          callbacks=[EarlyStoppingCallback('val', 'loss')]
          )

history = run.launch(40)

fig, ax = plt.subplots(1, 1)
ax.set_title('Validation loss')

per_epoch_results = history.metrics.groupby(['dataloader', 'metric', 'model', 'epoch'])['value'].mean()['val']['loss']
our = per_epoch_results['network']
our.plot(ax=ax, label='network')

ax.hlines(y=per_epoch_results['VAR'], xmin=0, xmax=len(our), color='red', label='VAR')
ax.hlines(y=per_epoch_results['1overN'], xmin=0, xmax=len(our), color='green', label='1overN')
ax.hlines(y=per_epoch_results['Random'], xmin=0, xmax=len(our), color='yellow', label='Random')
ax.hlines(y=per_epoch_results['InverseVol'], xmin=0, xmax=len(our), color='black', label='InverseVol')

plt.legend()
