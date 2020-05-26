"""
=====================
Vector autoregression
=====================

This example demonstrates how one can validate deepdow on synthetic data.
We choose to model our returns with the vector autoregression model.
Specifically, it links future returns to lagged returns with a linear
model. See [Lütkepohl2005]_ for more details.

References
----------
.. [Lütkepohl2005]
    Lütkepohl, Helmut. New introduction to multiple time series analysis. Springer Science & Business Media, 2005.

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
    """Benchmark representing the ground truth return process."""

    def __init__(self, process):
        self.process = process

    def __call__(self, x):
        n_samples, n_channels, lookback, n_assets = x.shape

        assert n_channels == 1

        x_np = x.detach().numpy()  # (n_samples, n_channels, lookback, n_assets)
        weights_list = [forecast(x_np[i, 0], self.process.coefs, None, 1).argmax() for i in range(n_samples)]

        result = torch.zeros(n_samples, n_assets).to(x.dtype)

        for i, w_ix in enumerate(weights_list):
            result[i, w_ix] = 1

        return result


coefs = np.load('../examples/var_coefs.npy')  # (lookback=12, n_assets=8)

_, _, n_assets = coefs.shape
process = VARProcess(coefs, None, np.eye(n_assets) * 1e-5)

data = process.simulate_var(10000)

n_timesteps = len(data)
lookback, gap, horizon = 12, 0, 1

X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(data[i - lookback: i, :])
    y_list.append(data[i + gap: i + gap + horizon, :])

X = np.stack(X_list, axis=0)[:, None, ...]
y = np.stack(y_list, axis=0)[:, None, ...]

dataset = InRAMDataset(X, y)

lookback = 12
network = LinearNet(1, lookback, 8, p=0.5)
network.to(torch.float)
dataloader = RigidDataLoader(dataset,
                             indices=list(range(5000)),
                             batch_size=256,
                             lookback=lookback)
val_dataloaders = {'train': dataloader,
                   'val': RigidDataLoader(dataset,
                                          indices=list(range(5000, 9800)),
                                          batch_size=128,
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
network = per_epoch_results['network']
network.plot(ax=ax, label='network')

ax.hlines(y=per_epoch_results['VAR'], xmin=0, xmax=len(network), color='red', label='VAR')
ax.hlines(y=per_epoch_results['1overN'], xmin=0, xmax=len(network), color='green', label='1overN')
ax.hlines(y=per_epoch_results['Random'], xmin=0, xmax=len(network), color='yellow', label='Random')
ax.hlines(y=per_epoch_results['InverseVol'], xmin=0, xmax=len(network), color='black', label='InverseVol')

plt.legend()
