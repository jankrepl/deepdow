"""
===============
Getting started
===============

Welcome to :code:`deepdow`! This tutorial is going to demonstrate all the essential features.
Before you continue, make sure to check out :ref:`basics` to familiarize yourself with the core ideas
of :code:`deepdow`. This hands-on tutorial is divided into 4 sections

1. Dataset creation and loading
2. Network definition
3. Training
4. Evaluation and visualization of results
"""

# %%
# Preliminaries
# ^^^^^^^^^^^^^
# Let us start with importing all important dependencies.

from deepdow.benchmarks import Benchmark, OneOverN, Random
from deepdow.callbacks import EarlyStoppingCallback
from deepdow.data import InRAMDataset, RigidDataLoader, prepare_standard_scaler, Scale
from deepdow.data.synthetic import sin_single
from deepdow.experiments import Run
from deepdow.layers import SoftmaxAllocator
from deepdow.losses import MeanReturns, SharpeRatio, MaximumDrawdown
from deepdow.visualize import generate_metrics_table, generate_weights_table, plot_metrics, plot_weight_heatmap
import matplotlib.pyplot as plt
import numpy as np
import torch

# %%
# In order to be able to reproduce all results we set both the :code:`numpy` and :code:`torch` seed.

torch.manual_seed(4)
np.random.seed(5)

# %%
# Dataset creation and loading
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this example, we are going to be using a synthetic dataset. Asset returns are going to be
# sine functions where the frequency and phase are randomly selected for each asset. First of
# all let us set all the parameters relevant to data creation.
n_timesteps, n_assets = 1000, 20
lookback, gap, horizon = 40, 2, 20
n_samples = n_timesteps - lookback - horizon - gap + 1

# %%
# Additionally, we will use approximately 80% of the data for training and 20% for testing.
split_ix = int(n_samples * 0.8)
indices_train = list(range(split_ix))
indices_test = list(range(split_ix + lookback + horizon, n_samples))

print('Train range: {}:{}\nTest range: {}:{}'.format(indices_train[0], indices_train[-1],
                                                     indices_test[0], indices_test[-1]))

# %%
# Now we can generate the synthetic asset returns of with shape :code:`(n_timesteps, n_assets)`.
returns = np.array([sin_single(n_timesteps,
                               freq=1 / np.random.randint(3, lookback),
                               amplitude=0.05,
                               phase=np.random.randint(0, lookback)
                               ) for _ in range(n_assets)]).T

# %%
# We also add some noise.
returns += np.random.normal(scale=0.02, size=returns.shape)

# %%
# See below the first 100 timesteps of 2 assets.
plt.plot(returns[:100, [1, 2]])


# %%
# To obtain the feature matrix :code:`X` and the target :code:`y` we apply the rolling window
# strategy.
X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(returns[i - lookback: i, :])
    y_list.append(returns[i + gap: i + gap + horizon, :])

X = np.stack(X_list, axis=0)[:, None, ...]
y = np.stack(y_list, axis=0)[:, None, ...]

print('X: {}, y: {}'.format(X.shape, y.shape))

# %%
# As commonly done in every deep learning application, we want to scale our input features to
# be approximately centered around 0 and have a standard deviation of 1. In :code:`deepdow` we
# can achieve this with the :code:`prepare_standard_scaler` function that computes the mean
# and standard deviation of the input (for each channel). Additionally, we do not want to leak
# any information from our test set and therefore we only compute these statistics over the
# training set.
means, stds = prepare_standard_scaler(X, indices=indices_train)
print('mean: {}, std: {}'.format(means, stds))

# %%
# We can now construct the :code:`InRAMDataset`. By providing the optional :code:`transform` we
# make sure that when the samples are streamed they are always scaled based on our computed
# (training) statistics. See :ref:`inramdataset` for more details.

dataset = InRAMDataset(X, y, transform=Scale(means, stds))

# %%
# Using the :code:`dataset` we can now construct two dataloaders—one for training and the other one
# for testing. For more details see :ref:`dataloaders`.
dataloader_train = RigidDataLoader(dataset,
                                   indices=indices_train,
                                   batch_size=32)

dataloader_test = RigidDataLoader(dataset,
                                  indices=indices_test,
                                  batch_size=32)


# %%
# Network definition
# ^^^^^^^^^^^^^^^^^^
# Let us now write a custom network. See :ref:`writing_custom_networks`.
class GreatNet(torch.nn.Module, Benchmark):
    def __init__(self, n_assets, lookback, p=0.5):
        super().__init__()

        n_features = n_assets * lookback

        self.dropout_layer = torch.nn.Dropout(p=p)
        self.dense_layer = torch.nn.Linear(n_features, n_assets, bias=True)
        self.allocate_layer = SoftmaxAllocator(temperature=None)
        self.temperature = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, 1, lookback, n_assets).

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        n_samples, _, _, _ = x.shape
        x = x.view(n_samples, -1)  # flatten features
        x = self.dropout_layer(x)
        x = self.dense_layer(x)

        temperatures = torch.ones(n_samples).to(device=x.device, dtype=x.dtype) * self.temperature
        weights = self.allocate_layer(x, temperatures)

        return weights


# %%
# So what is this network doing? First of all, we make an assumption that assets and lookback will
# never change (the same shape and order at train and at inference time). This assumption
# is justified since we are using :code:`RigidDataLoader`.
# We can learn :code:`n_assets` linear models that have :code:`n_assets * lookback` features. In
# other words we have a dense layer that takes the flattened feature tensor :code:`x` and returns
# a vector of length :code:`n_assets`. Since elements of this vector can range from :math:`-\infty`
# to :math:`\infty` we turn it into an asset allocation via :code:`SoftmaxAllocator`.
# Additionally, we learn the :code:`temperature` from the data. This will enable us to learn the
# optimal trade-off between an equally weighted allocation (uniform distribution) and
# single asset portfolios.

# %%
network = GreatNet(n_assets, lookback)
print(network)

# %%
# In :code:`torch` networks are either in the **train** or **eval** mode. Since we are using
# dropout it is essential that we set the mode correctly based on what we are trying to do.
network = network.train()  # it is the default, however, just to make the distinction clear

# %%
# Training
# ^^^^^^^^
# It is now time to define our loss. Let's say we want to achieve multiple objectives at the same
# time. We want to minimize the drawdowns, maximize the mean returns and also maximize the Sharpe
# ratio. All of these losses are implemented in :code:`deepdow.losses`. To avoid confusion, they
# are always implemented in a way that **the lower the value of the loss the better**. To combine
# multiple objectives we can simply sum all of the individual losses. Similarly, if we want to
# assign more importance to one of them we can achieve this by multiplying by a constant. To learn
# more see :ref:`losses`.

loss = MaximumDrawdown() + 2 * MeanReturns() + SharpeRatio()

# %%
# Note that by default all the losses assume that we input logarithmic returns
# (:code:`input_type='log'`) and that they are in the 0th channel (:code:`returns_channel=0`).


# %%
# We now have all the ingredients ready for training of the neural network. :code:`deepdow` implements
# a simple wrapper :code:`Run` that implements the training loop and a minimal callback
# framework. For further information see :ref:`experiments`.

run = Run(network,
          loss,
          dataloader_train,
          val_dataloaders={'test': dataloader_test},
          optimizer=torch.optim.Adam(network.parameters(), amsgrad=True),
          callbacks=[EarlyStoppingCallback(metric_name='loss',
                                           dataloader_name='test',
                                           patience=15)])
# %%
# To run the training loop, we use the :code:`launch` where we specify the number of epochs.
history = run.launch(30)

# %%
# Evaluation and visualization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The :code:`history` object returned by :code:`launch` contains a lot of useful information related
# to training. Specifically, the property :code:`metrics` returns a comprehensive :code:`pd.DataFrame`.
# To display the average test loss per each epoch we can run following.

per_epoch_results = history.metrics.groupby(['dataloader', 'metric', 'model', 'epoch'])['value']

print(per_epoch_results.count())  # double check number of samples each epoch
print(per_epoch_results.mean())  # mean loss per epoch

# %%
per_epoch_results.mean()['test']['loss']['network'].plot()

# %%
# To get more insight into what our network predicts we can use the :code:`deepdow.visualize` module.
# Before we even start further evaluations, let us make sure the network is in eval model.
network = network.eval()

# %%
# To put the performance of our network in context, we also utilize benchmarks. :code:`deepdow`
# offers multiple benchmarks already. Additionally, one can provide custom simple benchmarks or
# some pre-trained networks.
benchmarks = {
    '1overN': OneOverN(),  # each asset has weight 1 / n_assets
    'random': Random(),  # random allocation that is however close 1OverN
    'network': network
}

# %%
# During training, the only mandatory metric/loss was the loss criterion that we tried to minimize.
# Naturally, one might be interested in many other metrics to evaluate the performance. See below
# an example.

metrics = {
    'MaxDD': MaximumDrawdown(),
    'Sharpe': SharpeRatio(),
    'MeanReturn': MeanReturns()
}

# %%
# Let us now use the above created objects. We first generate a table with all metrics over all
# samples and for all benchmarks. This is done via :code:`generate_metrics_table`.
metrics_table = generate_metrics_table(benchmarks,
                                       dataloader_test,
                                       metrics)

# %%
# And then we plot it with :code:`plot_metrics`.
plot_metrics(metrics_table)

# %%
# Each plot represents a different metric. The x-axis represents the timestamps in our
# test set. The different colors are capturing different models. How is the value of a metric
# computed? We assume that the investor predicts the portfolio at time x and buys it. He then
# holds it for :code:`horizon` timesteps. The actual metric is then computed over this time horizon.

# %%
# Finally, we are also interested in how the allocation/prediction looks like at each time step.
# We can use the :code:`generate_weights_table` function to create a :code:`pd.DataFrame`.
weight_table = generate_weights_table(network, dataloader_test)

# %%
# We then call the :code:`plot_weight_heatmap` to see a heatmap of weights.
plot_weight_heatmap(weight_table,
                    add_sum_column=True,
                    time_format=None,
                    time_skips=25)

# %%
# The rows represent different timesteps in our test set. The columns are all the assets in our
# universe. The values represent the weight in the portfolio. Additionally, we add a sum column
# to show that we are really generating valid allocations.
