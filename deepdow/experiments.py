"""Running experiments."""

import datetime
import time

from diffcp import SolverError
import numpy as np
import pandas as pd
import torch

from .benchmarks import Benchmark
from .callbacks import BenchmarkCallback, EarlyStoppingException, ProgressBarCallback, ValidationCallback
from .data import FlexibleDataLoader, RigidDataLoader
from .losses import Loss


class History:
    """A shared information database for the training process.

    Attributes
    ----------
    database : dict
        Keys are different epochs and values are ``pd.DataFrame`` with detailed results.

    """

    def __init__(self):
        self.database = {}  # dict where keys are epochs and values are lists

    @property
    def metrics(self):
        """Concatenate metrics over all epochs.

        Returns
        -------
        df : pd.DataFrame
            Each row represents a unique logging sample. Columns are `batch`, `current_time`, `dataloader`, `epoch`,
            `lookback`, `metric`, `model`, `timestamp`, `value`.
        """
        master_list = []  # over all epochs
        for l in self.database.values():
            master_list.extend(l)

        return pd.DataFrame(master_list)

    def metrics_per_epoch(self, epoch):
        """Results over a specified epoch.

        Returns
        -------
        df : pd.DataFrame
            Each row represents a unique logging sample. Columns are `batch`, `current_time`, `dataloader`, `epoch`,
            `lookback`, `metric`, `model`, `timestamp`, `value`.
        """
        return pd.DataFrame(self.database[epoch])

    def add_entry(self, model=None, metric=None, batch=None, epoch=None, dataloader=None,
                  lookback=None, timestamp=None, value=np.nan):
        """Add entry to the internal database."""
        if epoch not in self.database:
            self.database[epoch] = []

        self.database[epoch].append({'model': model,
                                     'metric': metric,
                                     'value': value,
                                     'batch': batch,
                                     'epoch': epoch,
                                     'dataloader': dataloader,
                                     'lookback': lookback,
                                     'timestamp': timestamp,
                                     'current_time': datetime.datetime.now()})

    def pretty_print(self, epoch=None):
        """Print nicely the internal database.

        Parameters
        ----------
        epoch : int or None
            If epoch given, then a results only over this epoch. If epoch is `None` then print results over all epochs.

        """
        if epoch is None:
            df = self.metrics

        else:
            df = self.metrics_per_epoch(epoch)
        pd.options.display.float_format = '{:,.3f}'.format
        print(df.groupby(['model', 'metric', 'epoch', 'dataloader'])['value'].mean().to_string())


class Run:
    """Represents one experiment.

    Parameters
    ----------
    network : nn.Module and Benchmark
        Network.

    loss : callable
        A callable computing per sample loss. Specifically it accepts `weights`, `y` and returns ``torch.Tensor``
         of shape `(n_samples,)`.

    train_dataloader : FlexibleDataLoader or RigidDataLoader
        DataLoader used for training.

    val_dataloaders  : None or dict
        If None no validation is performed. If ``dict`` then keys are names and values are
        instances of ``RigidDataLoader``.

    metrics : None or dict
        If None the only metric is the loss function. If ``dict`` then keys are names and values are
        instances of ``Loss``.

    benchmarks : None or dict
        If None then no benchmark models used. If ``dict`` then keys are names and values are instances of
        ``Benchmark``.

    device : torch.device or None
        Device on which to perform the deep network calculations. If None then `torch.device('cpu')` used.

    dtype : torch.dtype or None
        Dtype to use for all torch tensors. If None then `torch.double` used.

    optimizer : None or torch.optim.Optimizer
        Optimizer to be used. If None then using Adam with lr=0.01.

    callbacks : list
        List of callbacks to be used.


    Attributes
    ----------
    metrics : dict
        Keys represent metric names and values are callables. Note that it always has an element
        called 'loss' representing the actual loss.

    val_dataloaders : dict
        Keys represent dataloaders names and values are ``RigidDataLoader`` instances. Note that if empty then no
        logging is made.

    models : dict
        Keys represent model names and values are either `Benchmark` or `torch.nn.Module`. Note that it always
        has an element called `main` representing the main network.

    callbacks : list
        Full list of callbacks. There are three defaults ones `BenchmarkCallback`, `ValidationCallback` and
        `ProgressBarCallback`. On top of them there are the manually selected callbacks from `callbacks`.

    """

    def __init__(self, network, loss, train_dataloader, val_dataloaders=None, metrics=None,
                 benchmarks=None, device=None, dtype=None, optimizer=None, callbacks=None):
        # checks
        if not isinstance(train_dataloader, (FlexibleDataLoader, RigidDataLoader)):
            raise TypeError('The train_dataloader needs to be an instance of RigidDataLoader or FlexibleDataLoadeer.')

        if not isinstance(loss, Loss):
            raise TypeError('The loss needs to be an instance of Loss.')

        if not (isinstance(network, torch.nn.Module) and isinstance(network, Benchmark)):
            raise TypeError('The network needs to be a torch.nn.Module and Benchmark. ')

        self.network = network
        self.loss = loss
        self.train_dataloader = train_dataloader

        # metrics
        self.metrics = {
            'loss': loss}

        if metrics is None:
            pass

        elif isinstance(metrics, dict):
            if not all([isinstance(x, Loss) for x in metrics.values()]):
                raise TypeError('All values of metrics need to be Loss.')

            if 'loss' in metrics:
                raise ValueError("Cannot name a metric 'loss' - restricted for the actual loss.")

            self.metrics.update(metrics)

        else:
            raise TypeError('Invalid type of metrics: {}'.format(type(metrics)))

        # metrics_dataloaders
        self.val_dataloaders = {}

        if val_dataloaders is None:
            pass

        elif isinstance(val_dataloaders, dict):
            if not all([isinstance(x, RigidDataLoader) for x in val_dataloaders.values()]):
                raise TypeError('All values of val_dataloaders need to be RigidDataLoader.')

            self.val_dataloaders.update(val_dataloaders)

        else:
            raise TypeError('Invalid type of val_dataloaders: {}'.format(type(val_dataloaders)))

        # benchmarks
        self.models = {'main': network}

        if benchmarks is None:
            pass

        elif isinstance(benchmarks, dict):
            if not all([isinstance(x, Benchmark) for x in benchmarks.values()]):
                raise TypeError('All values of benchmarks need to be a Benchmark.')

            if 'main' in benchmarks:
                raise ValueError("Cannot name a benchmark 'main' - restricted for the main network.")

            self.models.update(benchmarks)
        else:
            raise TypeError('Invalid type of benchmarks: {}'.format(type(benchmarks)))

        self.callbacks = [BenchmarkCallback(), ValidationCallback(), ProgressBarCallback()] + (callbacks or [])
        # Inject self into callbacks
        for cb in self.callbacks:
            cb.run = self

        self.history = History()

        self.loss = loss
        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-2) if optimizer is None else optimizer
        self.current_epoch = -1

    def launch(self, n_epochs=1):
        """Launch the training and logging loop.

        Parameters
        ----------
        n_epochs : int
            Number of epochs.
        """
        try:
            self.network.to(device=self.device, dtype=self.dtype)
            # Train begin
            if self.current_epoch == -1:
                self.on_train_begin(metadata={'n_epochs': n_epochs})

            for _ in range(n_epochs):
                self.current_epoch += 1
                # Epoch begin
                self.on_epoch_begin(metadata={'epoch': self.current_epoch})

                for batch_ix, (X_batch, y_batch, timestamps, asset_names) in enumerate(self.train_dataloader):
                    # Batch begin
                    self.on_batch_begin(metadata={'asset_names': asset_names,
                                                  'batch': batch_ix,
                                                  'epoch': self.current_epoch,
                                                  'timestamps': timestamps,
                                                  'X_batch': X_batch,
                                                  'y_batch': y_batch})

                    # Get batch
                    X_batch, y_batch = X_batch.to(self.device).to(self.dtype), y_batch.to(self.device).to(self.dtype)

                    # Make sure network on the right device and train mode
                    self.network.train()

                    # Forward & Backward
                    weights = self.network(X_batch)
                    loss_per_sample = self.loss(weights, y_batch)
                    loss = loss_per_sample.mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Switch back to eval mode
                    self.network.eval()

                    # Batch end
                    self.on_batch_end(metadata={'asset_names': asset_names,
                                                'batch': batch_ix,
                                                'batch_loss': loss.item(),
                                                'epoch': self.current_epoch,
                                                'timestamps': timestamps,
                                                'weights': weights,
                                                'X_batch': X_batch,
                                                'y_batch': y_batch})

                # Epoch end
                self.on_epoch_end(metadata={'epoch': self.current_epoch,
                                            'n_epochs': n_epochs})

            # Train end
            self.on_train_end()

        except (EarlyStoppingException, KeyboardInterrupt, SolverError) as ex:
            print('Training interrupted')
            time.sleep(1)

            self.on_train_interrupt(metadata={'exception': ex,
                                              'locals': locals()})

        return self.history

    def on_train_begin(self, metadata=None):
        """Take actions at the beginning of the training."""
        for cb in self.callbacks:
            cb.on_train_begin(metadata=metadata)

    def on_train_interrupt(self, metadata=None):
        """Take actions when training interrupted."""
        for cb in self.callbacks:
            cb.on_train_interrupt(metadata=metadata)

    def on_train_end(self, metadata=None):
        """Take actions at the end of the training."""
        for cb in self.callbacks:
            cb.on_train_end(metadata=metadata)

    def on_epoch_begin(self, metadata=None):
        """Take actions at the beginning of an epoch."""
        for cb in self.callbacks:
            cb.on_epoch_begin(metadata=metadata)

    def on_epoch_end(self, metadata=None):
        """Take actions at the end of an epoch."""
        for cb in self.callbacks:
            cb.on_epoch_end(metadata=metadata)

    def on_batch_begin(self, metadata=None):
        """Take actions at the beginning of a batch."""
        for cb in self.callbacks:
            cb.on_batch_begin(metadata=metadata)

    def on_batch_end(self, metadata=None):
        """Take actions at the end of a batch."""
        for cb in self.callbacks:
            cb.on_batch_end(metadata=metadata)

    @property
    def hparams(self):
        """Collect relevant hyperparamters specifying an experiment."""
        res = {}
        res.update(self.network.hparams)
        res.update(self.train_dataloader.hparams)
        res.update({'device': str(self.device),
                    'dtype': str(self.dtype),
                    'loss': str(self.loss),
                    'weight_decay': self.optimizer.defaults.get('weight_decay', ''),
                    'lr': self.optimizer.defaults.get('lr', '')
                    })

        return res
