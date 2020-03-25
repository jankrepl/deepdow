import datetime
import pathlib
import time

from diffcp import SolverError
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from .benchmarks import Benchmark
from .callbacks import BenchmarkCallback, ProgressBarCallback, ValidationCallback
from .data import FlexibleDataLoader, RigidDataLoader
from .utils import MLflowUtils


class History:
    """A shared information database for the training process."""

    def __init__(self):
        self.database = {}  # dict where keys are epochs and values are lists
        self.misc = {}

    @property
    def metrics(self):
        master_list = []  # over all epochs
        for l in self.database.values():
            master_list.extend(l)

        return pd.DataFrame(master_list)

    def metrics_per_epoch(self, epoch):
        return pd.DataFrame(self.database[epoch])

    def add_entry(self, model=None, metric=None, batch=None, epoch=None, dataloader=None,
                  lookback=None, timestamp=None, value=np.nan):

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

    metrics : None or list or dict
        If None the only metric is the loss function. If ``list`` then represents a collection of unnamed callables.
        The names for logging are generated based on the order. if ``dict`` then keys are names and values are
        callables. In terms of logging, the metrics will be prepended by the name of the names of validation data
        loader, thatis "{validator_name}_{metric_name}".

    benchmarks : None or list or dict
        If None then no benchmark models used. If ``list`` then represents sequence of unnamed ``Benchmark`` or
        ``torch.nn.Module`` instances. If ``dict`` then keys are names and values are instances of ``Benchmark`` or
        ``torch.nn.Module``.

    device : torch.device or None
        Device on which to perform the deep network calculations. If None then `torch.device('cpu')` used.

    dtype : torch.dtype or None
        Dtype to use for all torch tensors. If None then `torch.double` used.

    optimizer : None or torch.optim.Optimizer
        Optimizer to be used. If None then using Adam with lr=0.01.


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

    """

    def __init__(self, network, loss, train_dataloader, val_dataloaders=None, metrics=None,
                 benchmarks=None, device=None, dtype=None, optimizer=None, callbacks=None):
        """Construct"""

        # checks
        if not isinstance(train_dataloader, (FlexibleDataLoader, RigidDataLoader)):
            raise TypeError('The train_dataloader needs to be an instance of TrainDataLoader.')

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
        elif isinstance(metrics, list):
            self.metrics.update({'metric_{}'.format(i): c for i, c in enumerate(metrics)})

        elif isinstance(metrics, dict):
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

        elif isinstance(benchmarks, list):
            self.models.update({'bm_{}'.format(i): bm for i, bm in enumerate(benchmarks)})

        elif isinstance(benchmarks, dict):
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
        self.dtype = dtype or torch.double
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-2) if optimizer is None else optimizer

    def launch(self, n_epochs=1, starting_epoch=0):
        """Launch the training and logging loop.


        Parameters
        ----------
        n_epochs : int
            Number of epochs.

        starting_epoch : int
            Initial epoch to start with (just for notation purposes - no model loading).
        """
        try:
            self.network.to(device=self.device, dtype=self.dtype)
            # Train begin
            self.on_train_begin()

            for e in range(starting_epoch, starting_epoch + n_epochs):
                # Epoch begin
                self.on_epoch_begin(metadata={'epoch': e})

                for batch_ix, (X_batch, y_batch, timestamps, asset_names) in enumerate(self.train_dataloader):
                    # Batch begin
                    self.on_batch_begin(metadata={'asset_names': asset_names,
                                                  'batch': batch_ix,
                                                  'epoch': e,
                                                  'timestamps': timestamps,
                                                  'X_batch': X_batch,
                                                  'y_batch': y_batch})

                    # Get batch
                    X_batch, y_batch = X_batch.to(self.device).to(self.dtype), y_batch.to(self.device).to(self.dtype)

                    # Make sure network on the right device and in eval mode
                    self.network.train()

                    # Forward & Backward
                    weights = self.network(X_batch)
                    loss_per_sample = self.loss(weights, y_batch)
                    loss = loss_per_sample.mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Batch end
                    self.on_batch_end(metadata={'asset_names': asset_names,
                                                'batch': batch_ix,
                                                'batch_loss': loss.item(),
                                                'epoch': e,
                                                'timestamps': timestamps,
                                                'weights': weights,
                                                'X_batch': X_batch,
                                                'y_batch': y_batch})

                # Epoch end
                self.on_epoch_end(metadata={'epoch': e})

            # Train end
            self.on_train_end()

        except (KeyboardInterrupt, SolverError) as ex:
            print('Training interrupted')
            time.sleep(1)

            self.on_train_interrupt(metadata={'exception': ex,
                                              'locals': locals()})

        return self.history

    def on_train_begin(self, metadata=None):
        for cb in self.callbacks:
            cb.on_train_begin(metadata=metadata)

    def on_train_interrupt(self, metadata=None):
        for cb in self.callbacks:
            cb.on_train_interrupt(metadata=metadata)

    def on_train_end(self, metadata=None):
        for cb in self.callbacks:
            cb.on_train_end(metadata=metadata)

    def on_epoch_begin(self, metadata=None):
        for cb in self.callbacks:
            cb.on_epoch_begin(metadata=metadata)

    def on_epoch_end(self, metadata=None):
        for cb in self.callbacks:
            cb.on_epoch_end(metadata=metadata)

    def on_batch_begin(self, metadata=None):
        for cb in self.callbacks:
            cb.on_batch_begin(metadata=metadata)

    def on_batch_end(self, metadata=None):
        for cb in self.callbacks:
            cb.on_batch_end(metadata=metadata)


class RunFresh:
    """Represents one experiment.

    Note that we use MLFlow for all logging.

    Parameters
    ----------
    network : nn.Module and Benchmark
        Network.

    loss : callable
        A callable computing per sample loss. Specifically it accepts `weights`, `y` and returns ``torch.Tensor``
         of shape `(n_samples,)`.

    train_dataloader : FlexibleDataLoader or RigidDataLoader
        DataLoader used for training.

    val_dataloaders  : None or list or dict
        If None no validation is performed. If ``list`` then represents a collection of unnamed ``RigidDataLoader``.
        The names for logging are generated based on the order. If ``dict`` then keys are names and values are
        instances of ``RigidDataLoader``.

    additional_lookbacks : tuple or None
        Additional lookbacks to analyze for each model which has `model.lookback_invariant=False`. Each
        will be thought of as a separate model. If None then only the original lookbacks from `val_dataloaders` are
        considered. In terms of logging, for each lookback we will create a separate mlflow run with a name
        "{model_name}_lb{lookback}".

    metrics : None or list or dict
        If None the only metric is the loss function. If ``list`` then represents a collection of unnamed callables.
        The names for logging are generated based on the order. if ``dict`` then keys are names and values are
        callables. In terms of logging, the metrics will be prepended by the name of the names of validation data
        loader, thatis "{validator_name}_{metric_name}".

    benchmarks : None or list or dict
        If None then no benchmark models used. If ``list`` then represents sequence of unnamed ``Benchmark`` or
        ``torch.nn.Module`` instances. If ``dict`` then keys are names and values are instances of ``Benchmark`` or
        ``torch.nn.Module``.

    mlflow_experiment_name : str
        Name of the mlflow experiment.

    device : torch.device or None
        Device on which to perform the deep network calculations. If None then `torch.device('cpu')` used.

    dtype : torch.dtype or None
        Dtype to use for all torch tensors. If None then `torch.double` used.

    optimizer : None or torch.optim.Optimizer
        Optimizer to be used. If None then using Adam with lr=0.01.


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

    lookbacks : list
        List of all considered lookbacks.

    mlflow_client : mlflow.tracking.MlflowClient
        MlflowClient instance for accessing and logging.

    mlflow_run_ids : dict
        Stores references to all MLflow runs. Keys represent {model_name}_lb{lookback} for models that are not lookback
        invariant else {model_name}. The values are the MLflow run ids. Note that the network is stored under `main`.
    

    """

    def __init__(self, network, loss, train_dataloader, val_dataloaders=None, additional_lookbacks=None, metrics=None,
                 benchmarks=None, mlflow_experiment_name='test', device=None, dtype=None, optimizer=None):
        """Construct"""

        # checks
        if not isinstance(train_dataloader, (FlexibleDataLoader, RigidDataLoader)):
            raise TypeError('The train_dataloader needs to be an instance of TrainDataLoader.')

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
        elif isinstance(metrics, list):
            self.metrics.update({'metric_{}'.format(i): c for i, c in enumerate(metrics)})

        elif isinstance(metrics, dict):
            if 'loss' in metrics:
                raise ValueError("Cannot name a metric 'loss' - restricted for the actual loss.")

            self.metrics.update(metrics)

        else:
            raise TypeError('Invalid type of metrics: {}'.format(type(metrics)))

        # metrics_dataloaders
        self.val_dataloaders = {}

        if val_dataloaders is None:
            pass

        elif isinstance(val_dataloaders, list):
            if not all([isinstance(x, RigidDataLoader) for x in val_dataloaders]):
                raise TypeError('All entries of val_dataloaders need to be RigidDataLoader.')

            self.val_dataloaders.update({'dl_{}'.format(i): dl for i, dl in enumerate(val_dataloaders)})
        elif isinstance(val_dataloaders, dict):
            if not all([isinstance(x, RigidDataLoader) for x in val_dataloaders.values()]):
                raise TypeError('All values of val_dataloaders need to be RigidDataLoader.')

            self.val_dataloaders.update(val_dataloaders)

        else:
            raise TypeError('Invalid type of val_dataloaders: {}'.format(type(val_dataloaders)))

        # lookbacks
        lookbacks_set = {dl.lookback for dl in self.val_dataloaders.values()} | set(additional_lookbacks or [])
        self.lookbacks = sorted(list(lookbacks_set))

        # benchmarks
        self.models = {'main': network}

        if benchmarks is None:
            pass

        elif isinstance(benchmarks, list):
            self.models.update({'bm_{}'.format(i): bm for i, bm in enumerate(benchmarks)})

        elif isinstance(benchmarks, dict):
            if 'main' in benchmarks:
                raise ValueError("Cannot name a benchmark 'main' - restricted for the main network.")

            self.models.update(benchmarks)
        else:
            raise TypeError('Invalid type of benchmarks: {}'.format(type(benchmarks)))

        self.loss = loss
        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.double
        self.mlflow_experiment_name = mlflow_experiment_name
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-2) if optimizer is None else optimizer

        # mlflow games
        self.mlflow_client = mlflow.tracking.MlflowClient()
        self.mlflow_parent_run_id = None  # to be populated in _initialize_mlflow
        self.mlflow_run_ids = {}  # to be populated in _initialize_mlflow
        self._initialize_mlflow()

        # temp
        self.stats = {}

    def _initialize_mlflow(self):
        """Setup mlflow hierarchy in the right way.

        We create one parent run and children runs. For lookback invariant models there is going to be a single run.
        Each named in {model_name}. For lookback dependant models there are going to be multiple runs based
        on the `len(self.lookbacks)` and the name is {model_name}_lb{lookback}.

        """
        mlflow.set_experiment(self.mlflow_experiment_name)

        with mlflow.start_run(run_name='Parent'):
            self.mlflow_parent_run_id = mlflow.active_run().info.run_id

            mlflow.log_params(self.train_dataloader.mlflow_params)
            mlflow.log_param('loss', str(self.loss))
            mlflow.log_param('device', str(self.device))

            for model_name, model in self.models.items():
                if not model.lookback_invariant:
                    for lb in self.lookbacks:
                        new_model_name = "{}_lb{}".format(model_name, lb)
                        with mlflow.start_run(run_name=new_model_name, nested=True):
                            mlflow.log_params(model.mlflow_params if hasattr(model, 'mlflow_params') else {})
                            self.mlflow_run_ids[new_model_name] = mlflow.active_run().info.run_id
                else:
                    with mlflow.start_run(run_name=model_name, nested=True):
                        mlflow.log_params(model.mlflow_params if hasattr(model, 'mlflow_params') else {})
                        self.mlflow_run_ids[model_name] = mlflow.active_run().info.run_id

    def launch(self, n_epochs=1, starting_epoch=0, epoch_end_freq=5, verbose=False):
        """Launch the training and logging loop.


        Parameters
        ----------
        n_epochs : int
            Number of epochs.

        starting_epoch : int
            Initial epoch to start with (just for notation purposes - no model loading).

        epoch_end_freq : int or None
            How frequently to run `on_epoch_end` (where all the validation takes place). The higher the less we
            perform logging. If `epoch_end_freq=1` then done after each epoch. If None then never run.

        verbose : bool
            Controls verbosity.

        """
        # Train begin
        self.on_train_begin()

        for e in range(starting_epoch, starting_epoch + n_epochs):
            # Epoch begin
            self.on_epoch_begin(e)

            for batch_ix, (X_batch, y_batch, _, _) in tqdm(enumerate(self.train_dataloader)):
                # Batch begin
                self.on_batch_begin(batch_ix)

                # Get batch
                X_batch, y_batch = X_batch.to(self.device).to(self.dtype), y_batch.to(self.device).to(self.dtype)

                # Make sure network on the right device and in eval mode
                self.network.to(device=self.device, dtype=self.dtype)
                self.network.train()

                # Forward & Backward
                weights = self.network(X_batch, debug_mode=False)
                loss_per_sample = self.loss(weights, y_batch)
                loss = loss_per_sample.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Batch end
                self.on_batch_end(batch_ix, loss.item())

            # Epoch end
            if epoch_end_freq is not None and e % epoch_end_freq == 0:
                self.on_epoch_end(e, verbose=verbose)

        # Train end
        self.on_train_end()

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch, verbose=False, log_mean_metrics=True, log_weights=True, log_timestamped_metrics=True):
        """Run on epoch end logic.

        We mostly use it to generate all the logs for MLflow.

        Parameters
        ----------
        epoch : int
            Current epoch.

        verbose : bool
            Controls verbosity.
        """

        print('Starting on epoch end computations ...')

        # Set all torch models to eval mode
        for m in self.models:
            if isinstance(m, torch.nn.Module):
                m.eval()

        first_time = epoch == 0  # probably should come up with a better condition

        results_list = []

        with torch.no_grad():
            for dl_name, dl in self.val_dataloaders.items():

                for batch_ix, (X_batch, y_batch, timestamps_batch, asset_names_batch) in enumerate(dl):
                    X_batch, y_batch = X_batch.to(self.device).to(self.dtype), y_batch.to(self.device).to(self.dtype)

                    for lookback_ix, lookback in enumerate(self.lookbacks):
                        if dl.lookback < lookback:
                            continue

                        X_batch_lb = X_batch[:, :, -lookback:, :]

                        for model_name, model in self.models.items():
                            if model.lookback_invariant and lookback_ix > 0:
                                continue

                            if not first_time and not (model is self.network or not model.deterministic):
                                continue

                            weights = model(X_batch_lb)
                            weights_df = pd.DataFrame(weights.cpu().numpy(), columns=asset_names_batch)

                            for metric_name, metric in self.metrics.items():
                                metric_per_s = metric(weights, y_batch).cpu()

                                df = pd.DataFrame(
                                    {'timestamps': timestamps_batch,  # (n_samples,)
                                     'model_name': model_name,  # str
                                     'metric_name': metric_name,  # str
                                     'lookback': '' if model.lookback_invariant else lookback,  # int
                                     'dl_name': dl_name,  # str
                                     'metric_value': metric_per_s.numpy(),  # (n_samples,)
                                     'weights': [weights_df.iloc[i, :] for i in range(len(weights_df))]  # (n_samples,)
                                     })

                                results_list.append(df)

            if not results_list:
                return

            # Prepare master dataframe
            master_df = pd.concat(results_list, axis=0)
            master_df['model_name_lb'] = master_df['model_name'] + master_df['lookback'].apply(
                lambda x: "{}".format('_lb{}'.format(x) if x else ''))
            master_df['dl_metric_name'] = master_df['dl_name'] + '_' + master_df['metric_name']

            # self.stats = master_df

            # Metrics per model
            if log_mean_metrics:
                per_model_mean_metrics = self.parser_per_model_mean_metrics(master_df)
                for name, run_id in self.mlflow_run_ids.items():
                    if name in per_model_mean_metrics:
                        with mlflow.start_run(run_id=run_id):
                            mlflow.log_metrics(per_model_mean_metrics[name], step=epoch)
                    else:
                        MLflowUtils.copy_metrics(run_id, step=epoch)

            # Weights per model
            if log_weights:
                per_model_weights = self.parser_per_model_weights(master_df)
                for name, weights_dict in per_model_weights.items():
                    with mlflow.start_run(run_id=self.mlflow_run_ids[name]):
                        root_path = pathlib.Path(mlflow.get_artifact_uri()[6:])
                        for dl_name, weights_df in weights_dict.items():
                            final_dir = root_path / str(epoch) / dl_name
                            final_dir.mkdir(parents=True, exist_ok=True)
                            weights_df_styled = weights_df.style.background_gradient(cmap='Reds').render()

                            with open(str(final_dir / 'weights.html'), "w") as f:
                                f.write(weights_df_styled)

            if log_timestamped_metrics:
                per_split_timestamped_metrics = self.parser_per_split_timestamped_metrics(master_df)
                with mlflow.start_run(run_id=self.mlflow_parent_run_id):
                    root_path = pathlib.Path(mlflow.get_artifact_uri()[6:])
                    for dl_name, temp in per_split_timestamped_metrics.items():
                        final_dir = root_path / str(epoch) / dl_name
                        final_dir.mkdir(parents=True, exist_ok=True)
                        for metric_name, df in temp.items():
                            fig, ax = plt.subplots(1, 1, figsize=(30, 10))
                            df.plot(ax=ax)
                            fig.savefig('{}/{}.png'.format(final_dir, metric_name))

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self, batch, loss_step):
        if 'loss' in self.stats:
            self.stats['loss'].append(loss_step)
        else:
            self.stats['loss'] = [loss_step]

        # print('mean: {}'.format(np.mean(self.stats['loss'][-10:])))

    @staticmethod
    def parser_per_model_mean_metrics(master_df):
        """Parse dataframe into MLflow metrics.

        Parameters
        ----------
        master_df : pd.DataFrame
            DataFrame with the following columns: timestamp, model_name, metric_name, lookback, dl_name, metric_value,
            weights, model_name_lb and dl_metric_name. Each row is unique.

        Returns
        -------
        dict
            Nested dictionary. First level keys represent model names (+lookback) and the second level represents
            metric names (+ validation name)
        """
        multiindex_df = master_df.groupby(['model_name_lb', 'dl_metric_name'])['metric_value'].mean()

        return multiindex_df.unstack(level=0).to_dict()

    @staticmethod
    def parser_per_model_weights(master_df):
        """Parse dataframe into model weights."""
        res = {}

        for x in master_df[master_df['metric_name'] == 'loss'].groupby(['model_name_lb', 'dl_name']):
            model_name, dl_name = x[0]

            temp_df = pd.DataFrame({ts: weights for _, (ts, weights) in x[1][['timestamps', 'weights']].iterrows()}).T
            temp_df.sort_index(inplace=True)

            if model_name not in res:
                res[model_name] = {dl_name: temp_df}
            else:
                res[model_name][dl_name] = temp_df

        return res

    @staticmethod
    def parser_per_split_timestamped_metrics(master_df):
        """Parse dataframe into timestamped metrics."""
        all_splits = master_df['dl_name'].unique()
        all_metrics = master_df['metric_name'].unique()

        return {dl_name: {
            metric_name: pd.pivot_table(master_df[master_df['dl_metric_name'] == '{}_{}'.format(dl_name, metric_name)],
                                        index='timestamps',
                                        columns='model_name_lb',
                                        values='metric_value') for metric_name in all_metrics} for dl_name in
            all_splits}
