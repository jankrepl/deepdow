import pathlib
import time

from IPython.display import display
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from deepdow.benchmarks import OneOverN, Random
from deepdow.data import FlexibleDataLoader, RigidDataLoader


class RunFresh:
    """Represents one experiment.

    Note that we use MLFlow for all logging.

    Parameters
    ----------
    network : nn.Module
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

    device : torch.device
        Device on which to perform the deep network calculations.


    optimizer : None or torch.optim.Optimizer
        Optimizer to be used. If None then using Adam with lr=0.01.


    Attributes
    ----------
    metrics : dict
        Keys represent metric names and values are callables. Note that it always has an element
        called 'loss' representing the actual loss.

    metrics_dataloaders : dict
        Keys represent dataloaders names and values are ``RigidDataLoader`` instances. Note that if empty then no
        logging is made.

    models : dict
        Keys represent model names and values are either `Benchmark` or `torch.nn.Module`. Note that it always
        has an element called `main` representing the main network. Each model will is logged into a nested MLflow run.

    """

    def __init__(self, network, loss, train_dataloader, val_dataloaders=None, additional_lookbacks=None, metrics=None,
                 benchmarks=None, mlflow_experiment_name='test', device=None, optimizer=None):
        """Construct"""

        # checks
        if not isinstance(train_dataloader, (FlexibleDataLoader, RigidDataLoader)):
            raise TypeError('The train_dataloader needs to be an instance of TrainDataLoader.')

        if not isinstance(network, torch.nn.Module):
            raise TypeError('The network needs to be a torch.nn.Module')

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
        self.mlflow_experiment_name = mlflow_experiment_name
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-2) if optimizer is None else optimizer

        # mlflow games
        self.mlflow_client = mlflow.tracking.MlflowClient()
        self.mlflow_parent_run_id = None  # to be populated in _initialize_mlflow
        self.mlflow_children_run_ids = None  # to be populated in _initialize_mlflow
        self._initialize_mlflow()

        # temp

        self.stats = None

    def _initialize_mlflow(self):
        """Setup mlflow hierarchy in the right way."""
        mlflow.set_experiment(self.mlflow_experiment_name)

        with mlflow.start_run(run_name='Parent'):
            self.mlflow_parent_run_id = mlflow.active_run().info.run_id
            self.mlflow_children_run_ids = {}

            mlflow.log_params(self.train_dataloader.mlflow_params)
            mlflow.log_param('loss', str(self.loss))
            mlflow.log_param('device', str(self.device))

            for model_name, model in self.models.items():
                if not model.lookback_invariant:
                    for lb in self.lookbacks:
                        new_model_name = "{}_lb{}".format(model_name, lb)
                        with mlflow.start_run(run_name=new_model_name, nested=True):
                            mlflow.log_params(model.mlflow_params if hasattr(model, 'mlflow_params') else {})
                            self.mlflow_children_run_ids[new_model_name] = mlflow.active_run().info.run_id
                else:
                    with mlflow.start_run(run_name=model_name, nested=True):
                        mlflow.log_params(model.mlflow_params if hasattr(model, 'mlflow_params') else {})
                        self.mlflow_children_run_ids[model_name] = mlflow.active_run().info.run_id

    def launch(self, n_epochs=1, starting_epoch=0, epoch_end_freq=5, verbose=False):
        """Launch the training and logging loop.


        Parameters
        ----------
        n_epochs : int
            Number of epochs.

        starting_epoch : int
            Initial epoch to start with (just for notation purposes - no model loading).

        epoch_end_freq : int
            How frequently to run `on_epoch_end` (where all the validation takes place). The higher the less we
            perform logging. If `epoch_end_freq=1` then done after each epoch.

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
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Make sure network on the right device and in eval mode
                self.network = self.network.to(self.device)
                self.network = self.network.train()

                # Forward & Backward
                weights = self.network(X_batch, debug_mode=False)
                loss_per_sample = self.loss(weights, y_batch)
                loss = loss_per_sample.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Batch end
                self.on_batch_end(batch_ix)

            # Epoch end
            if e % epoch_end_freq == 0:
                self.on_epoch_end(e, verbose=verbose)

        # Train end
        self.on_train_end()

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_begin(self, epoch):
        print('Epoch: {}'.format(epoch))
        time.sleep(0.25)

    def on_epoch_end(self, epoch, verbose=False):
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

        results_list = []

        with torch.no_grad():
            for dl_name, dl in self.val_dataloaders.items():

                for batch_ix, (X_batch, y_batch, timestamps_batch, asset_names_batch) in enumerate(dl):

                    for lookback_ix, lookback in enumerate(self.lookbacks):
                        if dl.lookback < lookback:
                            continue

                        X_batch_lb = X_batch[:, :, -lookback:, :]

                        for model_name, model in self.models.items():
                            if model.lookback_invariant and lookback_ix > 0:
                                continue

                            if isinstance(model, torch.nn.Module):
                                weights = model(X_batch_lb, debug_mode=False)
                            else:
                                weights = model(X_batch_lb)

                            weights_df = pd.DataFrame(weights.detach().numpy(), columns=asset_names_batch)

                            for metric_name, metric in self.metrics.items():
                                metric_per_s = metric(weights, y_batch).to(torch.device('cpu'))

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

            master_df = pd.concat(results_list, axis=0)

            self.stats = master_df

            # sanity checker - that the DF looks good
            per_model_mean_metrics = self.parser_per_model_mean_metrics(master_df)

            for name, metrics_dict in per_model_mean_metrics.items():
                with mlflow.start_run(run_id=self.mlflow_children_run_ids[name]):
                    mlflow.log_metrics(metrics_dict, step=epoch)

            per_model_weights = self.parser_per_model_weights(master_df)
            for name, weights_dict in per_model_weights.items():
                with mlflow.start_run(run_id=self.mlflow_children_run_ids[name]):
                    root_path = pathlib.Path(mlflow.get_artifact_uri()[6:])
                    for dl_name, weights_df in weights_dict.items():
                        final_dir = root_path / str(epoch) / dl_name
                        final_dir.mkdir(parents=True, exist_ok=True)
                        weights_df_styled = weights_df.style.background_gradient(cmap='Reds').render()

                        with open(str(final_dir / 'weights.html'), "w") as f:
                            f.write(weights_df_styled)

    def on_batch_begin(self, batch):
        pass

    def on_batch_end(self, batch):
        pass

    @staticmethod
    def parser_per_model_mean_metrics(master_df):
        """Parse dataframe into MLflow metrics.

        Parameters
        ----------
        master_df : pd.DataFrame




        Returns
        -------
        dict
            Nested dictionary. First level keys represent model names (+lookback) and the second level represents
            metric names (+ validation name)
        """
        master_df['model_name_lb'] = master_df['model_name'] + master_df['lookback'].apply(
            lambda x: "{}".format('_lb{}'.format(x) if x else ''))
        master_df['dl_metric_name'] = master_df['dl_name'] + '_' + master_df['metric_name']

        multiindex_df = master_df.groupby(['model_name_lb', 'dl_metric_name'])['metric_value'].mean()

        return multiindex_df.unstack(level=0).to_dict()

    @staticmethod
    def parser_per_model_weights(master_df):
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
