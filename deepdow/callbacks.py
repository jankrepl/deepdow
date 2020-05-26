"""Collection of different callbacks."""
import pathlib
import sys

import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

from .utils import ChangeWorkingDirectory


class Callback:
    """Parent class for all callbacks.

    General construct that allows for taking different actions at different points of the training process. One
    can provide a list of callbacks to the ``deepdow.experiments.Run``.

    Notes
    -----
    To implement new callbacks one needs to subclass this class.

    """

    def on_train_begin(self, metadata):
        """Take actions at the beginning of the training.

        Parameters
        ----------
        metadata : dict
            Dictionary that is going to be populated with relevant data within `Run.launch`.
        """
        pass

    def on_train_interrupt(self, metadata):
        """Take actions on training interruption.

        Parameters
        ----------
        metadata : dict
            Dictionary that is going to be populated with relevant data within `Run.launch`. Keys
            available are 'exception', 'locals`.
        """
        pass

    def on_train_end(self, metadata):
        """Take actions at the end of the training.

        Parameters
        ----------
        metadata : dict
            Dictionary that is going to be populated with relevant data within `Run.launch`.
        """
        pass

    def on_epoch_begin(self, metadata):
        """Take actions at the beginning of an epoch.

        Parameters
        ----------
        metadata : dict
            Dictionary that is going to be populated with relevant data within `Run.launch`. Keys
            available are 'epoch'.
        """
        pass

    def on_epoch_end(self, metadata):
        """Take actions at the beginning of an epoch.

        Parameters
        ----------
        metadata : dict
            Dictionary that is going to be populated with relevant data within `Run.launch`. Keys
            available are `epoch`, `n_epochs`.
        """
        pass

    def on_batch_begin(self, metadata):
        """Take actions at the beginning of a batch.

        Parameters
        ----------
        metadata : dict
            Dictionary that is going to be populated with relevant data within `Run.launch`. Keys
            available are 'asset_names', 'batch', 'epoch', 'timestamps', 'X_batch', 'y_batch'.
        """
        pass

    def on_batch_end(self, metadata):
        """Take actions at the beginning of a batch.

        Parameters
        ----------
        metadata : dict
            Dictionary that is going to be populated with relevant data within `Run.launch`. Keys
            available are 'asset_names', 'batch', 'batch_loss', 'epoch', 'timestamps', 'weights',
            'X_batch', 'y_batch'.
        """
        pass


class EarlyStoppingException(Exception):
    """Custom exception raised by EarlyStoppingCallback to stop the training."""


class BenchmarkCallback(Callback):
    """Computation of benchmarks performance over different metrics and dataloaders.

    Parameters
    ----------
    lookbacks : list or None
        If ``list`` then list of integers representing the different lookbacks. The benchmarks will be run for
        all of them. If None then just the default one implied by the dataloader.

    Attributes
    ----------
    run : deepdow.experiments.Run
        Run instance that is using this callback.

    Notes
    -----
    Very useful for establishing baselines for deep learning models.

    """

    def __init__(self, lookbacks=None):
        self.lookbacks = lookbacks

        self.run = None

    def on_train_begin(self, metadata):
        """Compute performance of all benchmarks."""
        with torch.no_grad():
            for dl_name, dl in self.run.val_dataloaders.items():
                for bm_name, bm in self.run.models.items():
                    if bm_name == 'main':
                        continue
                    for batch, (X_batch, y_batch, timestamps_batch, _) in enumerate(dl):
                        X_batch = X_batch.to(self.run.device).to(self.run.dtype)
                        y_batch = y_batch.to(self.run.device).to(self.run.dtype)

                        lookbacks = []
                        if self.lookbacks is None:
                            lookbacks.append(X_batch.shape[2])
                        else:
                            lookbacks = self.lookbacks

                        for lb in lookbacks:

                            weights = bm(X_batch[:, :, -lb:, :])

                            for metric_name, metric in self.run.metrics.items():
                                metric_per_s = metric(weights, y_batch).detach().cpu().numpy()

                                for metric_value, ts in zip(metric_per_s, timestamps_batch):
                                    self.run.history.add_entry(timestamp=ts,
                                                               epoch=-1,
                                                               model=bm_name,
                                                               batch=batch,
                                                               lookback=lb,
                                                               dataloader=dl_name,
                                                               metric=metric_name,
                                                               value=metric_value)

        if len(self.run.models) > 1:
            self.run.history.pretty_print(-1)


class EarlyStoppingCallback(Callback):
    """Early stopping callback.

    In the background, we keep a running minimum of a metric of interest. If it does not change for more than
    `patience` epochs the training is stopped.

    Parameters
    ----------
    dataloader_name : str
        Name of the dataloader, needs to correspond to a key in `val_dataloaders` in ``deepdow.experiments.Run``.

    metric_name : str
        Name of the metric to use (the lower the better),  needs to correspond to a key in `metrics` in
        ``deepdow.experiments.Run``.

    patience : int
        Number of epochs without improvement before the training is stopped.

    Attributes
    ----------
    min : float
        Running minimum of the metric.

    n_epochs_no_improvement : int
        Number of epochs without improvement - not going below the previous minimum.
    """

    def __init__(self, dataloader_name, metric_name, patience=5):
        self.dataloader_name = dataloader_name
        self.metric_name = metric_name
        self.patience = patience

        self.min = np.inf
        self.n_epochs_no_improvement = 0
        self.run = None  # will be injected with an instance of ``Run``.

    def on_train_begin(self, metadata):
        """Check if dataloader name and metric name even exist."""
        if self.dataloader_name not in self.run.val_dataloaders:
            raise ValueError('Did not find the dataloader {}'.format(self.dataloader_name))

        if self.metric_name not in self.run.metrics:
            raise ValueError('Did not find the metric {}'.format(self.metric_name))

    def on_epoch_end(self, metadata):
        """Extract statistics and if necessary stop training."""
        epoch = metadata['epoch']
        stats = self.run.history.metrics_per_epoch(epoch)

        if not (len(stats['lookback'].unique()) == 1 and len(stats['model'].unique()) == 1):
            raise ValueError('EarlyStoppingCallback needs to have a single lookback and model')  # pragma: no cover

        stats_formatted = stats.groupby(['dataloader', 'metric'])['value'].mean().unstack(-1)
        current_metric = stats_formatted.loc[self.dataloader_name, self.metric_name]

        if current_metric < self.min:
            self.min = current_metric
            self.n_epochs_no_improvement = 0
        else:
            self.n_epochs_no_improvement += 1  # pragma: no cover

        if self.n_epochs_no_improvement >= self.patience:
            raise EarlyStoppingException()

    def on_train_interrupt(self, metadata):
        """Handle ``EarlyStoppingException``."""
        ex = metadata['exception']

        if isinstance(ex, EarlyStoppingException):
            msg = 'Training stopped early because there was no improvement in {}_{} for {} epochs'.format(
                self.dataloader_name, self.metric_name, self.patience)
            print(msg)


class MLFlowCallback(Callback):
    """MLFlow logging callback.

    Parameters
    ----------
    run_name : str or None
        If ``str`` then represents the name of a new run to be created. If None then the user eithers provides
        `run_id` of an existing run and everything will be logged into it or a new run with random name would be
        generated.

    mlflow_path : str or pathlib.Path or None
        If ``str`` or ``pathlib.Path`` then represents the absolute path to a folder in which `mlruns` lie.
        If None then home folder used.

    experiment_name : str or None
        Experiment to be use. If None using the default one.

    run_id : str or None
        If provided and `run_name` is None then continuing an existing run. If None than a new run is created.

    log_benchmarks : bool
        If True then all benchmarks will be logged under separate mlflow runs.

    Attributes
    ----------
    run : deepdow.experiments.Run
        Run instance that is using this callback.

    """

    def __init__(self, run_name=None, mlflow_path=None, experiment_name=None, run_id=None, log_benchmarks=False):
        self.run_name = run_name
        self.mlflow_path = mlflow_path
        self.experiment_name = experiment_name

        if run_name is not None and run_id is not None:
            raise ValueError('Cannot provide both run_id and run_name')

        self.run_id = run_id
        self.log_benchmarks = log_benchmarks

        self._run_id = run_id or None
        self.run = None

        with ChangeWorkingDirectory(self.mlflow_path):
            self._client = mlflow.tracking.MlflowClient()

    def on_train_begin(self, metadata):
        """Log hyperparameters and potentially benchmarks performance."""
        with ChangeWorkingDirectory(self.mlflow_path):
            # log some metadata
            if self.experiment_name is not None:
                mlflow.set_experiment(self.experiment_name)

            with mlflow.start_run(run_name=self.run_name, run_id=self._run_id):
                # if run_id is not None then run_name is ignored
                self._run_id = mlflow.active_run().info.run_id
                params = {
                    'device': self.run.device,
                    'dtype': self.run.dtype,
                    'train_dataloader': self.run.train_dataloader.__class__.__name__
                }
                params.update(self.run.train_dataloader.hparams)
                params.update(self.run.network.hparams)

                mlflow.log_params(params)

            if self.log_benchmarks:
                try:
                    df = self.run.history.metrics_per_epoch(-1)  # only benchmarks
                    for bm_name in df['model'].unique():
                        with mlflow.start_run(run_name=bm_name):
                            temp_df = df[df['model'] == bm_name]
                            metrics = {'_'.join(list(map(lambda x: str(x), k))): v for k, v in
                                       temp_df.groupby(['dataloader', 'metric', 'lookback'])['value'].mean().items()}

                            mlflow.log_metrics(metrics, step=0)
                            mlflow.log_metrics(metrics, step=10)
                except KeyError:
                    return

    def on_epoch_end(self, metadata):
        """Read relevant results and log into MLflow."""
        epoch = metadata.get('epoch')

        with ChangeWorkingDirectory(self.mlflow_path):
            # log some metadata
            if self.experiment_name is not None:
                mlflow.set_experiment(self.experiment_name)

            with mlflow.start_run(run_id=self._run_id):
                try:
                    df = self.run.history.metrics_per_epoch(epoch)

                    metrics = {'_'.join(list(map(lambda x: str(x), k))): v for k, v in
                               df.groupby(['dataloader', 'metric', 'lookback'])['value'].mean().items()}
                    mlflow.log_metrics(metrics, step=epoch)

                except KeyError:
                    return


class ModelCheckpointCallback(Callback):
    """Model checkpointing callback.

    In the background, we keep a running minimum of a metric of interest.

    Parameters
    ----------
    folder_path : str or pathlib.Path
        Directory to which to save the checkpoints.

    dataloader_name : str
        Name of the dataloader, needs to correspond to a key in `val_dataloaders` in ``deepdow.experiments.Run``.

    metric_name : str
        Name of the metric to use (the lower the better),  needs to correspond to a key in `metrics` in
        ``deepdow.experiments.Run``.

    verbose : bool
        If True, each checkpointing triggers a message.

    Attributes
    ----------
    min : float
        Running minimum of the metric.
    """

    def __init__(self, folder_path, dataloader_name, metric_name, verbose=False, save_best_only=False):
        self.folder_path = pathlib.Path(folder_path)
        if self.folder_path.is_file():
            raise NotADirectoryError('The checkpointing path needs to be a folder.')

        self.dataloader_name = dataloader_name
        self.metric_name = metric_name
        self.verbose = verbose
        self.save_best_only = save_best_only

        self.min = np.inf
        self.run = None  # will be injected with an instance of ``Run``.

    def on_train_begin(self, metadata):
        """Check if dataloader name and metric name even exist."""
        self.folder_path.mkdir(parents=True, exist_ok=True)

        if self.dataloader_name not in self.run.val_dataloaders:
            raise ValueError('Did not find the dataloader {}'.format(self.dataloader_name))

        if self.metric_name not in self.run.metrics:
            raise ValueError('Did not find the metric {}'.format(self.metric_name))

    def on_epoch_end(self, metadata):
        """Store checkpoint if metric is in its all time low."""
        epoch = metadata['epoch']
        stats = self.run.history.metrics_per_epoch(epoch)

        if not (len(stats['lookback'].unique()) == 1 and len(stats['model'].unique()) == 1):
            raise ValueError('ModelCheckpointCallback needs to have a single lookback and model')  # pragma: no cover

        stats_formatted = stats.groupby(['dataloader', 'metric'])['value'].mean().unstack(-1)
        current_metric = stats_formatted.loc[self.dataloader_name, self.metric_name]

        if current_metric < self.min:
            self.min = current_metric

            checkpoint_path = self.folder_path / 'model_{:02d}__{:.4f}.pth'.format(epoch, current_metric)
            torch.save(self.run.network, str(checkpoint_path))

            if self.verbose:
                print('Checkpointed {}'.format(checkpoint_path))


class ProgressBarCallback(Callback):
    """Progress bar reporting remaining steps and relevant metrics.

    Attributes
    ----------
    bar : tqdm.tqdm
        Bar object that is going to be instantiated at the beginning of each epoch.

    metrics : dict
        Keys are equal to `self.run.metrics.keys()` and the values are list that are appended on batch end with
        after gradient step metrics.

    run : Run
        Run object that is running the main training loop. One can get access to multiple useful things like the
        network (`run.network`), train dataloader (`run.train_dataloader`) etc.

    output : str, {'stdout', 'stderr'}
        Where to output the progress bar.
    """

    def __init__(self, output='stderr', n_decimals=3):
        self.bar = None
        self.metrics = {}
        self.n_decimals = n_decimals

        if output == 'stderr':
            self.output = sys.stderr
        elif output == 'stdout':
            self.output = sys.stdout
        else:
            raise ValueError('Unrecognized output {}'.format(output))

        self.run = None

    def on_epoch_begin(self, metadata):
        """Initialize tqdm bar and metric lists."""
        self.bar = tqdm.tqdm(total=len(self.run.train_dataloader),
                             leave=True,
                             desc='Epoch {}'.format(metadata['epoch']),
                             file=self.output)
        self.metrics = {metric: [] for metric in self.run.metrics.keys()}

    def on_epoch_end(self, metadata):
        """Update finished progress bar with latest epoch metrics."""
        epoch = metadata.get('epoch')

        try:
            df = self.run.history.metrics_per_epoch(epoch)
            additional_metrics = {'_'.join(list(map(lambda x: str(x), k))): v for k, v in
                                  df.groupby(['dataloader', 'metric'])['value'].mean().items()}

        except KeyError:
            # no val_dataloaders
            additional_metrics = {}

        old_postfix = self.bar.postfix
        new_postfix = self.create_custom_postfix_str(additional_metrics)

        final_postfix = "{}, {}".format(old_postfix, new_postfix)
        self.bar.set_postfix_str(final_postfix)

        del self.bar

    def on_batch_end(self, metadata):
        """Update progress bar with batch metrics."""
        weights = metadata.get('weights')
        y_batch = metadata.get('y_batch')

        for metric, cal in self.run.metrics.items():
            self.metrics[metric].append(cal(weights, y_batch).mean().item())

        log_dict = {m: np.mean(vals) for m, vals in self.metrics.items()}

        self.bar.update()
        self.bar.set_postfix_str(self.create_custom_postfix_str(log_dict))

    @staticmethod
    def create_custom_postfix_str(metrics, n_decimals=5):
        """Create a custom string with metrics.

        Parameters
        ----------
        metrics : dict
            Keys represent names and the

        n_decimals : int
            Number of decimals to display.

        Returns
        -------
        formatted : str
            Nicely formatted string to be appended to the progress bar.

        """
        fmt_str = "{}={:." + str(n_decimals) + "f}"
        str_l = [fmt_str.format(k, v) for k, v in metrics.items()]

        return ", ".join(str_l)


class TensorBoardCallback(Callback):
    """Tensorboard logging interface.

    Currently supports:
        - images (evolution of predicted weights over time)
        - histograms (activations of input and outputs of all layers)
        - hyperparamters
        - scalars (logged metrics)

    Parameters
    ----------
    log_dir : None or str or pathlib.Path
        Represent the folder where to checkpoints will be saved. If None then using
        `cwd/runs/CURRENT_DATETIME_HOSTNAME`. Else the exact path.

    ts : datetime.datetime or None
        If ``datetime.datetime``, then only logging specific sample corresponding to provided timestamp.
        If None then logging every sample.

    log_benchmarks : bool
        If True, then benchmark metrics are logged to scalars. The folder is `log_dir / bm_name`.

    Attributes
    ----------
    run : deepdow.experiments.Run
        Run instance that is using this callback.

    """

    def __init__(self, log_dir=None, ts=None, log_benchmarks=False):
        self.log_dir = pathlib.Path(log_dir) if log_dir is not None else pathlib.Path.cwd()
        self.writer = SummaryWriter(self.log_dir)
        self.counter = 0
        self.ts = ts
        self.log_benchmarks = log_benchmarks

        self.run = None

        self.activations = {}
        self.handles = []
        self.weights = []

    def on_train_begin(self, metadata):
        """Log benchmarks performance."""
        n_epochs = metadata.get('n_epochs')

        if self.log_benchmarks:
            try:
                df = self.run.history.metrics_per_epoch(-1)  # only benchmarks
                for bm_name in df['model'].unique():
                    temp_df = df[df['model'] == bm_name]
                    metrics = {'/'.join(list(map(lambda x: str(x), k))): v for k, v in
                               temp_df.groupby(['dataloader', 'metric', 'lookback'])['value'].mean().items()}

                    bm_writer = SummaryWriter(self.log_dir / bm_name)

                    for metric_name, metric_value in metrics.items():
                        for global_step in range(n_epochs):
                            bm_writer.add_scalar(metric_name, metric_value, global_step=global_step)

            except KeyError:
                return

    def on_batch_begin(self, metadata):
        """Set up forward hooks."""
        timestamps = metadata.get('timestamps')

        if self.ts is not None and self.ts not in timestamps:
            return

        def hook(model, inp, out):
            self.activations[model] = (inp, out)

        for layer in self.run.network._modules.values():
            self.handles.append(layer.register_forward_hook(hook))

    def on_batch_end(self, metadata):
        """Log activations."""
        timestamps = metadata.get('timestamps')
        weights = metadata.get('weights')

        # cache weights
        self.weights.append(pd.DataFrame(weights.detach().cpu().numpy(), index=timestamps))

        # add activations
        self._add_activations(metadata)

    def on_epoch_end(self, metadata):
        """Log images, metrics and hyperparamters."""
        epoch = metadata.get('epoch')
        n_epochs = metadata.get('n_epochs')

        # create weight image
        master_df = pd.concat(self.weights).sort_index()
        self.writer.add_image('weights', master_df.values[np.newaxis, ...], global_step=metadata['epoch'])
        self.weights = []

        # log scalars
        try:
            df = self.run.history.metrics_per_epoch(epoch)

            metrics = {'/'.join(list(map(lambda x: str(x), k))): v for k, v in
                       df.groupby(['dataloader', 'metric', 'lookback'])['value'].mean().items()}

            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(metric_name, metric_value, global_step=epoch)

            if epoch == n_epochs - 1:
                self.writer.add_hparams(self.run.hparams, metrics)

        except KeyError:
            pass

    def _add_activations(self, metadata):
        """Add activations."""
        X_batch = metadata.get('X_batch')
        timestamps = metadata.get('timestamps')

        if self.ts is not None and self.ts not in timestamps:
            return

        ix = timestamps.index(self.ts) if self.ts is not None else list(range(len(X_batch)))
        self.writer.add_histogram(tag='inputs', values=X_batch[ix], global_step=self.counter)

        for s, io in self.activations.items():
            for i, x in enumerate(io):
                if torch.is_tensor(x):
                    self.writer.add_histogram(s.__class__.__name__ + "_{}".format('inp' if i == 0 else 'out'),
                                              x[ix],
                                              global_step=self.counter)
                else:
                    for j, y in enumerate(x):
                        if y is None:
                            continue  # pragma: no cover
                        self.writer.add_histogram(s.__class__.__name__ + "_{}_{}".format('inp' if i == 0 else 'out', j),
                                                  y[ix],
                                                  global_step=self.counter)

        for handle in self.handles:
            handle.remove()

        self.handles = []
        self.activations = {}

        self.counter += 1


class ValidationCallback(Callback):
    """Logging of all metrics for all validation dataloaders.

    Parameters
    ----------
    freq : int
        With what frequiency to compute metrics. If equal to 1 then every epoch. The higher
        the less frequent the logging will be.

    lookbacks : list or None
        If ``list`` then list of integers representing the different lookbacks. The benchmarks will be run for
        all of them. If None then just the default one implied by the dataloader.

    Attributes
    ----------
    run : deepdow.experiments.Run
        Run instance that is using this callback.
    """

    def __init__(self, freq=1, lookbacks=None):
        self.freq = freq
        self.lookbacks = lookbacks

        self.run = None  # to be populated later

    def on_epoch_end(self, metadata):
        """Compute metrics and log them into the history object."""
        epoch = metadata.get('epoch')
        model = self.run.network
        model.eval()

        if epoch % self.freq == 0:
            with torch.no_grad():
                for dl_name, dl in self.run.val_dataloaders.items():
                    for batch, (X_batch, y_batch, timestamps_batch, _) in enumerate(dl):
                        X_batch = X_batch.to(self.run.device).to(self.run.dtype)
                        y_batch = y_batch.to(self.run.device).to(self.run.dtype)

                        lookbacks = []
                        if self.lookbacks is None:
                            lookbacks.append(X_batch.shape[2])
                        else:
                            lookbacks = self.lookbacks

                        for lb in lookbacks:
                            weights = model(X_batch[:, :, -lb:, :])

                            for metric_name, metric in self.run.metrics.items():
                                metric_per_s = metric(weights, y_batch).detach().cpu().numpy()

                                for metric_value, ts in zip(metric_per_s, timestamps_batch):
                                    self.run.history.add_entry(timestamp=ts,
                                                               model='network',
                                                               epoch=epoch,
                                                               batch=batch,
                                                               lookback=lb,
                                                               dataloader=dl_name,
                                                               metric=metric_name,
                                                               value=metric_value)
