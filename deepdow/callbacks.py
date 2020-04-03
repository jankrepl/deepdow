import pathlib
import sys

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm

from .data import RigidDataLoader
from .utils import ChangeWorkingDirectory


class Callback:
    def on_train_begin(self, metadata):
        pass

    def on_train_interrupt(self, metadata):
        pass

    def on_train_end(self, metadata):
        pass

    def on_epoch_begin(self, metadata):
        pass

    def on_epoch_end(self, metadata):
        pass

    def on_batch_begin(self, metadata):
        pass

    def on_batch_end(self, metadata):
        pass


class BenchmarkCallback(Callback):
    def __init__(self, lookbacks=None):
        self.lookbacks = lookbacks

        self.run = None

    def on_train_begin(self, metadata):
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
                params.update(self.run.train_dataloader.mlflow_params)
                params.update(self.run.network.mlflow_params)

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

                    # print(history.metrics.groupby(['model', 'metric', 'epoch', 'dataloader'])['value'].mean())


class ProgressBarCallback(Callback):
    """Progress bar reporting remaining steps and running training metrics.

    Attributes
    ----------
    bar : tqdm.tqdm
        Bar object that is going to be instantiated at the beginning of each epoch.

    metrics : dict
        Keys are equal to `self.run.metrics.keys()` and the values are list that are appended on batch end with
        after gradient step metrics.

    run : Run
        Run object that is running the main trainign loop. One can get access to multiple useful things like the
        network (`run.network`), train dataloader (`run.train_dataloader`)

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

    @staticmethod
    def create_custom_postfix_str(dict, n_decimals=5):
        fmt_str = "{}={:." + str(n_decimals) + "f}"
        str_l = [fmt_str.format(k, v) for k, v in dict.items()]

        return ", ".join(str_l)

    def on_epoch_begin(self, metadata):

        self.bar = tqdm.tqdm(total=len(self.run.train_dataloader),
                             leave=True,
                             file=self.output)
        self.metrics = {metric: [] for metric in self.run.metrics.keys()}

    def on_epoch_end(self, metadata):
        # collect
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
        weights = metadata.get('weights')
        y_batch = metadata.get('y_batch')

        for metric, cal in self.run.metrics.items():
            self.metrics[metric].append(cal(weights, y_batch).mean().item())

        log_dict = {m: np.mean(vals) for m, vals in self.metrics.items()}

        self.bar.update()
        self.bar.set_postfix_str(self.create_custom_postfix_str(log_dict))


class TensorBoardCallback(Callback):
    def __init__(self, log_dir=None, ts=None):
        """

        Parameters
        ----------
        log_dir
        ts : datetime.datetime or None
            If ``datetime.datetime``, then only logging specific sample corresponding to provided timestamp.
            If None then logging every sample.
        """
        self.writer = SummaryWriter(log_dir)
        self.counter = 0
        self.ts = ts

        self.run = None

        self.activations = {}
        self.handles = []

    def on_batch_begin(self, metadata):
        timestamps = metadata.get('timestamps')

        if self.ts is not None and self.ts not in timestamps:
            return

        def hook(model, inp, out):
            self.activations[model] = (inp, out)

        for layer in self.run.network._modules.values():
            self.handles.append(layer.register_forward_hook(hook))

    def on_batch_end(self, metadata):
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
                        self.writer.add_histogram(s.__class__.__name__ + "_{}_{}".format('inp' if i == 0 else 'out', j),
                                                  y[ix],
                                                  global_step=self.counter)

        for handle in self.handles:
            handle.remove()

        self.handles = []
        self.activations = {}

        self.counter += 1


class ValidationCallback(Callback):
    def __init__(self, freq=1, lookbacks=None):
        self.freq = freq
        self.lookbacks = lookbacks

        self.run = None  # to be populated later

    def on_epoch_end(self, metadata):
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
