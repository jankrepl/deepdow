import pathlib
import pprint
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
from .utils import ChangeWorkingDirectory, MLflowUtils


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


class ActivationExtractorCallback(Callback):
    def __init__(self, ts):
        self.ts = ts

        self.activations = {}
        self.handles = []

    def on_batch_begin(self, metadata):
        timestamps = metadata.get('timestamps')

        if self.ts not in timestamps:
            return

        def hook(model, inp, out):
            self.activations[model] = (inp, out)

        for layer in self.run.network._modules.values():
            self.handles.append(layer.register_forward_hook(hook))

    def on_batch_end(self, metadata):
        timestamps = metadata.get('timestamps')

        if self.ts not in timestamps:
            return

        # print([repr(m) for m in self.activations.keys()])

        # get weights
        ix = timestamps.index(self.ts)
        class2pointer = {v.__class__.__name__: v for v in self.activations.keys()}

        weights = self.activations[class2pointer['PortfolioOptimization']][1][[ix]].detach().numpy()

        sns.heatmap(weights)
        plt.show()
        #
        # act_np = {k.__class__.__name__: v.detach().numpy().shape for k, v in self.activations.items()}
        # pprint.pprint(act_np)
        # cleanup
        for handle in self.handles:
            handle.remove()

        self.handles = []

        self.activations = {}


class AveragePortfolioCallback(Callback):
    """At the end of each epoch compute average portfolio."""

    def __init__(self, freq=1, n_to_display=10):
        self.freq = freq
        self.n_to_display = n_to_display

        self.run = None  # to be populated later

        self.weights_list = []  # list of pd.DataFrame

    def on_train_begin(self, metadata):
        if not isinstance(self.run.train_dataloader, RigidDataLoader):
            raise ValueError('The {} only supports RigidDataLoader'.format(self.__class__.__name__))

    def on_epoch_begin(self, metadata):
        self.weights_list = []

    def on_batch_end(self, metadata):
        asset_names = metadata.get('asset_names')
        timestamps = metadata.get('timestamps')
        weights = metadata.get('weights')

        self.weights_list.append(pd.DataFrame(weights.detach().numpy(), index=timestamps, columns=asset_names))

    def on_epoch_end(self, metadata):
        epoch = metadata.get('epoch')

        if epoch % self.freq == 0:
            final_df = pd.concat(self.weights_list, axis=0)
            results = pd.DataFrame({'mean': final_df.mean(),
                                    'std': final_df.std()})

            print('Average portfolio:')
            print(results.sort_values('mean', ascending=False).iloc[:self.n_to_display])


class BenchmarkCallback(Callback):
    def __init__(self, lookbacks=None):
        self.lookbacks = lookbacks

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


class EmbeddingCallback(Callback):
    def __init__(self, ts=None):
        self.ts = ts

        self.done_for_this_epoch = False

    def on_epoch_begin(self, metadata):
        self.done_for_this_epoch = False

    def on_batch_end(self, metadata):
        X_batch = metadata.get('X_batch')
        timestamps = metadata.get('timestamps')
        tc_features = metadata.get('tc_features')
        rets = metadata.get('rets')
        covmat_sqrt = metadata.get('covmat_sqrt')

        if self.done_for_this_epoch:
            return

        if self.ts is None:
            img_raw = X_batch[0, 0].cpu().detach().numpy()
            img_emb = tc_features[0].cpu().detach().numpy()
            img_rets = rets[0].cpu().detach().numpy()
            img_covmat = covmat_sqrt[0].cpu().detach().numpy()
        else:
            if self.ts in timestamps:
                ix = timestamps.index(self.ts)
                img_raw = X_batch[ix, 0].cpu().detach().numpy()
                img_emb = tc_features[ix].cpu().detach().numpy()
                img_rets = rets[[ix]].cpu().detach().numpy()
                img_covmat = covmat_sqrt[ix].cpu().detach().numpy()
            else:
                return

        _, (ax_raw, ax_emb, ax_rets, ax_covmat) = plt.subplots(1, 4)
        ax_raw.set_title(str(self.ts))
        sns.heatmap(img_raw, ax=ax_raw, vmin=-2, vmax=2)
        sns.heatmap(img_emb, ax=ax_emb)
        sns.heatmap(img_rets, ax=ax_rets)
        sns.heatmap(img_covmat, ax=ax_covmat)

        plt.show()
        self.done_for_this_epoch = True


class ExceptionCallback(Callback):
    def on_batch_begin(self, metadata):
        raise ValueError('value error')

    def on_train_interrupt(self, metadata):
        print('Ok, something went wrong but we are gonna fix it')
        print(metadata.get('locals').keys())


class GammaStatsCallback(Callback):
    def __init__(self, freq=1):
        self.freq = freq

        self.run = None  # to be populated later

        self.gamma_list = []  # list of pd.Series

    def on_epoch_begin(self, metadata):
        self.gamma_list = []

    def on_batch_end(self, metadata):
        gamma = metadata.get('gamma')
        timestamps = metadata.get('timestamps')
        self.gamma_list.append(pd.Series(gamma.detach().numpy(), index=timestamps))

    def on_epoch_end(self, metadata):
        epoch = metadata.get('epoch')
        result = pd.concat(self.gamma_list)

        if epoch % self.freq == 0:
            print('Gamma: mean={} std={}'.format(result.mean(), result.std()))


class InputCheckerCallback(Callback):
    def on_batch_begin(self, metadata):
        X_batch = metadata.get('X_batch')
        y_batch = metadata.get('y_batch')

        print("X - min:{}, mean:{}, max:{}".format(X_batch.min(), X_batch.mean(), X_batch.max()))
        print("y - min:{}, mean:{}, max:{}".format(y_batch.min(), y_batch.mean(), y_batch.max()))


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
        If None then current working directory used.

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

        self._client = mlflow.tracking.MlflowClient()
        self._run_id = run_id or None
        self.run = None

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
                df = self.run.history.metrics_per_epoch(-1)  # only benchmarks
                for bm_name in df['model'].unique():
                    with mlflow.start_run(run_name=bm_name):
                        temp_df = df[df['model'] == bm_name]
                        metrics = {'_'.join(list(map(lambda x: str(x), k))): v for k, v in
                                   temp_df.groupby(['dataloader', 'metric', 'lookback'])['value'].mean().items()}

                        mlflow.log_metrics(metrics, step=0)
                        mlflow.log_metrics(metrics, step=10)

    def on_epoch_end(self, metadata):
        epoch = metadata.get('epoch')

        with ChangeWorkingDirectory(self.mlflow_path):
            # log some metadata
            if self.experiment_name is not None:
                mlflow.set_experiment(self.experiment_name)

            with mlflow.start_run(run_id=self._run_id):
                df = self.run.history.metrics_per_epoch(epoch)
                metrics = {'_'.join(list(map(lambda x: str(x), k))): v for k, v in
                           df.groupby(['dataloader', 'metric', 'lookback'])['value'].mean().items()}
                mlflow.log_metrics(metrics, step=epoch)

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
        df = self.run.history.metrics_per_epoch(epoch)
        additional_metrics = {'_'.join(list(map(lambda x: str(x), k))): v for k, v in
                              df.groupby(['dataloader', 'metric'])['value'].mean().items()}

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


class MLFlowCallback_(Callback):
    def __init__(self, experiment_name='test', parent_run_name='parent', log_mean_metrics=True,
                 log_weights=False, log_timestamped_metrics=False, freq=20):
        self.experiment_name = experiment_name
        self.parent_run_name = parent_run_name
        self.log_mean_metrics = log_mean_metrics
        self.log_weights = log_weights
        self.log_timestamped_metrics = log_timestamped_metrics
        self.freq = freq

        self.parent_run_id = None
        self.run_ids = {}  # run ids of the children
        self.run = None

    def _initialize_mlflow(self):
        """Setup mlflow hierarchy in the right way.

        We create one parent run and children runs. For lookback invariant models there is going to be a single run.
        Each named in {model_name}. For lookback dependant models there are going to be multiple runs based
        on the `len(self.lookbacks)` and the name is {model_name}_lb{lookback}.

        """
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=self.parent_run_name):
            self.mlflow_parent_run_id = mlflow.active_run().info.run_id

            mlflow.log_params(self.run.train_dataloader.mlflow_params)
            mlflow.log_param('loss', str(self.run.loss))
            mlflow.log_param('device', str(self.run.device))

            for model_name, model in self.run.models.items():
                if not model.lookback_invariant:
                    for lb in self.run.lookbacks:
                        new_model_name = "{}_lb{}".format(model_name, lb)
                        with mlflow.start_run(run_name=new_model_name, nested=True):
                            mlflow.log_params(model.mlflow_params if hasattr(model, 'mlflow_params') else {})
                            self.run_ids[new_model_name] = mlflow.active_run().info.run_id
                else:
                    with mlflow.start_run(run_name=model_name, nested=True):
                        mlflow.log_params(model.mlflow_params if hasattr(model, 'mlflow_params') else {})
                        self.run_ids[model_name] = mlflow.active_run().info.run_id

    def on_train_begin(self, metadata):
        self._initialize_mlflow()

    def on_epoch_end(self, metadata):
        epoch = metadata.get('epoch')

        if epoch % self.freq != 0:
            return

        print('Starting on epoch end computations ...')

        # Set all torch models to eval mode
        for m in self.run.models:
            if isinstance(m, torch.nn.Module):
                m.eval()

        first_time = epoch == 0  # probably should come up with a better condition

        results_list = []

        with torch.no_grad():
            for dl_name, dl in self.run.val_dataloaders.items():

                for batch_ix, (X_batch, y_batch, timestamps_batch, asset_names_batch) in enumerate(dl):
                    X_batch = X_batch.to(self.run.device).to(self.run.dtype)
                    y_batch = y_batch.to(self.run.device).to(self.run.dtype)
                    for lookback_ix, lookback in enumerate(self.run.lookbacks):
                        if dl.lookback < lookback:
                            continue

                        X_batch_lb = X_batch[:, :, -lookback:, :]

                        for model_name, model in self.run.models.items():
                            if model.lookback_invariant and lookback_ix > 0:
                                continue

                            if not first_time and not (model is self.run.network or not model.deterministic):
                                continue

                            weights = model(X_batch_lb)
                            weights_df = pd.DataFrame(weights.cpu().numpy(), columns=asset_names_batch)

                            for metric_name, metric in self.run.metrics.items():
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

            # Metrics per model
            if self.log_mean_metrics:
                per_model_mean_metrics = self.parser_per_model_mean_metrics(master_df)
                for name, run_id in self.run_ids.items():
                    if name in per_model_mean_metrics:
                        with mlflow.start_run(run_id=run_id):
                            mlflow.log_metrics(per_model_mean_metrics[name], step=epoch)
                    else:
                        MLflowUtils.copy_metrics(run_id, step=epoch)

            # Weights per model
            if self.log_weights:
                per_model_weights = self.parser_per_model_weights(master_df)
                for name, weights_dict in per_model_weights.items():
                    with mlflow.start_run(run_id=self.run_ids[name]):
                        root_path = pathlib.Path(mlflow.get_artifact_uri()[6:])
                        for dl_name, weights_df in weights_dict.items():
                            final_dir = root_path / str(epoch) / dl_name
                            final_dir.mkdir(parents=True, exist_ok=True)
                            weights_df_styled = weights_df.style.background_gradient(cmap='Reds').render()

                            with open(str(final_dir / 'weights.html'), "w") as f:
                                f.write(weights_df_styled)

            if self.log_timestamped_metrics:
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
