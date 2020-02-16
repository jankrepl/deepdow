"""Collection of utilities and helpers."""

import mlflow
import numpy as np
import pandas as pd


class PandasChecks:
    """General checks for pandas objects."""

    @staticmethod
    def check_no_gaps(index):
        """Check if a time index has no gaps.

        Parameters
        ----------
        index : pd.DatetimeIndex
            Time index to be checked for gaps.

        Raises
        ------
        TypeError
            If inconvenient type.

        IndexError
            If there is a gap.

        """
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError('Unsupported type: {}'.format(type(index)))

        correct_index = pd.date_range(index[0], periods=len(index), freq=index.freq)

        if not correct_index.equals(index):
            raise IndexError('Index has gaps.')

    @staticmethod
    def check_valid_entries(table):
        """Check if input table has no nan or +-inf entries.

        Parameters
        ----------
        table : pd.Series or pd.DataFrame
            Input table.

        Raises
        ------
        TypeError
            Inappropriate type of `table`.

        ValueError
            At least one entry invalid.

        """
        if not isinstance(table, (pd.Series, pd.DataFrame)):
            raise TypeError('Unsupported type: {}'.format(type(table)))

        if not np.all(np.isfinite(table.values)):
            raise ValueError('There is an invalid entry')

    @staticmethod
    def check_indices_agree(*frames):
        """Check if inputs are pd.Series or pd.DataFrame with same indices / columns.

        Parameters
        ----------
        frames : list
            Elements are either `pd.Series` or `pd.DataFrame`.

        Raises
        ------
        TypeError
            If elements are not `pd.Series` or `pd.DataFrame`.

        IndexError
            If indices/colums do not agree.

        """
        if not all([isinstance(x, (pd.Series, pd.DataFrame)) for x in frames]):
            raise TypeError('Some elements are not pd.Series or pd.DataFrame')

        reference_index = frames[0].index

        for i, f in enumerate(frames):
            if not f.index.equals(reference_index):
                raise IndexError('The {} entry has wrong index: {}'.format(i, f.index))

            if isinstance(f, pd.DataFrame) and not f.columns.equals(reference_index):
                raise IndexError('The {} entry has wrong columns: {}'.format(i, f.columns))


class MLflowUtils:
    @staticmethod
    def copy_metrics(run_id, step, client=None):
        """Copy the latest value of all metrics into a new step.

        Can be used in evaluation loops to avoid recomputing metrics on benchmarks that are deterministic.

        Parameters
        ----------
        run_id : str
            Unique MLflow run indentifier.

        step : int
            Number of the step under which to copy the previous results.

        client : None or mlflow.tracking.MlflowClient
            If not None, then instance of an existing client. If None then instantiated from scratch.

        Returns
        -------
        success : bool
            If True, at least one metric existed and copied. If False, no metrics found.

        """
        client = client or mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        old_meltrics = run.data.metrics

        if old_meltrics:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(run.data.metrics, step=step)

        return bool(old_meltrics)


def prices_to_returns(prices, use_log=True):
    """Convert prices to returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Rows represent different time points and the columns represent different assets.

    use_log : bool
        If True, then logarithmic returns are use (natural logarithm. If False, then standard returns.

    Returns
    -------
    returns : pd.DataFrame
        Returns per asset per period. The first period is deleted.

    """
    # checks

    if use_log:
        values = np.log(prices.values) - np.log(prices.shift(1).values)
    else:
        values = (prices.values - prices.shift(1).values) / prices.shift(1).values

    return pd.DataFrame(values[1:, :], index=prices.index[1:], columns=prices.columns)
