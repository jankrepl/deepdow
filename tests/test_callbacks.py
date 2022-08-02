"""Collection of tests focused on the callbacks module."""
import datetime
import pathlib

import pandas as pd
import pytest

from deepdow.callbacks import (
    BenchmarkCallback,
    Callback,
    EarlyStoppingCallback,
    EarlyStoppingException,
    ModelCheckpointCallback,
    MLFlowCallback,
    ProgressBarCallback,
    TensorBoardCallback,
    ValidationCallback,
)

ALL_METHODS = [
    "on_train_begin",
    "on_epoch_begin",
    "on_batch_begin",
    "on_train_interrupt",
    "on_batch_end",
    "on_epoch_end",
    "on_train_end",
]

ALL_CALLBACKS = [
    BenchmarkCallback,
    Callback,
    MLFlowCallback,
    ProgressBarCallback,
    TensorBoardCallback,
    ValidationCallback,
]


@pytest.mark.parametrize("lookbacks", [None, [2, 3]])
def test_benchmark(run_dummy, metadata_dummy, lookbacks):
    cb = BenchmarkCallback(lookbacks)
    cb.run = run_dummy

    run_dummy.callbacks = []  # make sure there are no default callbacks
    run_dummy.callbacks.append(cb)

    for method_name in ALL_METHODS:
        getattr(run_dummy, method_name)(metadata_dummy)

    assert isinstance(run_dummy.history.metrics_per_epoch(-1), pd.DataFrame)
    assert len(run_dummy.history.metrics["epoch"].unique()) == 1


class TestEarlyStoppingCallback:
    def test_error(self, run_dummy, metadata_dummy):
        dataloader_name = list(run_dummy.val_dataloaders.keys())[0]
        metric_name = list(run_dummy.metrics.keys())[0]

        cb_wrong_dataloader = EarlyStoppingCallback(
            dataloader_name="fake", metric_name=metric_name
        )
        cb_wrong_metric = EarlyStoppingCallback(
            dataloader_name=dataloader_name, metric_name="fake"
        )

        cb_wrong_dataloader.run = run_dummy
        cb_wrong_metric.run = run_dummy

        with pytest.raises(ValueError):
            cb_wrong_dataloader.on_train_begin(metadata_dummy)

        with pytest.raises(ValueError):
            cb_wrong_metric.on_train_begin(metadata_dummy)

    def test_basic(self, run_dummy, metadata_dummy):
        dataloader_name = list(run_dummy.val_dataloaders.keys())[0]
        metric_name = list(run_dummy.metrics.keys())[0]

        cb = EarlyStoppingCallback(
            dataloader_name=dataloader_name,
            metric_name=metric_name,
            patience=0,
        )

        cb.run = run_dummy
        cb_val = ValidationCallback()
        cb_val.run = run_dummy

        run_dummy.callbacks = [
            cb_val,
            cb,
        ]  # make sure there are no default callbacks

        with pytest.raises(EarlyStoppingException):
            for method_name in ALL_METHODS:
                getattr(run_dummy, method_name)(metadata_dummy)

        cb.on_train_interrupt({"exception": EarlyStoppingException()})


class TestMLFlowCallback:
    @pytest.mark.parametrize(
        "log_benchmarks", [True, False], ids=["log_bmarks", "dont_log_bmarks"]
    )
    def test_independent(
        self, run_dummy, metadata_dummy, tmpdir, log_benchmarks
    ):
        with pytest.raises(ValueError):
            MLFlowCallback(
                run_name="name",
                run_id="some_id",
                log_benchmarks=log_benchmarks,
                mlflow_path=pathlib.Path(str(tmpdir)),
            )

        cb = MLFlowCallback(
            mlflow_path=pathlib.Path(str(tmpdir)),
            experiment_name="test",
            log_benchmarks=log_benchmarks,
        )
        cb.run = run_dummy

        run_dummy.callbacks = [cb]  # make sure there are no default callbacks

        for method_name in ALL_METHODS:
            getattr(run_dummy, method_name)(metadata_dummy)

    def test_benchmarks(self, run_dummy, metadata_dummy, tmpdir):
        cb = MLFlowCallback(
            mlflow_path=pathlib.Path(str(tmpdir)),
            experiment_name="test",
            log_benchmarks=True,
        )
        cb_bm = BenchmarkCallback()

        cb.run = run_dummy
        cb_bm.run = run_dummy

        run_dummy.callbacks = [
            cb_bm,
            cb,
        ]  # make sure there are no default callbacks

        for method_name in ALL_METHODS:
            getattr(run_dummy, method_name)(metadata_dummy)

    def test_validation(self, run_dummy, metadata_dummy, tmpdir):
        cb = MLFlowCallback(
            mlflow_path=pathlib.Path(str(tmpdir)),
            experiment_name="test",
            log_benchmarks=False,
        )
        cb_val = ValidationCallback()

        cb.run = run_dummy
        cb_val.run = run_dummy

        run_dummy.callbacks = [
            cb_val,
            cb,
        ]  # make sure there are no default callbacks

        for method_name in ALL_METHODS:
            getattr(run_dummy, method_name)(metadata_dummy)


class TestModelCheckpointCallback(Callback):
    def test_error(self, run_dummy, metadata_dummy, tmpdir):
        dataloader_name = list(run_dummy.val_dataloaders.keys())[0]
        metric_name = list(run_dummy.metrics.keys())[0]

        folder_path = pathlib.Path(str(tmpdir))
        some_file_path = folder_path / "some_file.txt"
        some_file_path.touch()

        with pytest.raises(NotADirectoryError):
            ModelCheckpointCallback(
                folder_path=some_file_path,
                dataloader_name=dataloader_name,
                metric_name=metric_name,
            )

        cb_wrong_dataloader = ModelCheckpointCallback(
            folder_path, dataloader_name="fake", metric_name=metric_name
        )
        cb_wrong_metric = ModelCheckpointCallback(
            folder_path, dataloader_name=dataloader_name, metric_name="fake"
        )

        cb_wrong_dataloader.run = run_dummy
        cb_wrong_metric.run = run_dummy

        with pytest.raises(ValueError):
            cb_wrong_dataloader.on_train_begin(metadata_dummy)

        with pytest.raises(ValueError):
            cb_wrong_metric.on_train_begin(metadata_dummy)

    def test_basic(self, run_dummy, metadata_dummy, tmpdir):
        dataloader_name = list(run_dummy.val_dataloaders.keys())[0]
        metric_name = list(run_dummy.metrics.keys())[0]

        cb = ModelCheckpointCallback(
            folder_path=pathlib.Path(str(tmpdir)),
            dataloader_name=dataloader_name,
            metric_name=metric_name,
            verbose=True,
        )

        cb.run = run_dummy
        cb_val = ValidationCallback()
        cb_val.run = run_dummy

        run_dummy.callbacks = [
            cb_val,
            cb,
        ]  # make sure there are no default callbacks

        for method_name in ALL_METHODS:
            getattr(run_dummy, method_name)(metadata_dummy)

        cb.on_train_interrupt({"exception": EarlyStoppingException()})


class TestProgressBarCallback:
    @pytest.mark.parametrize("output", ["stderr", "stdout"])
    def test_independent(self, run_dummy, metadata_dummy, output):
        with pytest.raises(ValueError):
            ProgressBarCallback(output="{}_fake".format(output))

        cb = ProgressBarCallback(output=output)

        cb.run = run_dummy

        run_dummy.callbacks = [cb]  # make sure there are no default callbacks

        for method_name in ALL_METHODS:
            getattr(run_dummy, method_name)(metadata_dummy)

    def test_validation(self, run_dummy, metadata_dummy):
        cb = ProgressBarCallback()
        cb_val = ValidationCallback()

        cb.run = run_dummy
        cb_val.run = run_dummy

        run_dummy.callbacks = [
            cb_val,
            cb,
        ]  # make sure there are no default callbacks

        for method_name in ALL_METHODS:
            getattr(run_dummy, method_name)(metadata_dummy)


class TestTensorBoardCallback:
    @pytest.mark.parametrize(
        "ts_type", ["single_inside", "single_outside", "all"]
    )
    def test_independent(self, run_dummy, metadata_dummy, tmpdir, ts_type):

        if ts_type == "single_inside":
            ts = metadata_dummy["timestamps"][0]
        elif ts_type == "single_outside":
            ts = datetime.datetime.now()
        elif ts_type == "all":
            ts = None
        else:
            ValueError()

        cb = TensorBoardCallback(log_dir=pathlib.Path(str(tmpdir)), ts=ts)
        cb.run = run_dummy

        run_dummy.callbacks = [cb]  # make sure there are no default callbacks

        for method_name in ALL_METHODS:
            if method_name == "on_batch_end":
                run_dummy.network(
                    metadata_dummy["X_batch"]
                )  # let the forward hook take effect

            getattr(run_dummy, method_name)(metadata_dummy)

    @pytest.mark.parametrize(
        "bm_available",
        [True, False],
        ids=["bmarks_available", "bmarks_unavailable"],
    )
    def test_benchmark(self, run_dummy, metadata_dummy, bm_available, tmpdir):
        cb = TensorBoardCallback(
            log_benchmarks=True, log_dir=pathlib.Path(str(tmpdir))
        )
        cb_bm = BenchmarkCallback()

        cb.run = run_dummy
        cb_bm.run = run_dummy

        run_dummy.callbacks = (
            [cb_bm, cb] if bm_available else [cb]
        )  # make sure there are no default callbacks

        for method_name in ALL_METHODS:
            getattr(run_dummy, method_name)(metadata_dummy)

    def test_validation(self, run_dummy, metadata_dummy, tmpdir):
        cb = TensorBoardCallback(log_dir=pathlib.Path(str(tmpdir)))
        cb_val = ValidationCallback()

        cb.run = run_dummy
        cb_val.run = run_dummy

        run_dummy.callbacks = [
            cb_val,
            cb,
        ]  # make sure there are no default callbacks

        for method_name in ALL_METHODS:
            getattr(run_dummy, method_name)(metadata_dummy)


@pytest.mark.parametrize("lookbacks", [None, [2, 3]])
def test_validation(run_dummy, metadata_dummy, lookbacks):
    cb = ValidationCallback(lookbacks=lookbacks)
    cb.run = run_dummy

    run_dummy.callbacks = [cb]  # make sure there are no default callbacks

    for method_name in ALL_METHODS:
        getattr(run_dummy, method_name)(metadata_dummy)

    assert isinstance(
        run_dummy.history.metrics_per_epoch(metadata_dummy["epoch"]),
        pd.DataFrame,
    )
    assert len(run_dummy.history.metrics["epoch"].unique()) == 1
