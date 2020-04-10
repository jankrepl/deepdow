import pandas as pd
import pytest
import torch

from deepdow.benchmarks import OneOverN
from deepdow.callbacks import Callback
from deepdow.experiments import History, Run
from deepdow.losses import MeanReturns, StandardDeviation
from deepdow.nn import DummyNet


def test_basic():
    n_channels = 2
    x = torch.rand(10, n_channels, 4, 5)
    network = DummyNet(n_channels=n_channels)
    y = network(x)

    print(y)


def test_history():
    history = History()

    history.add_entry(model='whatever', epoch=1)
    history.add_entry(model='whatever_2', epoch=1, value=3)

    history.add_entry(model='1111', epoch=2)

    metrics_1 = history.metrics_per_epoch(1)
    metrics_2 = history.metrics_per_epoch(2)

    metrics_all = history.metrics

    assert isinstance(metrics_1, pd.DataFrame)
    assert isinstance(metrics_2, pd.DataFrame)
    assert isinstance(metrics_all, pd.DataFrame)

    assert len(metrics_1) == 2
    assert len(metrics_2) == 1
    assert len(metrics_all) == 3

    with pytest.raises(KeyError):
        history.metrics_per_epoch(3)

    history.pretty_print(epoch=1)
    history.pretty_print(epoch=None)


class TestRun:
    def test_wrong_construction_1(self, dataloader_dummy):
        """Wrong positional arguments."""
        with pytest.raises(TypeError):
            Run('this_is_fake', MeanReturns(), dataloader_dummy)

        with pytest.raises(TypeError):
            Run(DummyNet(), 'this_is_fake', dataloader_dummy)

        with pytest.raises(TypeError):
            Run(DummyNet(), MeanReturns(), 'this_is_fake')

    def test_wrong_construction_2(self, dataloader_dummy):
        """Wrong keyword arguments."""
        with pytest.raises(TypeError):
            Run(DummyNet(), MeanReturns(), dataloader_dummy, metrics='this_is_fake')

        with pytest.raises(TypeError):
            Run(DummyNet(), MeanReturns(), dataloader_dummy, metrics={'a': 'this_is_fake'})

        with pytest.raises(ValueError):
            Run(DummyNet(), MeanReturns(), dataloader_dummy, metrics={'loss': MeanReturns()})

        with pytest.raises(TypeError):
            Run(DummyNet(), MeanReturns(), dataloader_dummy, val_dataloaders='this_is_fake')

        with pytest.raises(TypeError):
            Run(DummyNet(), MeanReturns(), dataloader_dummy, val_dataloaders={'val': 'this_is_fake'})

        with pytest.raises(TypeError):
            Run(DummyNet(), MeanReturns(), dataloader_dummy, benchmarks='this_is_fake')

        with pytest.raises(TypeError):
            Run(DummyNet(), MeanReturns(), dataloader_dummy, benchmarks={'uniform': 'this_is_fake'})

        with pytest.raises(ValueError):
            Run(DummyNet(), MeanReturns(), dataloader_dummy, benchmarks={'main': OneOverN()})

    @pytest.mark.parametrize('additional_kwargs', [True, False])
    def test_attributes_after_construction(self, dataloader_dummy, additional_kwargs):
        network = DummyNet()
        loss = MeanReturns()

        kwargs = {}
        if additional_kwargs:
            kwargs.update({'metrics': {'std': StandardDeviation()},
                           'val_dataloaders': {'val': dataloader_dummy},
                           'benchmarks': {'whatever': OneOverN()}})

        run = Run(network, loss, dataloader_dummy, **kwargs)

        assert network is run.network
        assert loss is run.loss
        assert dataloader_dummy is run.train_dataloader
        assert isinstance(run.metrics, dict)
        assert isinstance(run.val_dataloaders, dict)

    def test_launch(self, dataloader_dummy):
        network = DummyNet(n_channels=dataloader_dummy.dataset.X.shape[1])
        loss = MeanReturns()
        run = Run(network, loss, dataloader_dummy)

        run.launch(n_epochs=1)

    def test_launch_interrupt(self, dataloader_dummy, monkeypatch):
        network = DummyNet(n_channels=dataloader_dummy.dataset.X.shape[1])
        loss = MeanReturns()

        class TempCallback(Callback):
            def on_train_begin(self, metadata):
                raise KeyboardInterrupt()

        monkeypatch.setattr('time.sleep', lambda x: None)
        run = Run(network, loss, dataloader_dummy, callbacks=[TempCallback()])

        run.launch(n_epochs=1)
