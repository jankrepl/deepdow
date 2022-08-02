import datetime
from unittest.mock import MagicMock, Mock

from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import pytest

from deepdow.losses import MeanReturns
from deepdow.visualize import (
    plot_weight_anim,
    plot_weight_heatmap,
    generate_cumrets,
    generate_metrics_table,
    generate_weights_table,
    plot_metrics,
)


class TestGenerateCumrets:
    def test_errors(self, dataloader_dummy, network_dummy):
        with pytest.raises(TypeError):
            generate_cumrets({"bm_1": "WRONG"}, dataloader_dummy)

        with pytest.raises(TypeError):
            generate_cumrets({"bm_1": network_dummy}, "FAKE")

    def test_basic(self, dataloader_dummy, network_dummy):
        cumrets_dict = generate_cumrets(
            {"bm_1": network_dummy}, dataloader_dummy
        )

        assert isinstance(cumrets_dict, dict)
        assert len(cumrets_dict) == 1
        assert "bm_1" in cumrets_dict
        assert cumrets_dict["bm_1"].shape == (
            len(dataloader_dummy.dataset),
            dataloader_dummy.horizon,
        )


class TestGenerateMetricsTable:
    def test_errors(self, dataloader_dummy, network_dummy):
        with pytest.raises(TypeError):
            generate_metrics_table(
                {"bm_1": "WRONG"}, dataloader_dummy, {"metric": MeanReturns()}
            )

        with pytest.raises(TypeError):
            generate_metrics_table(
                {"bm_1": network_dummy}, "FAKE", {"metric": MeanReturns()}
            )

        with pytest.raises(TypeError):
            generate_metrics_table(
                {"bm_1": network_dummy}, dataloader_dummy, {"metric": "FAKE"}
            )

    def test_basic(self, dataloader_dummy, network_dummy):
        metrics_table = generate_metrics_table(
            {"bm_1": network_dummy}, dataloader_dummy, {"rets": MeanReturns()}
        )

        assert isinstance(metrics_table, pd.DataFrame)
        assert len(metrics_table) == len(dataloader_dummy.dataset)
        assert {"metric", "value", "benchmark", "timestamp"} == set(
            metrics_table.columns.to_list()
        )


def test_plot_metrics(monkeypatch):
    n_entries = 100
    metrics_table = pd.DataFrame(
        np.random.random((n_entries, 2)), columns=["value", "timestamp"]
    )
    metrics_table["metric"] = "M"
    metrics_table["benchmark"] = "B"

    fake_plt = Mock()
    fake_plt.subplots.return_value = None, MagicMock()
    fake_pd = Mock()

    monkeypatch.setattr("deepdow.visualize.plt", fake_plt)
    monkeypatch.setattr("deepdow.visualize.pd", fake_pd)

    plot_metrics(metrics_table)


class TestGenerateWeightsTable:
    def test_errors(self, dataloader_dummy, network_dummy):
        with pytest.raises(TypeError):
            generate_weights_table("FAKE", dataloader_dummy)

        with pytest.raises(TypeError):
            generate_weights_table(network_dummy, "FAKE")

    def test_basic(self, dataloader_dummy, network_dummy):
        weights_table = generate_weights_table(network_dummy, dataloader_dummy)

        assert isinstance(weights_table, pd.DataFrame)
        assert len(weights_table) == len(dataloader_dummy.dataset)
        assert set(weights_table.index.to_list()) == set(
            dataloader_dummy.dataset.timestamps
        )
        assert (
            weights_table.columns.to_list()
            == dataloader_dummy.dataset.asset_names
        )


class TestPlotWeightAnim:
    def test_errors(self):
        with pytest.raises(ValueError):
            plot_weight_anim(
                pd.DataFrame([[0, 1], [1, 2]], columns=["others", "asset_1"])
            )

        with pytest.raises(ValueError):
            plot_weight_anim(
                pd.DataFrame([[0, 1], [1, 2]]), n_displayed_assets=3
            )

        with pytest.raises(ValueError):
            plot_weight_anim(
                pd.DataFrame([[0, 1], [1, 2]], columns=["a", "b"]),
                n_displayed_assets=1,
                always_visible=["a", "b"],
            )

    @pytest.mark.parametrize(
        "colors",
        [None, {"asset_1": "green"}, ListedColormap(["green", "red"])],
    )
    def test_portfolio_evolution(self, monkeypatch, colors):
        n_timesteps = 4
        n_assets = 3
        n_displayed_assets = 2

        weights = pd.DataFrame(
            np.random.random((n_timesteps, n_assets)),
            index=pd.date_range(start="1/1/2000", periods=n_timesteps),
            columns=["asset_{}".format(i) for i in range(n_assets)],
        )

        weights[
            "asset_0"
        ] = 0  # the smallest but we will force its display anyway

        fake_functanim = Mock()
        fake_functanim.return_value = Mock(spec=FuncAnimation)

        monkeypatch.setattr("deepdow.visualize.FuncAnimation", fake_functanim)
        plt_mock = Mock()
        plt_mock.subplots = Mock(return_value=[Mock(), Mock()])

        monkeypatch.setattr("deepdow.visualize.plt", plt_mock)
        ani = plot_weight_anim(
            weights,
            n_displayed_assets=n_displayed_assets,
            always_visible=["asset_0"],
            n_seconds=10,
            figsize=(1, 1),
            colors=colors,
        )

        assert isinstance(ani, FuncAnimation)


class TestPlotWeightHeatmap:
    @pytest.mark.parametrize("add_sum_column", [True, False])
    @pytest.mark.parametrize("time_format", [None, "%d-%m-%Y"])
    def test_basic(self, time_format, add_sum_column, monkeypatch):
        n_timesteps = 20
        n_assets = 10
        index = (
            list(range(n_timesteps))
            if time_format is None
            else pd.date_range("1/1/2000", periods=n_timesteps)
        )

        weights = pd.DataFrame(
            np.random.random(size=(n_timesteps, n_assets)), index=index
        )

        fake_axes = Mock(spec=Axes)
        fake_axes.xaxis = Mock()

        fake_sns = Mock()
        fake_sns.heatmap.return_value = fake_axes

        monkeypatch.setattr("deepdow.visualize.sns", fake_sns)
        ax = plot_weight_heatmap(
            weights, time_format=time_format, add_sum_column=add_sum_column
        )

        assert isinstance(ax, Axes)
        assert fake_sns.heatmap.call_count == 1
        assert fake_axes.tick_params.call_count == 2

    def test_sum_column(self):
        with pytest.raises(ValueError):
            now = datetime.datetime.now()
            df = pd.DataFrame(
                np.zeros((2, 2)), columns=["asset", "sum"], index=[now, now]
            )
            plot_weight_heatmap(df, add_sum_column=True)
