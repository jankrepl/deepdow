from unittest.mock import Mock

from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import pytest

from deepdow.visualize import create_weight_anim, create_weight_heatmap, generate_weights_table


class TestGenerateWeightsTable:
    def test_errors(self, dataloader_dummy, network_dummy):
        with pytest.raises(TypeError):
            generate_weights_table('FAKE', dataloader_dummy)

        with pytest.raises(TypeError):
            generate_weights_table(network_dummy, 'FAKE')

    def test_basic(self, dataloader_dummy, network_dummy):
        weights_table = generate_weights_table(network_dummy, dataloader_dummy)

        assert isinstance(weights_table, pd.DataFrame)
        assert len(weights_table) == len(dataloader_dummy.dataset)
        assert set(weights_table.index.to_list()) == set(dataloader_dummy.dataset.timestamps)
        assert weights_table.columns.to_list() == dataloader_dummy.dataset.asset_names


class TestCreateWeightAnim:
    def test_errors(self):
        with pytest.raises(ValueError):
            create_weight_anim(pd.DataFrame([[0, 1], [1, 2]], columns=['others', 'asset_1']))

        with pytest.raises(ValueError):
            create_weight_anim(pd.DataFrame([[0, 1], [1, 2]]), n_displayed_assets=3)

        with pytest.raises(ValueError):
            create_weight_anim(pd.DataFrame([[0, 1], [1, 2]], columns=['a', 'b']),
                               n_displayed_assets=1,
                               always_visible=['a', 'b'])

    @pytest.mark.parametrize('colors', [None, {'asset_1': 'green'}, ListedColormap(['green', 'red'])])
    def test_portfolio_evolution(self, monkeypatch, colors):
        n_timesteps = 4
        n_assets = 3
        n_displayed_assets = 2

        weights = pd.DataFrame(np.random.random((n_timesteps, n_assets)),
                               index=pd.date_range(start='1/1/2000', periods=n_timesteps),
                               columns=['asset_{}'.format(i) for i in range(n_assets)])

        weights['asset_0'] = 0  # the smallest but we will force its display anyway

        fake_functanim = Mock()
        fake_functanim.return_value = Mock(spec=FuncAnimation)

        monkeypatch.setattr('deepdow.visualize.FuncAnimation', fake_functanim)
        plt_mock = Mock()
        plt_mock.subplots = Mock(return_value=[Mock(), Mock()])

        monkeypatch.setattr('deepdow.visualize.plt', plt_mock)
        ani = create_weight_anim(weights,
                                 n_displayed_assets=n_displayed_assets,
                                 always_visible=['asset_0'],
                                 n_seconds=10,
                                 figsize=(1, 1),
                                 colors=colors)

        assert isinstance(ani, FuncAnimation)


class TestCreateWeightHeatmap:
    @pytest.mark.parametrize('add_sum_column', [True, False])
    @pytest.mark.parametrize('time_format', [None, '%d-%m-%Y'])
    def test_basic(self, time_format, add_sum_column, monkeypatch):
        n_timesteps = 20
        n_assets = 10
        index = list(range(n_timesteps)) if time_format is None else pd.date_range('1/1/2000',
                                                                                   periods=n_timesteps)

        weights = pd.DataFrame(np.random.random(size=(n_timesteps, n_assets)),
                               index=index)

        fake_axes = Mock(spec=Axes)
        fake_axes.xaxis = Mock()

        fake_sns = Mock()
        fake_sns.heatmap.return_value = fake_axes

        monkeypatch.setattr('deepdow.visualize.sns', fake_sns)
        ax = create_weight_heatmap(weights,
                                   time_format=time_format,
                                   add_sum_column=add_sum_column)

        assert isinstance(ax, Axes)
        assert fake_sns.heatmap.call_count == 1
        assert fake_axes.tick_params.call_count == 2
4