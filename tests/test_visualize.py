from unittest.mock import Mock

from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import pytest

from deepdow.visualize import create_weight_anim, generate_weights_table


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

        monkeypatch.setattr('deepdow.visualize.FuncAnimation', lambda *args, **kwargs: Mock(spec=FuncAnimation))
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
