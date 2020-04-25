from unittest.mock import Mock

from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import pytest

from deepdow.visualize import portfolio_evolution


class TestPortoflioEvolution:
    def test_errors(self):
        with pytest.raises(ValueError):
            portfolio_evolution(pd.DataFrame([[0, 1], [1, 2]], columns=['others', 'asset_1']))

        with pytest.raises(ValueError):
            portfolio_evolution(pd.DataFrame([[0, 1], [1, 2]]), n_displayed_assets=3)

        with pytest.raises(ValueError):
            portfolio_evolution(pd.DataFrame([[0, 1], [1, 2]], columns=['a', 'b']),
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
        ani = portfolio_evolution(weights,
                                  n_displayed_assets=n_displayed_assets,
                                  always_visible=['asset_0'],
                                  n_seconds=10,
                                  figsize=(1, 1),
                                  colors=colors)

        assert isinstance(ani, FuncAnimation)
