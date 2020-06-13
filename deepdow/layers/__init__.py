"""Collection of layers."""

from .collapse import (AttentionCollapse, AverageCollapse, ElementCollapse, ExponentialCollapse,
                       MaxCollapse, SumCollapse)
from .allocate import (AnalyticalMarkowitz, NCO, NumericalMarkowitz, Resample, SoftmaxAllocator,
                       SparsemaxAllocator, WeightNorm)
from .misc import Cov2Corr, CovarianceMatrix, KMeans, MultiplyByConstant
from .transform import Conv, RNN, Warp, Zoom

__all__ = ['AnalyticalMarkowitz',
           'AttentionCollapse',
           'AverageCollapse',
           'Conv',
           'Cov2Corr',
           'CovarianceMatrix',
           'ElementCollapse',
           'ExponentialCollapse',
           'KMeans',
           'MaxCollapse',
           'MultiplyByConstant',
           'NCO',
           'NumericalMarkowitz',
           'Resample',
           'RNN',
           'SoftmaxAllocator',
           'SparsemaxAllocator',
           'SumCollapse',
           'Warp',
           'WeightNorm',
           'Zoom']
