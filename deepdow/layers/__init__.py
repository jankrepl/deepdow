"""Collection of layers."""

from .collapse import AttentionCollapse, AverageCollapse, ElementCollapse, MaxCollapse, SumCollapse
from .convex import Markowitz
from .misc import CovarianceMatrix, MultiplyByConstant, SoftmaxAllocator
from .transform import Conv, RNN

__all__ = ['AttentionCollapse',
           'AverageCollapse',
           'Conv',
           'CovarianceMatrix',
           'ElementCollapse',
           'Markowitz',
           'MaxCollapse',
           'MultiplyByConstant',
           'RNN',
           'SoftmaxAllocator',
           'SumCollapse']
