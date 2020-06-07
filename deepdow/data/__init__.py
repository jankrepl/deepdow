"""Module dealing with data."""

from .augment import (Compose, Dropout, Multiply, Noise)
from .load import (FlexibleDataLoader, InRAMDataset, RigidDataLoader)

__all__ = ['Compose',
           'Dropout',
           'FlexibleDataLoader',
           'InRAMDataset',
           'Multiply',
           'Noise',
           'RigidDataLoader']
