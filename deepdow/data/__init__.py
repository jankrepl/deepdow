"""Module dealing with data."""

from .load import (Compose, Dropout, FlexibleDataLoader, InRAMDataset, Multiply, Noise,
                   RigidDataLoader)

__all__ = ['Compose',
           'Dropout',
           'FlexibleDataLoader',
           'InRAMDataset',
           'Multiply',
           'Noise',
           'RigidDataLoader']
