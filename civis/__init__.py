"""
CIVis: A visualization server for Calcium Imaging data.

This package provides tools for visualizing and analyzing CI (Calcium Imaging) data.
"""

__version__ = "0.1.0"

from . import servers
from .src.CITank import CITank
from .src.VirmenTank import VirmenTank
from .src.ElecTank import ElecTank

__all__ = ['servers', 'CITank', 'ElecTank', 'VirmenTank']

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())