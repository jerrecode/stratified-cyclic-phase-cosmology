"""Stratified Cyclic Phase Cosmology scientific package."""

from .models.standard import ExpansionParameters, FLRWExpansion
from .models.phase import PeriodicPotential, SCPCParameters, integrate_scpc

__all__ = [
    "ExpansionParameters",
    "FLRWExpansion",
    "PeriodicPotential",
    "SCPCParameters",
    "integrate_scpc",
]

__version__ = "0.1.0"
