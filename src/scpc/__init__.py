"""Stratified Cyclic Phase Cosmology scientific package."""

from .models.phase import PeriodicPotential, SCPCParameters, integrate_scpc
from .models.standard import ExpansionParameters, FLRWExpansion

__all__ = [
    "ExpansionParameters",
    "FLRWExpansion",
    "PeriodicPotential",
    "SCPCParameters",
    "integrate_scpc",
]

__version__ = "0.1.0"
