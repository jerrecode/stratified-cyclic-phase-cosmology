"""Stratified Cyclic Phase Cosmology research software."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scpc-cosmology")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
