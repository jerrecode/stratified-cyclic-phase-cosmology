"""Download and read the official DESI DR2 cosmology-chain products.

The chain products are external data. This module deliberately does not vendor them
into the source tree: it records the exact public URLs and hashes downloaded files so
that publication figures can be regenerated without silently substituting a local
posterior approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np

DESI_DR2_CHAIN_ROOT = "https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/cobaya"


@dataclass(frozen=True)
class ChainSet:
    """In-memory Cobaya chain table with per-sample weights."""

    names: tuple[str, ...]
    values: np.ndarray
    weights: np.ndarray
    source_files: tuple[Path, ...]

    def column(self, *aliases: str) -> np.ndarray:
        """Return a parameter column, accepting common naming aliases."""
        canonical = {_canonical(name): i for i, name in enumerate(self.names)}
        for alias in aliases:
            key = _canonical(alias)
            if key in canonical:
                return self.values[:, canonical[key]]
        raise KeyError(f"None of the aliases {aliases!r} occur in chain columns {self.names!r}")

    def omega_m(self) -> np.ndarray:
        try:
            return self.column("omegam", "omega_m", "Omega_m")
        except KeyError:
            h = self.column("H0") / 100.0
            ombh2 = self.column("ombh2", "omega_b_h2")
            omch2 = self.column("omch2", "omega_c_h2")
            try:
                omnuh2 = self.column("omnuh2", "omega_nu_h2")
            except KeyError:
                omnuh2 = np.zeros_like(h)
            return (ombh2 + omch2 + omnuh2) / h**2

    def h0(self) -> np.ndarray:
        return self.column("H0", "hubble")

    def omega_b_h2(self) -> np.ndarray:
        return self.column("ombh2", "omega_b_h2")

    def r_drag_mpc(self) -> np.ndarray:
        return self.column("rdrag", "r_drag", "rd", "r_d")

    def h_r_drag_mpc(self) -> np.ndarray:
        """Return h*r_d in Mpc, preferring a derived-chain column when available."""
        for aliases in (
            ("hrdrag", "h_rdrag", "h_rd", "h_r_d"),
            ("H0rdrag_over_100", "H0_rd_over_100"),
        ):
            try:
                return self.column(*aliases)
            except KeyError:
                pass
        return (self.h0() / 100.0) * self.r_drag_mpc()


def _canonical(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chain_directory_url(model: str, dataset: str) -> str:
    if "/" in model or "/" in dataset or ".." in model or ".." in dataset:
        raise ValueError("model and dataset must be simple DESI directory names")
    return f"{DESI_DR2_CHAIN_ROOT}/{model}/{dataset}"


def download_file(url: str, destination: Path, *, timeout: float = 120.0) -> dict[str, object]:
    """Download one public product atomically and return provenance metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "scpc-reproducibility/1"})
    with urlopen(request, timeout=timeout) as response, pending.open("wb") as out:  # noqa: S310
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            out.write(block)
    pending.replace(destination)
    return {
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def ensure_official_chain_set(
    root: Path,
    *,
    model: str,
    dataset: str,
    chain_numbers: Iterable[int] = (1, 2, 3, 4),
    download: bool = False,
) -> tuple[list[Path], list[dict[str, object]]]:
    """Resolve the four official chain files, optionally downloading missing files."""
    target = root / model / dataset
    base_url = chain_directory_url(model, dataset)
    paths: list[Path] = []
    provenance: list[dict[str, object]] = []
    for number in chain_numbers:
        path = target / f"chain.{number}.txt"
        url = f"{base_url}/chain.{number}.txt"
        if not path.exists():
            if not download:
                raise FileNotFoundError(
                    f"Missing {path}. Re-run with official-chain download enabled; source is {url}"
                )
            provenance.append(download_file(url, path))
        else:
            provenance.append(
                {
                    "url": url,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        paths.append(path)
    return paths, provenance


def _header_names(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.lstrip().startswith("#"):
                names = tuple(line.lstrip()[1:].strip().split())
                if len(names) >= 3:
                    return names
            elif line.strip():
                break
    raise ValueError(f"No Cobaya header found in {path}")


def load_cobaya_chains(paths: Iterable[Path], *, burn_fraction: float = 0.0) -> ChainSet:
    """Read compatible Cobaya text chains and discard burn-in per chain by row count."""
    if not 0.0 <= burn_fraction < 1.0:
        raise ValueError("burn_fraction must lie in [0, 1)")
    arrays: list[np.ndarray] = []
    source_files = tuple(Path(path) for path in paths)
    if not source_files:
        raise ValueError("At least one chain file is required")
    names = _header_names(source_files[0])
    for path in source_files:
        if _header_names(path) != names:
            raise ValueError(f"Incompatible chain header in {path}")
        table = np.loadtxt(path, comments="#", ndmin=2)
        if table.shape[1] != len(names):
            raise ValueError(f"Header/data column mismatch in {path}: {len(names)} != {table.shape[1]}")
        cut = int(np.floor(burn_fraction * table.shape[0]))
        table = table[cut:]
        if table.size:
            arrays.append(table)
    if not arrays:
        raise ValueError("No posterior samples remain after burn-in removal")
    values = np.concatenate(arrays, axis=0)
    canonical = {_canonical(name): i for i, name in enumerate(names)}
    weight_index = canonical.get("weight")
    if weight_index is None:
        raise ValueError("Cobaya chain is missing the required weight column")
    weights = values[:, weight_index].astype(float, copy=True)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Invalid posterior weights")
    return ChainSet(names=names, values=values, weights=weights, source_files=source_files)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probabilities: Iterable[float]) -> np.ndarray:
    """Deterministic weighted quantiles with midpoint CDF convention."""
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    q = np.asarray(tuple(probabilities), dtype=float)
    if x.ndim != 1 or w.shape != x.shape:
        raise ValueError("values and weights must be one-dimensional with matching shape")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("values/weights must be finite and weights non-negative with positive sum")
    if np.any((q < 0) | (q > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    order = np.argsort(x)
    xs = x[order]
    ws = w[order]
    cdf = (np.cumsum(ws) - 0.5 * ws) / ws.sum()
    return np.interp(q, cdf, xs, left=xs[0], right=xs[-1])


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))


def weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    mean_x = np.average(x_arr, weights=w)
    mean_y = np.average(y_arr, weights=w)
    cov = np.average((x_arr - mean_x) * (y_arr - mean_y), weights=w)
    var_x = np.average((x_arr - mean_x) ** 2, weights=w)
    var_y = np.average((y_arr - mean_y) ** 2, weights=w)
    return float(cov / np.sqrt(var_x * var_y))
