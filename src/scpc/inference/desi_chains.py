"""Download, validate, and analyse the official DESI DR2 cosmology chains.

The released DESI chains remain external products. Publication reproduction records
both the sample files and the release-side metadata/checkpoint products, and uses
GetDist for convergence diagnostics and marginalized densities.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

DESI_DR2_CHAIN_ROOT = "https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/cobaya"
# The DESI chain data model documents these products for standard (non-post-processed)
# Cobaya chains. It documents input.yaml *or* updated.yaml; the DR2 products used here
# provide the latter, so we do not invent auxiliary files such as margestats/progress.
DESI_CHAIN_METADATA = (
    "chain.checkpoint",
    "chain.covmat",
    "chain.updated.yaml",
)


@dataclass(frozen=True)
class ChainSet:
    """In-memory Cobaya chain table with per-sample weights."""

    names: tuple[str, ...]
    values: np.ndarray
    weights: np.ndarray
    source_files: tuple[Path, ...]

    def column(self, *aliases: str) -> np.ndarray:
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


def download_file(
    url: str,
    destination: Path,
    *,
    timeout: float = 30.0,
    retries: int = 6,
) -> dict[str, object]:
    """Download one public product atomically with retry and HTTP Range resume.

    The DESI public archive is occasionally slow from hosted CI regions. Partial data are
    kept only in a ``.part`` file and retried with a Range request; the final path appears
    only after EOF has been reached successfully. The returned checksum is always computed
    over the completed local file.
    """
    if timeout <= 0 or retries < 1:
        raise ValueError("timeout must be positive and retries must be at least one")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        offset = pending.stat().st_size if pending.exists() else 0
        headers = {
            "User-Agent": "scpc-reproducibility/3",
            "Accept-Encoding": "identity",
        }
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                status = int(getattr(response, "status", response.getcode()))
                resumed = offset > 0 and status == 206
                mode = "ab" if resumed else "wb"
                with pending.open(mode) as out:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        out.write(block)
            pending.replace(destination)
            metadata = _file_metadata(destination, url)
            metadata["download_attempts"] = attempt
            metadata["resumed"] = bool(offset)
            return metadata
        except HTTPError as exc:
            last_error = exc
            if exc.code == 416 and pending.exists():
                pending.unlink()
            if 400 <= exc.code < 500 and exc.code not in {408, 416, 429}:
                raise RuntimeError(f"Permanent HTTP error downloading {url}: {exc}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc

        if attempt < retries:
            time.sleep(min(2.0 ** (attempt - 1), 16.0))

    raise RuntimeError(
        f"Failed to download {url} after {retries} attempts; partial file retained at {pending}"
    ) from last_error


def _file_metadata(path: Path, url: str) -> dict[str, object]:
    return {
        "url": url,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def ensure_official_chain_set(
    root: Path,
    *,
    model: str,
    dataset: str,
    chain_numbers: Iterable[int] = (1, 2, 3, 4),
    download: bool = False,
) -> tuple[list[Path], list[dict[str, object]]]:
    """Resolve the official chain sample files, optionally downloading missing files."""
    target = root / model / dataset
    base_url = chain_directory_url(model, dataset)
    paths: list[Path] = []
    provenance: list[dict[str, object]] = []
    for number in chain_numbers:
        path = target / f"chain.{number}.txt"
        url = f"{base_url}/chain.{number}.txt"
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; official source is {url}")
            provenance.append(download_file(url, path))
        else:
            provenance.append(_file_metadata(path, url))
        paths.append(path)
    return paths, provenance


def ensure_official_chain_metadata(
    root: Path,
    *,
    model: str,
    dataset: str,
    download: bool = False,
) -> list[dict[str, object]]:
    """Resolve documented release-side chain metadata with checksums."""
    target = root / model / dataset
    base_url = chain_directory_url(model, dataset)
    provenance: list[dict[str, object]] = []
    for filename in DESI_CHAIN_METADATA:
        path = target / filename
        url = f"{base_url}/{filename}"
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; official source is {url}")
            provenance.append(download_file(url, path))
        else:
            provenance.append(_file_metadata(path, url))
    return provenance


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
    """Read compatible Cobaya text chains and discard a declared fraction per chain."""
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


def _import_getdist():
    try:
        from getdist import loadMCSamples  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("GetDist is required; install the 'inference' extra") from exc
    return loadMCSamples


def load_getdist_samples(directory: Path, *, burn_fraction: float = 0.0):
    """Load the official Cobaya chain root using GetDist's native Cobaya support."""
    if not 0.0 <= burn_fraction < 1.0:
        raise ValueError("burn_fraction must lie in [0, 1)")
    loader = _import_getdist()
    root = str(Path(directory) / "chain")
    return loader(root, no_cache=True, settings={"ignore_rows": float(burn_fraction)})


def select_converged_burn_fraction(
    directory: Path,
    *,
    candidates: Iterable[float] = (0.0, 0.05, 0.10, 0.20, 0.30),
    max_rminus1: float = 0.01,
) -> tuple[float, object, dict[str, object]]:
    """Choose the earliest burn cut whose GetDist multivariate R-1 passes the declared threshold.

    This replaces an unexplained fixed row cut with an auditable convergence rule. The
    official ``chain.checkpoint`` is retained in provenance; GetDist independently checks
    the released sample files used for the plotted posterior.
    """
    trials: list[dict[str, float | int]] = []
    selected = None
    selected_samples = None
    for candidate in candidates:
        fraction = float(candidate)
        samples = load_getdist_samples(directory, burn_fraction=fraction)
        rminus1 = float(samples.getGelmanRubin())
        weights = np.asarray(samples.weights, dtype=float)
        weighted_ess = float(weights.sum() ** 2 / np.dot(weights, weights))
        trial = {
            "burn_fraction": fraction,
            "getdist_rminus1": rminus1,
            "rows": int(samples.numrows),
            "weighted_ess": weighted_ess,
        }
        trials.append(trial)
        if np.isfinite(rminus1) and rminus1 <= max_rminus1:
            selected = fraction
            selected_samples = samples
            break
    if selected_samples is None or selected is None:
        best = min(trials, key=lambda row: float(row["getdist_rminus1"]))
        raise RuntimeError(
            "Official chains failed the predeclared GetDist convergence threshold "
            f"R-1 <= {max_rminus1}; best trial was {best}"
        )
    diagnostics = {
        "selection_rule": "earliest candidate with GetDist multivariate R-1 <= threshold",
        "max_rminus1": float(max_rminus1),
        "selected_burn_fraction": selected,
        "trials": trials,
        "getdist_convergence_summary": selected_samples.getConvergeTests(
            what=("MeanVar", "GelmanRubin", "SplitTest", "CorrLengths")
        ),
    }
    return selected, selected_samples, diagnostics


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probabilities: Iterable[float]) -> np.ndarray:
    """Deterministic weighted quantiles with midpoint-CDF convention."""
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
