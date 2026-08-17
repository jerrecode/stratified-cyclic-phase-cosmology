#!/usr/bin/env bash
set -Eeuo pipefail

# Resilient, resumable prefetch of the public DESI DR2 Cobaya chains used by
# scripts/reproduce_desi_dr2_bbn.py. The Python reproduction remains the
# authoritative parser/checksummer; this helper only makes hosted CI less
# sensitive to intermittent connectivity to data.desi.lbl.gov.

ROOT="${1:-data/external/desi_dr2_cosmology_chains}"
BASE_URL="https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/cobaya/base"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for resilient DESI prefetch" >&2
  exit 2
fi

# Byte sizes are pinned from the public DESI directory indexes dated
# 2025-08-25. They provide a cheap completeness check before the Python
# pipeline computes and records SHA-256 digests of every completed file.
declare -A EXPECTED_BYTES=(
  ["desi-bao-all/chain.1.txt"]=8385894
  ["desi-bao-all/chain.2.txt"]=8091391
  ["desi-bao-all/chain.3.txt"]=8095450
  ["desi-bao-all/chain.4.txt"]=8876131
  ["desi-bao-all/chain.checkpoint"]=108
  ["desi-bao-all/chain.covmat"]=115
  ["desi-bao-all/chain.updated.yaml"]=4396
  ["desi-bao-all_schoneberg2024-bbn/chain.1.txt"]=20477032
  ["desi-bao-all_schoneberg2024-bbn/chain.2.txt"]=21783682
  ["desi-bao-all_schoneberg2024-bbn/chain.3.txt"]=20106674
  ["desi-bao-all_schoneberg2024-bbn/chain.4.txt"]=20838960
  ["desi-bao-all_schoneberg2024-bbn/chain.checkpoint"]=108
  ["desi-bao-all_schoneberg2024-bbn/chain.covmat"]=242
  ["desi-bao-all_schoneberg2024-bbn/chain.updated.yaml"]=4891
)

DATASETS=(
  "desi-bao-all"
  "desi-bao-all_schoneberg2024-bbn"
)
FILES=(
  "chain.1.txt"
  "chain.2.txt"
  "chain.3.txt"
  "chain.4.txt"
  "chain.checkpoint"
  "chain.covmat"
  "chain.updated.yaml"
)

file_size() {
  python - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).stat().st_size)
PY
}

for dataset in "${DATASETS[@]}"; do
  target_dir="${ROOT}/base/${dataset}"
  mkdir -p "${target_dir}"

  for filename in "${FILES[@]}"; do
    key="${dataset}/${filename}"
    expected="${EXPECTED_BYTES[$key]}"
    destination="${target_dir}/${filename}"
    partial="${destination}.part"
    url="${BASE_URL}/${dataset}/${filename}"

    if [[ -f "${destination}" ]]; then
      actual="$(file_size "${destination}")"
      if [[ "${actual}" == "${expected}" ]]; then
        echo "DESI cache hit: ${key} (${actual} bytes)"
        continue
      fi
      echo "Final file has unexpected size; preserving it as resumable partial: ${key} ${actual}/${expected}" >&2
      rm -f "${partial}"
      mv "${destination}" "${partial}"
    fi

    if [[ -f "${partial}" ]]; then
      partial_size="$(file_size "${partial}")"
      if (( partial_size > expected )); then
        echo "Discarding oversized partial ${key}: ${partial_size}/${expected}" >&2
        rm -f "${partial}"
      elif (( partial_size == expected )); then
        mv "${partial}" "${destination}"
        echo "Promoted complete cached partial: ${key} (${expected} bytes)"
        continue
      else
        echo "Resuming ${key} at ${partial_size}/${expected} bytes"
      fi
    else
      echo "Downloading ${key} (${expected} bytes)"
    fi

    # Force IPv4 because hosted CI has repeatedly timed out before establishing
    # a connection to the DESI archive on some runner networks. Keep partial
    # files for the next retry/workflow run and use HTTP Range resume via -C -.
    curl \
      --ipv4 \
      --fail \
      --location \
      --silent \
      --show-error \
      --connect-timeout 20 \
      --max-time 420 \
      --retry 10 \
      --retry-all-errors \
      --retry-delay 3 \
      --retry-max-time 900 \
      --continue-at - \
      --output "${partial}" \
      "${url}"

    actual="$(file_size "${partial}")"
    if [[ "${actual}" != "${expected}" ]]; then
      echo "Incomplete DESI download retained for resume: ${key} ${actual}/${expected}" >&2
      exit 1
    fi
    mv "${partial}" "${destination}"
    echo "Completed ${key} (${actual} bytes)"
  done
done

echo "DESI DR2 chain prefetch complete under ${ROOT}"
