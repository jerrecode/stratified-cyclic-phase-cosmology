#!/usr/bin/env bash
set -Eeuo pipefail

# Resilient, resumable prefetch of the exact public DESI DR2 Cobaya products
# used by scripts/reproduce_desi_dr2_bbn.py.  Size checks catch incomplete
# transfers; SHA-256 pins make the selected release products immutable inputs
# to the reproducible publication workflow.

ROOT="${1:-data/external/desi_dr2_cosmology_chains}"
BASE_URL="https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/cobaya/base"

for command in curl sha256sum; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "${command} is required for resilient DESI prefetch" >&2
    exit 2
  fi
done

# Byte sizes and SHA-256 digests were recorded from the official DESI DR2
# public chain products reproduced successfully on 2026-08-17.  The Python
# inference report independently records the same file hashes as provenance.
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

declare -A EXPECTED_SHA256=(
  ["desi-bao-all/chain.1.txt"]="fde589a17a4ca0fd58371b61290ada939dd672484cb3d523adacb8565a340ced"
  ["desi-bao-all/chain.2.txt"]="b6ce2202c5fd11049a0de5c277b017e91aa9fbc9c25d6fdce1d166e7f0ef72b7"
  ["desi-bao-all/chain.3.txt"]="fb0f68fbcd6e8704f7b7e785fcddf1efad23a32fc2c5a9fb2cc7d971e15fb697"
  ["desi-bao-all/chain.4.txt"]="ea1b353e6a6ca61172c540746f31df30c3d2db8a848936f9db654548b37605d2"
  ["desi-bao-all/chain.checkpoint"]="79d61a30d42a9ff2a8f053af63eeafd4f14f52c0a3ece303cd570e69af5bf85c"
  ["desi-bao-all/chain.covmat"]="849bc1475199dd8aa59c5cdeeb83d1043c7c90bfce461ee38d7546d35d0dec5d"
  ["desi-bao-all/chain.updated.yaml"]="20bb21e67d2e734a8a7ebd250fc99c26acbd811424898389a9d83005c87695d3"
  ["desi-bao-all_schoneberg2024-bbn/chain.1.txt"]="fcfe9bec94e3e4039a7da1890fa9cc42a3827045fbf9839976bd22d13089726d"
  ["desi-bao-all_schoneberg2024-bbn/chain.2.txt"]="77d2a7d12293e650d5284a3f5915de02e6c86c5a7bcea7479a3e05a6b50038cf"
  ["desi-bao-all_schoneberg2024-bbn/chain.3.txt"]="e0ed1e0ab3a221fc4733f572f9dc4e71d8b9d9f3cd85a24589ac8b5ba4d43f32"
  ["desi-bao-all_schoneberg2024-bbn/chain.4.txt"]="5f0e5ea871d39803c19ce4f6d7a0ca87d3bd72a7d3aac2f59246301c17cf00a6"
  ["desi-bao-all_schoneberg2024-bbn/chain.checkpoint"]="ed917379268e5c17118621c6d55a63afd5689818194a89954985412cfdf97cc1"
  ["desi-bao-all_schoneberg2024-bbn/chain.covmat"]="a1c27124552c953850e82da24c76962d257198247103e25237cb75e807af0e1b"
  ["desi-bao-all_schoneberg2024-bbn/chain.updated.yaml"]="02efb2e95afcd91cac3c90218f21a01dbf39ed7113b32f6e3f13c98c4d8ad1c0"
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

file_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

verify_final() {
  local key="$1"
  local path="$2"
  local expected_bytes="${EXPECTED_BYTES[$key]}"
  local expected_sha="${EXPECTED_SHA256[$key]}"
  local actual_bytes actual_sha
  actual_bytes="$(file_size "${path}")"
  if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
    return 1
  fi
  actual_sha="$(file_sha256 "${path}")"
  [[ "${actual_sha}" == "${expected_sha}" ]]
}

for dataset in "${DATASETS[@]}"; do
  target_dir="${ROOT}/base/${dataset}"
  mkdir -p "${target_dir}"

  for filename in "${FILES[@]}"; do
    key="${dataset}/${filename}"
    expected="${EXPECTED_BYTES[$key]}"
    expected_sha="${EXPECTED_SHA256[$key]}"
    destination="${target_dir}/${filename}"
    partial="${destination}.part"
    url="${BASE_URL}/${dataset}/${filename}"

    if [[ -f "${destination}" ]]; then
      actual="$(file_size "${destination}")"
      if verify_final "${key}" "${destination}"; then
        echo "DESI verified cache hit: ${key} (${actual} bytes, sha256=${expected_sha})"
        continue
      fi
      echo "Final file failed the pinned size/SHA-256 check; redownloading ${key}" >&2
      rm -f "${destination}" "${partial}"
    fi

    if [[ -f "${partial}" ]]; then
      partial_size="$(file_size "${partial}")"
      if (( partial_size > expected )); then
        echo "Discarding oversized partial ${key}: ${partial_size}/${expected}" >&2
        rm -f "${partial}"
      elif (( partial_size == expected )); then
        actual_sha="$(file_sha256 "${partial}")"
        if [[ "${actual_sha}" == "${expected_sha}" ]]; then
          mv "${partial}" "${destination}"
          echo "Promoted cryptographically verified cached partial: ${key}"
          continue
        fi
        echo "Discarding complete-size partial with wrong SHA-256: ${key}" >&2
        rm -f "${partial}"
      else
        echo "Resuming ${key} at ${partial_size}/${expected} bytes"
      fi
    else
      echo "Downloading ${key} (${expected} bytes)"
    fi

    # Force IPv4 because hosted CI previously timed out while establishing
    # connections to the DESI archive on some runner networks. Keep partial
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
    actual_sha="$(file_sha256 "${partial}")"
    if [[ "${actual_sha}" != "${expected_sha}" ]]; then
      echo "DESI SHA-256 mismatch for ${key}: got ${actual_sha}, expected ${expected_sha}" >&2
      rm -f "${partial}"
      exit 1
    fi
    mv "${partial}" "${destination}"
    echo "Completed and verified ${key} (${actual} bytes, sha256=${actual_sha})"
  done
done

echo "DESI DR2 chain prefetch complete and cryptographically pinned under ${ROOT}"
