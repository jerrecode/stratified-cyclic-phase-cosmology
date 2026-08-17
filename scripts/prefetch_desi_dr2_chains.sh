#!/usr/bin/env bash
set -Eeuo pipefail

# Resilient, resumable prefetch of the exact public DESI DR2 Cobaya products
# used by scripts/reproduce_desi_dr2_bbn.py. Size checks catch incomplete
# transfers; SHA-256 pins make the selected release products immutable inputs
# to the reproducible publication workflow.

ROOT="${1:-data/external/desi_dr2_cosmology_chains}"
PIN_MANIFEST="${2:-configs/inference/desi_dr2_chain_pins.json}"

for command in curl sha256sum python; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "${command} is required for resilient DESI prefetch" >&2
    exit 2
  fi
done

if [[ ! -f "${PIN_MANIFEST}" ]]; then
  echo "DESI pin manifest not found: ${PIN_MANIFEST}" >&2
  exit 2
fi

BASE_URL="$(python - "${PIN_MANIFEST}" <<'PY'
import json
from pathlib import Path
import sys
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("schema_version") != 1:
    raise SystemExit("unsupported DESI pin-manifest schema_version")
base_url = manifest.get("base_url")
if not isinstance(base_url, str) or not base_url.startswith("https://data.desi.lbl.gov/"):
    raise SystemExit("unexpected DESI pin-manifest base_url")
print(base_url.rstrip("/"))
PY
)"

manifest_rows="$(python - "${PIN_MANIFEST}" <<'PY'
import json
from pathlib import Path
import re
import sys
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
products = manifest.get("products")
if not isinstance(products, list) or not products:
    raise SystemExit("DESI pin manifest contains no products")
seen = set()
for product in products:
    dataset = product.get("dataset")
    filename = product.get("filename")
    size = product.get("bytes")
    digest = product.get("sha256")
    if not isinstance(dataset, str) or not dataset or "/" in dataset or ".." in dataset:
        raise SystemExit(f"invalid DESI dataset name: {dataset!r}")
    if not isinstance(filename, str) or not filename or "/" in filename or ".." in filename:
        raise SystemExit(f"invalid DESI filename: {filename!r}")
    key = (dataset, filename)
    if key in seen:
        raise SystemExit(f"duplicate DESI pin: {dataset}/{filename}")
    seen.add(key)
    if not isinstance(size, int) or size <= 0:
        raise SystemExit(f"invalid byte length for {dataset}/{filename}")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise SystemExit(f"invalid SHA-256 for {dataset}/{filename}")
    print(f"{dataset}\t{filename}\t{size}\t{digest}")
PY
)"

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
  local path="$1"
  local expected_bytes="$2"
  local expected_sha="$3"
  local actual_bytes actual_sha
  actual_bytes="$(file_size "${path}")"
  if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
    return 1
  fi
  actual_sha="$(file_sha256 "${path}")"
  [[ "${actual_sha}" == "${expected_sha}" ]]
}

while IFS=$'\t' read -r dataset filename expected expected_sha; do
  [[ -n "${dataset}" ]] || continue
  key="${dataset}/${filename}"
  target_dir="${ROOT}/base/${dataset}"
  mkdir -p "${target_dir}"
  destination="${target_dir}/${filename}"
  partial="${destination}.part"
  url="${BASE_URL}/${dataset}/${filename}"

  if [[ -f "${destination}" ]]; then
    actual="$(file_size "${destination}")"
    if verify_final "${destination}" "${expected}" "${expected_sha}"; then
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
done <<< "${manifest_rows}"

echo "DESI DR2 chain prefetch complete and cryptographically pinned under ${ROOT}"
