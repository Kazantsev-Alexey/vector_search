#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data/hw2"
ARCHIVE="${DATA_DIR}/laion-small-clip.tgz"
URL="https://storage.googleapis.com/ann-filtered-benchmark/datasets/laion-small-clip.tgz"

mkdir -p "${DATA_DIR}"

if [ -f "${DATA_DIR}/vectors.npy" ]; then
  echo "vectors.npy already exists"
  exit 0
fi

if [ ! -f "${ARCHIVE}" ]; then
  echo "Downloading"
  curl -L "${URL}" -o "${ARCHIVE}"
fi

echo "Unpacking"
tar -xzf "${ARCHIVE}" -C "${DATA_DIR}"

echo "Done"