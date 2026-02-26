#!/usr/bin/env bash

set -euo pipefail

echo "Cleaning macOS metadata files..."

# Remove AppleDouble resource fork files
find . -type f -name "._*" -print -delete

# Remove .DS_Store files
find . -type f -name ".DS_Store" -print -delete

# Remove Spotlight index folders (if present)
find . -type d -name ".Spotlight-V100" -print -exec rm -rf {} +

# Remove trash folders (if present)
find . -type d -name ".Trashes" -print -exec rm -rf {} +

echo "Done. Your repo is spiritually cleansed."