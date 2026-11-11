#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Download model weights for SwiftTailor inference.
# Weights are distributed via GitHub Releases.

set -euo pipefail

WEIGHTS_DIR="weights"
mkdir -p "$WEIGHTS_DIR/patternmaker" "$WEIGHTS_DIR/garmentsewer"

# --- PatternMaker weights ---
# TODO: update here later


# --- GarmentSewer weights ---
# TODO: update here later

# --- Base InternVL3-2B tokenizer + source (required by PatternMaker) ---
# Download the tokenizer and the InternVL source modules come from the base model.
# Fetch only the tokenizer + source/config files (no *.safetensors)
if [ -f "cache/InternVL3-2B-Instruct/tokenizer_config.json" ]; then
    echo "InternVL3-2B tokenizer already exists. Skipping."
else
    echo "Downloading InternVL3-2B tokenizer + source files..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='OpenGVLab/InternVL3-2B-Instruct',
    local_dir='cache/InternVL3-2B-Instruct',
    allow_patterns=['*.json', '*.py', '*.txt'],
)
"
    echo "InternVL3-2B tokenizer saved to cache/InternVL3-2B-Instruct/"
fi

echo "All weights downloaded successfully."
