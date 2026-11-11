#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Download model weights for SwiftTailor inference.
# Weights are distributed via GitHub Releases.

set -euo pipefail

WEIGHTS_DIR="weights"
RELEASE_URL="https://github.com/Qualcomm-AI-research/garment-gen/releases/download/v1.0.0"

# --- PatternMaker + GarmentSewer weights ---
if [ -f "$WEIGHTS_DIR/patternmaker/model.safetensors" ] && [ -f "$WEIGHTS_DIR/garmentsewer/dpt.pt" ]; then
    echo "SwiftTailor weights already exist. Skipping."
else
    echo "Downloading SwiftTailor weights (~8.6GB)..."
    for part in weights.tar.part-0{0..4}; do
        curl -fL --retry 3 -C - -o "$part" "$RELEASE_URL/$part"
    done
    echo "Extracting weights..."
    cat weights.tar.part-* | tar -xf -
    rm -f weights.tar.part-*
    echo "Weights extracted to $WEIGHTS_DIR/"
fi

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
