#!/bin/bash
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
#
# Single-GPU benchmark :
#   Stage A: gen_patternmaker_testset.py (This can be run parralelly on a single GPU, 13 GB VRAM/threads, change NSHARDS)
#   Stage B: gen_garmentsewer_testset.py once, over the merged Stage-A output.

set -euo pipefail

REPO="/PATH/TO/SWIFTTAILOR_REPO"
CONDA_ENV="${CONDA_ENV:-swifttailor}"
OUT="${OUT:-$REPO/output/benchmark}"
LIST="${LIST:-$REPO/configs/testset_500samples_full_pipeline.txt}"
NSHARDS="${NSHARDS:-2}"
BASE_PORT="${BASE_PORT:-29500}"

cd "$REPO"
mkdir -p "$OUT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
PY="$(command -v python)"

echo "[bench] repo=$REPO out=$OUT nshards=$NSHARDS list=$LIST"
echo "[bench] conda env=$CONDA_ENV python=$PY"

# --- Split the sample list into $NSHARDS interleaved pieces ---
shard_dir="$OUT/_shards"
mkdir -p "$shard_dir"
"$PY" - "$LIST" "$shard_dir" "$NSHARDS" <<'PYEOF'
import sys
list_path, shard_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = [l.strip() for l in open(list_path) if l.strip()]
for i in range(n):
    with open(f"{shard_dir}/shard{i}.txt", "w") as f:
        f.write("\n".join(lines[i::n]))
    print(f"[bench] shard {i}: {len(lines[i::n])} samples")
PYEOF

# --- Stage A: launch $NSHARDS gen_patternmaker_testset.py in parallel ---
pids=()
for ((i=0; i<NSHARDS; i++)); do
    shard_out="$OUT/shard${i}"
    shard_log="$OUT/shard${i}.log"
    port=$((BASE_PORT + i))
    mkdir -p "$shard_out"
    (
        MASTER_PORT="$port" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        "$PY" scripts/benchmark/gen_patternmaker_testset.py \
            --sample-list "$shard_dir/shard${i}.txt" \
            --output-dir "$shard_out" \
            --python "$PY" \
            > "$shard_log" 2>&1
    ) &
    pids+=($!)
    echo "[bench] shard ${i} pid=${pids[-1]} port=${port} log=${shard_log}"
done

fail=0
for p in "${pids[@]}"; do
    if ! wait "$p"; then
        echo "[bench][error] shard pid=$p failed" >&2
        fail=$((fail + 1))
    fi
done
if (( fail > 0 )); then
    echo "[bench][fatal] $fail/$NSHARDS shard(s) failed -- see $OUT/shard*.log" >&2
    exit 1
fi
echo "[bench] Stage A done."

# --- Merge shard outputs into $OUT/patternmaker/ via symlinks ---
merged="$OUT/patternmaker"
mkdir -p "$merged"
for ((i=0; i<NSHARDS; i++)); do
    src_dir="$OUT/shard${i}/patternmaker"
    [ -d "$src_dir" ] || continue
    for d in "$src_dir"/*/; do
        [ -d "$d" ] || continue
        leaf=$(basename "$d")
        ln -sfn "$d" "$merged/$leaf"
    done
done
n_merged=$(find "$merged" -maxdepth 1 -mindepth 1 | wc -l)
echo "[bench] merged $n_merged leaves into $merged"

# --- Stage B: garmentsewer + metrics over the merged view ---
"$PY" scripts/benchmark/gen_garmentsewer_testset.py \
    --patternmaker-dir "$merged" \
    --output-dir "$OUT"

echo "[bench] done."
