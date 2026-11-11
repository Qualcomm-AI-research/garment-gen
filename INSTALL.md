# Installation Guide

This guide covers installing the complete SwiftTailor environment including:
- SwiftTailor core package (PatternMaker + GarmentSewer)
- NVIDIA Warp (custom fork for cloth simulation)
- GarmentCode / PyGarment (parametric garment framework)

## Prerequisites

- Linux (tested on Ubuntu 20.04+)
- NVIDIA GPU with CUDA support (tested on H100, A100)
- CUDA Toolkit 11.5+ installed system-wide (`nvcc` available in PATH)
- GCC 7.2+ (for building Warp native library)
- conda (Miniconda or Anaconda)
- Git

## Step 1: Create Conda Environment

```bash
conda create -n swifttailor python=3.10 -y
conda activate swifttailor
```

## Step 2: Install PyTorch with CUDA

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124
```

Verify:
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Step 3: Install Python Dependencies

Install PyTorch (Step 2) **before** this step: pytorch3d builds against `torch` and
needs it importable at build time, so pass `--no-build-isolation`
(see [facebookresearch/pytorch3d#1892](https://github.com/facebookresearch/pytorch3d/issues/1892)).

```bash
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    --no-build-isolation  # For benchmarking
pip install -r requirements.txt
```

## Step 4: Install GarmentCode (PyGarment)

GarmentCode is the parametric garment programming framework used for cloth simulation.

```bash
git clone https://github.com/maria-korosteleva/GarmentCode.git
cd GarmentCode
pip install -e .

cd ..
```

Verify:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/GarmentCode # Always export GarmentCode into PythonPath
python -c "from pygarment.data_config import Properties; print('PyGarment OK')"
python -c "from pygarment.meshgen.boxmeshgen import BoxMesh; print('BoxMesh OK')"
```

## Step 5: Install SwiftTailor Package

```bash
pip install -e .
```

Verify:
```bash
python -c "from swifttailor.garmentsewer.dpt import DPTGIM; print('GarmentSewer OK')"
python -c "from swifttailor.patternmaker.st_model import SwiftTailorForCausalLM; print('PatternMaker OK')"
```

## Step 6: Download Weights and Data

```bash
# Download model weights
bash scripts/download_weights.sh
```

Expected layout:

```
weights/
├── patternmaker/
│   ├── config.json
│   └── model.safetensors
└── garmentsewer/
    └── dpt.pt
```

## Step 7: Install NVIDIA Warp (Optional)

We can skip this if we do not need to reproduce PatternMaker + GarmentCode results. GarmentCode uses a custom fork of NVIDIA Warp with cloth simulation extensions.

```bash
# Clone the custom NvidiaWarp from GarmentCode
sudo apt-get install git-lfs # Install git-lfs first
git clone https://github.com/maria-korosteleva/NvidiaWarp-GarmentCode.git

# Build native library (requires CUDA toolkit and GCC)
cd NvidiaWarp-GarmentCode
export CUDA_PATH=/usr/local/cuda  # adjust if CUDA is elsewhere
python build_lib.py
pip install -e .
cd ..
```

Verify:
```bash
python -c "import warp as wp; wp.init(); print(f'Warp {wp.config.version} - OK')"
```

Expected output shows available CUDA devices and Warp version.

> **Note:** The `build_lib.py` step compiles CUDA kernels and takes ~2 minutes. If you see
> "Warp Clang/LLVM build error" at the end, this is non-critical — the main `warp.so` library
> was already built successfully.


## Troubleshooting

| Issue | Solution |
|-------|----------|
| `pygarment` not found | After installing GarmentCode, `export PYTHONPATH=$PYTHONPATH:$(pwd)/GarmentCode`   |
| `nvcc` not found | Install CUDA Toolkit and set `export CUDA_PATH=/path/to/cuda` |
| `warp.so` build fails | Ensure GCC 7.2+ and CUDA 11.5+ are available |
| `git-lfs: not found` during Warp clone | Install git-lfs to complete cloning the repository |
| `packman` permission denied | Set permission for `NvidiaWarp-GarmentCode/tools/packman/packman` to be executable |
| `numpy<2` conflict | GarmentCode requires numpy<2; this is pinned in requirements.txt |
| `flash_attn` build error | Optional; PatternMaker falls back to standard attention |
| `deepspeed` not found | `pip install deepspeed` (needed for checkpoint loading) |
| opencv-python numpy incompatibility warning | We pin opencv-python==4.10.0.84 which supports numpy<2 |

## Dataset Download
Our benchmark is run on [GarmentCodeDataset](https://igl.ethz.ch/projects/GarmentCodeData/) and the text generated in [GCD-MM](https://huggingface.co/georgeNakayama/AIpparel) from [AIpparel](https://github.com/georgeNakayama/AIpparel-Code). The list of test samples can be found at [`testset_500samples_full_pipeline.txt`](configs/testset_500samples_full_pipeline.txt), one `<garments_dir>/default_body/<sample_id>` entry per line. Our testset data folder with compulsory files is structured as follows:

```
data/test_set/
└── garments_5000_X/                                       # garments directory (one per data shard)
    └── default_body/
        └── rand_XXXXXXXXX/                                     # sample_id
            ├── captions.json                                   # GCD-MM generated text captions (description, occasion, etc.)
            ├── rand_XXXXXXXXXXX_sim.ply                        # groundtruth - simulated garment mesh (Warp cloth sim output)
            ├── rand_XXXXXXXXXXX_render_front.png               # rendered images
            └── rand_XXXXXXXXXXX_render_back.png                    
```

