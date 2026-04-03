# SignFormer Environment Setup Guide

This setup guide covers both the provided Conda environment and a pip-based setup for users who do not use Conda.

## Conda Environment File

Use the repository's `signformer_environment.yml` file:

```bash
conda env create -f signformer_environment.yml
conda activate signformer
```

## Non-Conda Environment File

Choose based on your hardware and OS:

### Windows with NVIDIA GPU (CUDA 11.3)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r signformer_pip_requirements.txt
```

For `uv`, install and use Python 3.8 explicitly:

```bash
uv python install 3.8.20
uv venv --python 3.8.20 .venv
.\.venv\Scripts\activate
uv pip install -r signformer_pip_requirements.txt
```

### Windows without GPU (CPU-only)

For machines without NVIDIA GPUs, use the CPU-only variant (slower but works):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r signformer_pip_requirements_cpu.txt
```

With `uv`:

```bash
uv python install 3.8.20
uv venv --python 3.8.20 .venv
.\.venv\Scripts\activate
uv pip install -r signformer_pip_requirements_cpu.txt
```

### Linux / WSL (CPU-only)

Linux and WSL use CPU-only PyTorch wheels by default (GPU support requires additional CUDA setup):

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r signformer_linux_requirements.txt
```

### macOS

Not yet supported with current pinned packages. A separate macOS requirements file would be needed.


## Post-installation Verification

```bash
python -c "
import torch
import torchvision
import torchaudio
import mediapipe as mp
import sentencepiece
import pose_format
print('✓ All core imports successful')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

## Download the NLTK deps

```bash
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
print('✓ NLTK data downloaded')
"
```

## Important Notes
- Python 3.8 is required.
- GPU variant uses CUDA 11.3 PyTorch wheels; CPU variant uses CPU-only wheels.
- CPU-only runs slower but is compatible with any machine without GPU hardware.
- Keep these package versions pinned:
  - mediapipe==0.10.11
  - protobuf==3.20.1
  - opencv-contrib-python==4.10.0.84 (CPU and Linux variants) or 4.13.0.92 (GPU variant)
- The verified demo setup does not require fairseq to be installed as a standalone pip package.
- The demo resolves the main repo paths automatically when the repository layout is unchanged.

## Running the System End-to-End

```bash
conda activate signformer
cd path\to\your\project\root
```

### Clear any outdated overrides

```bash
Remove-Item Env:SLT_DATA_DIR -ErrorAction SilentlyContinue
```
### Set runtime options

```bash

$env:SLT_DEVICE="cpu"
$env:SLT_BEAM="1"
$env:SLT_MAX_LEN_B="24"
$env:SLT_RECORD_MAX_FRAMES="40"
$env:SLT_RECORD_EVERY_N="4"
$env:SLT_STORE_MAX_SIDE="160"
$env:SLT_PRESTACK_MAX_FRAMES="24"
$env:SLT_MAX_INPUT_FRAMES="24"
$env:SLT_MIN_INPUT_FRAMES="12"
$env:SLT_DRAW_LANDMARKS="1"
$env:SLT_ENABLE_TTS="1"
$env:SLT_TTS_DEVICE="cpu"
$env:SLT_TTS_SPEAKER_ID="7306"
```

### Run script

```bash
python -u .\src\app_preview_demo.py
```

### Mediapipe overlay toggle

If you want mediapipe keypoints overlayed on the live video:

```bash 
$env:SLT_DRAW_LANDMARKS="1"
```

### TTS Toggle

To enable voice synthesis:

```bash
$env:SLT_ENABLE_TTS="1"
```
## Full Example Run Script

```bash
conda activate signformer
cd "C:\root\to\your\dir\EEP55C34_Advanced_AI"

Remove-Item Env:SLT_DATA_DIR -ErrorAction SilentlyContinue

$env:SLT_DEVICE="cpu"
$env:SLT_BEAM="1"
$env:SLT_MAX_LEN_B="24"
$env:SLT_RECORD_MAX_FRAMES="40"
$env:SLT_RECORD_EVERY_N="4"
$env:SLT_STORE_MAX_SIDE="160"
$env:SLT_PRESTACK_MAX_FRAMES="24"
$env:SLT_MAX_INPUT_FRAMES="24"
$env:SLT_MIN_INPUT_FRAMES="12"
$env:SLT_DRAW_LANDMARKS="1"
$env:SLT_ENABLE_TTS="1"
$env:SLT_TTS_DEVICE="cpu"
$env:SLT_TTS_SPEAKER_ID="7306"

python -u .\src\app_preview_demo.py
```
