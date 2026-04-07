# SignFormer Environment Setup Guide

This setup guide covers both the provided Conda environment and a pip-based setup for users who do not use Conda.

```bash
cd into/src/folder
```


## Conda Environment File

Use the repository's `signformer_environment.yml` file:

```bash
conda env create -f signformer_environment.yml
conda activate signformer
```

## Windows Prerequisite (Important for PyTorch)

On Windows, install **Microsoft Visual C++ Redistributable 2015-2022 (x64)** before running the app.

- x86 only is not sufficient for this project.
- Missing x64 runtime can cause:
  - `OSError: [WinError 182] ... torch\\lib\\shm.dll` when recording/inference starts.

Install link:
- https://aka.ms/vs/17/release/vc_redist.x64.exe

Optional (PowerShell with winget):

```powershell
winget install Microsoft.VCRedist.2015+.x64
```

After install, restart terminal/session and re-run:

```powershell
python -c "import torch; print(torch.__version__)"
```

## Python Version Requirements

**IMPORTANT: Python 3.8 is required** for all non-Conda installations.

### Step 1: Verify/Install Python 3.8

First, check if you have Python 3.8 installed:

```bash
python3.8 --version
```

If not installed or you see an error, install it:
- **Windows**: Download from [python.org/downloads/release/python-3820/](https://www.python.org/downloads/release/python-3820/) or use `uv python install 3.8.20` or use `winget install Python.Python.3.8`

- **Linux (Ubuntu/Debian)**: `sudo apt-get update && sudo apt-get install python3.8 python3.8-venv`
- **Linux (Fedora/RHEL)**: `sudo dnf install python3.8`
- **macOS**: `brew install python@3.8`

### Step 2: Verify the installation

Run the version check script:

```bash
python3.8 check_python_version.py
```

Expected output:
```
Current Python version: 3.8.20
✅ Python 3.8 is compatible

You can now run:
   pip install -r signformer_pip_requirements.txt
```

If you see `python3.8: command not found`, Python 3.8 is not installed. Follow the install instructions above and try again.

## Non-Conda Environment File

Choose based on your hardware and OS. After choosing, **first verify Python 3.8 is installed** using the steps above.

### Windows with NVIDIA GPU (CUDA 11.3)

**Requires Python 3.8. After installing Python 3.8, run:**

```bash
python3.8 -m venv .venv
.venv\Scripts\activate
pip install -r signformer_pip_requirements.txt
```

Alternatively, use `uv` to handle Python version automatically:

```bash
uv python install 3.8.20
uv venv --python 3.8.20 .venv
.\.venv\Scripts\activate
uv pip install -r signformer_pip_requirements.txt
```

### Windows without GPU (CPU-only)

**Requires Python 3.8. After installing Python 3.8, run:**

```bash
python3.8 -m venv .venv
.venv\Scripts\activate
pip install -r signformer_pip_requirements_cpu.txt
```

Alternatively, use `uv`:

```bash
uv python install 3.8.20
uv venv --python 3.8.20 .venv
.\.venv\Scripts\activate
uv pip install -r signformer_pip_requirements_cpu.txt
```

### Linux / WSL (CPU-only)

**Requires Python 3.8+. After installing Python 3.8 (see above), run:**

```bash
python3.8 -m venv .venv
. .venv/bin/activate
pip install -r signformer_linux_requirements.txt
```


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
- Conda setup already installs Python 3.8 from `signformer_environment.yml`; global Python 3.8 is only required for non-Conda (`venv`) setup.
- On Windows, ensure Microsoft VC++ 2015-2022 **x64** runtime is installed (x86-only installs can fail when loading PyTorch DLLs).
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
