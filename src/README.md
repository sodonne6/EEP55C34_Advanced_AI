# SignFormer Environment Setup Guide

## Option 1: Create environment from conda YAML (Recommended)

```bash
conda env create -f signformer_environment.yml
conda activate signformer
```

This will handle PyTorch + CUDA and all dependencies in one go.

## Option 2: Manual setup with pip

If you prefer granular control or already have PyTorch installed:

```bash
# Create base environment
conda create -n signformer python=3.8 pip=22.2

# Activate
conda activate signformer

# Install PyTorch separately (adjust CUDA version as needed)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining packages
pip install -r signformer_requirements.txt
```

## Post-installation verification

```bash
# Activate the environment
conda activate signformer

# Test imports
python -c "
import torch
import fairseq
import torchvision
import mediapipe as mp
import sentencepiece
import pose_format
print('✓ All core imports successful')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
"

# Download NLTK data
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
print('✓ NLTK data downloaded')
"
```

## Key Notes

1. **Python 3.8 is required** - Some dependencies have strict compatibility with this version
2. **Do NOT upgrade these packages:**
   - `mediapipe==0.10.11`
   - `protobuf==3.20.1`
   - `opencv-contrib-python==4.13.0.92`
   - These versions work together to avoid JAX/JAXlib dependency hell
3. **PyTorch installation:** The environment.yml includes PyTorch with CUDA 11.8. Adjust the CUDA version if needed.
4. **Fairseq compilation:** If you encounter Cython errors when building Cython extensions, ensure you have a C compiler installed (Visual Studio Build Tools on Windows, GCC on Linux)

## Troubleshooting

### MediaPipe/Protobuf conflicts
If you get import errors with mediapipe, ensure protobuf is at the pinned version:
```bash
pip install --force-reinstall protobuf==3.20.1
```

### Fairseq build errors
If `fairseq` fails to compile Cython extensions:
```bash
pip install -U pip wheel setuptools cython
pip install --no-build-isolation fairseq
```

### CUDA not found on Windows
Make sure CUDA Toolkit 11.8+ is installed on your system. You can still run CPU-only:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Activating the environment

Every time you want to use SignFormer:
```bash
conda activate signformer
```

To deactivate:
```bash
conda deactivate
```
