#!/usr/bin/env python3
"""
Check if the current Python version meets SignFormer requirements.
Run this before installing requirements.txt to catch version issues early.
"""
import sys

MIN_MAJOR = 3
MIN_MINOR = 8

current = sys.version_info

print(f"Current Python version: {current.major}.{current.minor}.{current.micro}")

if current.major < MIN_MAJOR or (current.major == MIN_MAJOR and current.minor < MIN_MINOR):
    print(f"\n❌ ERROR: Python {MIN_MAJOR}.{MIN_MINOR}+ is required")
    print(f"   You are using Python {current.major}.{current.minor}")
    print(f"\nInstall Python 3.8:")
    print("   Windows: uv python install 3.8.20")
    print("   Linux:   apt-get install python3.8 python3.8-venv")
    print("   macOS:   brew install python@3.8")
    sys.exit(1)

print(f"\n✅ Python {current.major}.{current.minor} is compatible")
print("\nYou can now run:")
print("   pip install -r signformer_pip_requirements.txt")
sys.exit(0)
