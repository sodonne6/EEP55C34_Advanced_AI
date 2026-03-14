# EEP55C34 Advanced AI

## ASL to Speech - Group Project

This model intends to convert Continious ASL to Synthesised Speech

## Pipeline

### Original Architecture 

The original approach was inspired by SignFormer-GCN

![Architecture Diagram](https://github.com/sodonne6/EEP55C34_Advanced_AI/blob/main/project_notes/signformer_paper_diagram.png?raw=1)

### New Approach

The approach taken leans heavier on GCN influence by adding two more GCN paths which process the right/left hand coordinates extracted by mediapipe

![New Architecture Diagram](https://github.com/sodonne6/EEP55C34_Advanced_AI/blob/main/project_notes/signformer_architecture.png?raw=1)


## Datasets

How2Sign

## Team Members

Farida Shittu & Shane O'Donnell

## Installation

Clone repo locally or in colab

```bash
git clone --recurse-submodules https://github.com/sodonne6/EEP55C34_Advanced_AI.git
cd EEP55C34_Advanced_AI/project
```

Clone Submodules

```bash
git submodule update --init --recursive
```

## References

[1] S. H. Arib, R. Akter, S. Rahman, and S. Rahman, "SignFormer-GCN: Continuous sign language translation using spatio-temporal graph convolutional networks," PLoS ONE, vol. 20, no. 2, p. e0316298, Feb. 2025, doi: 10.1371/journal.pone.0316298.

[2] A. Duarte et al., “How2Sign: a large-scale multimodal dataset for continuous American sign language,” arXiv.org, Aug. 18, 2020. https://arxiv.org/abs/2008.08143


