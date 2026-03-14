# EEP55C34 Advanced AI

## ASL to Speech - Group Project

This model intends to convert Continious ASL to Synthesised Speech

## Pipeline

### Original Archtecture 

The original approach was inspired by SignFormer-GCN

![Architecture Diagram](https://github.com/sodonne6/EEP55C34_Advanced_AI/blob/main/project_notes/signformer_paper_diagram.png?raw=1)

### Original Approach

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



