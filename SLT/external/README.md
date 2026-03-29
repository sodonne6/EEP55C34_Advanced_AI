# External SignFormer GCN Training Reference

This folder contains the **original SignFormer GCN codebase** used as the base training repository for this project.

## Purpose of this folder

The external repo is kept here mainly for:

- the original training framework
- model architecture reference
- config structure
- baseline training/inference scripts

It is the upstream base used during development, but for this project it should be treated as a **starting point rather than a drop-in training solution**.

## Important note

For the current project setup, the dataset and model scripts in the original repo are **buggy / incompatible** with the manifests and feature format used here.

In practice, if you want to train a new model, you should **not rely on the untouched scripts in this external repo**.

Instead, use the project-specific fixes stored in:

```text
SLT/signformer_overrides/
```

## 1. Choose correct override set

Pick the override files associated with the archiceture you want to train

### Baseline
- sign_features_dataset.py
- sign2text_transformer.py
- graph.py

### 3-GCN I3D + MediaPipe
- sign_features_dataset_3_gcn.py
- sign2text_transformer_3_gcn.py
- graph.py

### ResNet50 + MediaPipe
- sign_features_dataset_resnet.py
- sign2text_transformer_resnet.py
- graph.py

Gated fusion
- sign_features_dataset_gated_fusion.py
- sign2text_transformer_gated_fusion.py
- graph_gated_fusion.py

### Example

```bash
PROJECT_ROOT="<path_to_project_root>"
REPO="$PROJECT_ROOT/SLT/external/signformer_gcn/English/slt_how2sign_wicv2023"
PATCH_DIR="$PROJECT_ROOT/SLT/signformer_overrides"

# Example: 3-GCN variant
SRC_DATASET="$PATCH_DIR/sign_features_dataset_3_gcn.py"
SRC_MODEL="$PATCH_DIR/sign2text_transformer_3_gcn.py"
SRC_GRAPH="$PATCH_DIR/graph.py"

DST_DATASET="$REPO/fairseq/data/sign_language/sign_features_dataset.py"
DST_MODEL="$REPO/fairseq/models/sign_to_text/sign2text_transformer.py"
DST_GRAPH="$REPO/fairseq/models/sign_to_text/graph.py"

# Optional backups of the original repo files
cp -f "$DST_DATASET" "${DST_DATASET}.bak"
cp -f "$DST_MODEL"   "${DST_MODEL}.bak"
cp -f "$DST_GRAPH"   "${DST_GRAPH}.bak"

# Copy in the fixed versions
cp -f "$SRC_DATASET" "$DST_DATASET"
cp -f "$SRC_MODEL"   "$DST_MODEL"
cp -f "$SRC_GRAPH"   "$DST_GRAPH"
```
## 2. Prepare the training config

Training configs live here:
```bash
SLT/external/signformer_gcn/English/slt_how2sign_wicv2023/examples/sign_language/config/
```
Create or edit an existing YAML for your run
```bash
<repo>/examples/sign_language/config/<config_name>.yaml
```

At minimum, the config should point to:
- the correct manifest directory through task.data
- the correct SentencePiece model through task.bpe_sentencepiece_model
- the correct train and validation subset names
- the intended checkpoint/output directory
- the correct feature type and model settings for your chosen override version

## 3. Launch Training

Call repo's Hydra training entrypoint directly

```bash
conda activate signformer

PROJECT_ROOT="<path_to_project_root>"
REPO="$PROJECT_ROOT/SLT/external/signformer_gcn/English/slt_how2sign_wicv2023"
CFG_DIR="$REPO/examples/sign_language/config"
CFG_NAME="<config_name>"

MANIFEST_DIR="<path_to_manifest_directory>"
SAVE_DIR="<path_to_experiment_root>"
WANDB_PROJECT="debug"
WANDB_NAME="<run_name>"

RUN_DIR="$SAVE_DIR/$WANDB_NAME"
CKPT_DIR="$RUN_DIR/ckpts"
TB_DIR="$RUN_DIR/tblog"
CONSOLE_LOG="$RUN_DIR/console.log"

mkdir -p "$CKPT_DIR" "$TB_DIR"

PYTHONPATH="$REPO:$PYTHONPATH" \
MP_DIR="$MANIFEST_DIR" \
SAVE_DIR="$SAVE_DIR" \
WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_NAME="$WANDB_NAME" \
WANDB_MODE=disabled \
WANDB_DISABLED=true \
python "$REPO/fairseq_cli/hydra_train.py" \
  --config-dir "$CFG_DIR" \
  --config-name "$CFG_NAME" \
  checkpoint.save_dir="$CKPT_DIR" \
  common.tensorboard_logdir="$TB_DIR" \
  "${RESTORE_ARGS[@]}" \
  2>&1 | tee -a "$CONSOLE_LOG"
```
