# AV-XAI: Explainable Multimodal Deepfake Detection

Implementation of an audio-visual deepfake detection framework using ViViT (video) and WavLM (audio) backbones with hierarchical fusion, cross-modal consistency learning, and a multi-technique explainability layer.

**Researchers:** Komal Yadav (PhD), Aditya Bhardwaj (PhD), Aayush Sharma (research intern, technical execution)
**Affiliation:** Bennett University
**Status:** In progress — dataset preparation phase

---

## Folder structure

```
AV-XAI/
├── setup.bat                       # Run first - creates env 'df'
├── requirements.txt
├── README.md                       # this file
├── project_log.md                  # running experiment log (UPDATE AS WE GO)
│
├── notebooks/
│   ├── 01_setup_and_download.ipynb # DFDC download + sampling + actor-disjoint splits
│   ├── 02_preprocessing.ipynb      # (next) face detection + audio extraction
│   ├── 03_model_definition.ipynb   # (later) ViViT + WavLM + fusion + XAI
│   ├── 04_training.ipynb           # (later) full training loop
│   ├── 05_adversarial_ensemble.ipynb # (later) 5-model ensemble
│   └── 06_evaluation_and_xai.ipynb # (later) test set evaluation + XAI generation
│
├── data/
│   ├── raw_zips/                   # transient: DFDC parts being processed (deleted as we go)
│   ├── raw_clips/                  # 30K sampled mp4 clips (will be deleted after preprocessing)
│   ├── preprocessed/
│   │   ├── faces/                  # 16-frame face crop sequences per clip
│   │   └── audio/                  # 10s audio at 16kHz per clip
│   ├── manifest.csv                # clip_id, label, actor_id, split, source_part
│   └── splits/                     # train.csv, val.csv, test.csv
│
├── checkpoints/                    # model .pt files
│   ├── model_1_heavy_jpeg/
│   ├── model_2_light_jpeg/
│   ├── model_3_gaussian_noise/
│   ├── model_4_temporal_resample/
│   └── model_5_mixed_aug/
│
├── logs/
│   ├── training/                   # TensorBoard logs per ensemble variant
│   └── evaluation/                 # Final metrics, confusion matrices
│
└── figures/                        # Paper figures: architecture, results, XAI samples
```

---

## Quick start

### 1. Prerequisites
- Windows 10/11
- Anaconda or Miniconda installed
- NVIDIA driver supporting CUDA 12.1 (verify with `nvidia-smi`)
- ~500 GB free disk space
- Kaggle account that has accepted the DFDC competition rules

### 2. Get Kaggle API credentials
1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token" — downloads `kaggle.json`
3. Move it to `C:\Users\<your-username>\.kaggle\kaggle.json`
4. Accept the DFDC rules at https://www.kaggle.com/c/deepfake-detection-challenge/rules

### 3. Set up environment
From an **Anaconda Prompt**, in this folder, run:
```
setup.bat
```
This creates the `df` conda env, installs PyTorch with CUDA 12.1, installs all dependencies, and registers a Jupyter kernel.

### 4. Run dataset preparation
```
conda activate df
jupyter notebook
```
Open `notebooks/01_setup_and_download.ipynb`, select kernel `Python (df)`, run cells top-to-bottom.

This will:
- Download DFDC parts one at a time
- Sample 30K clips with actor-disjoint train/val/test splits (24K / 3K / 3K)
- Maintain perfect class balance (15K real / 15K fake)
- Save 30K raw mp4 files to `data/raw_clips/`
- Build `data/manifest.csv` with full bookkeeping
- Delete each zip after processing to manage disk

**Expected runtime:** 6-10 hours (network-bound).

---

## Sampling design (important — read before running)

**Actor-disjoint splits.** Each "actor" in DFDC is identified by the `original` field in metadata.json. All clips derived from the same source clip share an actor ID. We hash actor IDs deterministically to assign each actor to exactly one of train/val/test. No actor appears in two splits, preventing face-memorization leakage.

**Class balance.** For each actor we keep 1 real + 1 fake clip → 30K clips total, perfectly balanced. DFDC's natural distribution is ~5:1 fake-heavy; we subsample fakes to balance.

**Stratification.** Hash-based splitting gives 80/10/10 across actors, which translates to 80/10/10 across clips. Class balance is preserved within each split since the structure is 1 real + 1 fake per actor.

---

## Hardware target
- Single NVIDIA RTX A6000 (48 GB VRAM)
- ~500 GB SSD
- Windows + AnyDesk for remote access
