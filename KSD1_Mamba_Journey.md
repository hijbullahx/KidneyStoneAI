# Kidney Stone AI - Vision Mamba Integration Journey

## Objective
Introduce Vision Mamba (Vim) as a state-of-the-art State Space Model (SSM) baseline to compare against our highly accurate Transformer models (ViT, DeiT, Swin, LeViT) for Q1 journal publication.

## Progress Log

### [Completed] Step 1: Workspace & Environment Initialization
* **File:** `ksd1_12_mamba_real.ipynb`
* **Action:** Successfully mapped Google Colab to the local Windows Drive structure (`O:\My Drive\papers\Sabbir\...`).
* **Result:** Verified workspace paths and extracted the clean Real Dataset (`ksd1_dataset_70_15_15_uniqueValTest_BALANCED_v2.zip`) into Colab's fast local runtime (`/content/ksd1_split_real`). Train, Val, and Test splits perfectly preserved containing `stone` and `normal` classes.

### [Pending] Step 2: Pure PyTorch Mamba Architecture & Training
* **Goal:** Implement a CUDA-independent Mamba block to avoid Colab compilation errors, configure data loaders, train on the Real dataset for 15 epochs, and evaluate.