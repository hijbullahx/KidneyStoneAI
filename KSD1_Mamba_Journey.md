# Kidney Stone AI - Vision Mamba Integration Journey

## Objective
Introduce Vision Mamba (Vim) as a state-of-the-art State Space Model (SSM) baseline to compare against our highly accurate Transformer models (ViT, DeiT, Swin, LeViT) for Q1 journal publication.

## Progress Log

### [Completed] Step 1: Workspace & Environment Initialization
* Verified workspace paths and extracted clean Real Dataset into Colab local runtime.

### [In Progress] Step 2: Pure PyTorch Mamba Architecture & Training
* **Hurdle:** CUDA OOM Error encountered at batch_size=32 due to Pure PyTorch sequential graph retention.
* **Fix:** Implemented Gradient Accumulation (micro-batch=4, accumulation_steps=8) to maintain an effective batch size of 32 while keeping maximum memory usage well below the 15GB T4 limit.