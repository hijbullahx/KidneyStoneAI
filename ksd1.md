# Kidney Stone Detection & Synthetic Data Generation (KSD1)

## Overview
This repository contains the core notebooks for the KSD1 project, focusing on classifying kidney stones using Vision Transformers (ViT) and augmenting the dataset using Generative Adversarial Networks (GANs).

## Notebooks

* **`ksd1_00_utils.ipynb`**: Establishes the foundational environment by mounting Google Drive and setting up project directories. It exports reusable Python modules (`ksd1_gan.py` and `ksd1_train.py`) that contain GAN configurations and core PyTorch training/evaluation loops featuring automatic mixed precision and CosineAnnealingLR.

* **`ksd1_02_gan_generation.ipynb`**: Implements a Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate synthetic 128x128 resolution images. It trains distinct models for the "Normal" and "stone" classes. The pipeline generates 4,000 images per class and packages them into a ZIP archive (`ksd1_gan_generated_128.zip`) for downstream tasks.

* **`ksd1_03_vit_real.ipynb`**: Handles the evaluation and explainability phase using a Vision Transformer (ViT) architecture. It loads a pre-trained model (`ksd1_vit_real_best_v2.pth`) for classification and includes an interactive Grad-CAM implementation via Jupyter widgets to visualize the model's decision-making process on test images.

## Author
Md. Taher Bin Omar Hijbullah