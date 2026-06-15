# Kidney Stone Detection & Synthetic Data Generation (KSD1)

## Overview
This repository contains the core notebooks for the KSD1 project, focusing on classifying kidney stones using advanced vision architectures including Vision Transformers (ViT), Data-efficient Image Transformers (DeiT), Swin Transformers, and LeViT. It also features a robust data augmentation pipeline utilizing Generative Adversarial Networks (GANs) to improve model generalization, culminating in a unified interactive explainability dashboard.

## Notebooks

### 1. Environment & GAN Generation
* **`ksd1_00_utils.ipynb`**: Establishes the foundational environment by mounting Google Drive and setting up project directories. It exports reusable Python modules (`ksd1_gan.py` and `ksd1_train.py`) that contain GAN configurations and core PyTorch training/evaluation loops featuring automatic mixed precision and CosineAnnealingLR.
* **`ksd1_02_gan_generation.ipynb`**: Implements a Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate synthetic 128x128 resolution images. It trains distinct models for the "Normal" and "stone" classes. The pipeline generates 4,000 images per class and packages them into a ZIP archive (`ksd1_gan_generated_128.zip`) for downstream tasks.

### 2. Vision Transformer (ViT) Pipeline
* **`ksd1_03_vit_real.ipynb`**: Handles the evaluation and explainability phase using a Vision Transformer (ViT) architecture. It loads a pre-trained model (`ksd1_vit_real_best_v2.pth`) for classification and includes an interactive Grad-CAM implementation via Jupyter widgets to visualize the model's decision-making process on test images.
* **`ksd1_04_vit_real_gan.ipynb`**: Integrates the synthetic GAN images with the real dataset (using a 1:0.5 real-to-GAN ratio) to create a mixed training set. It evaluates the ViT model on this augmented dataset to assess the performance impact of synthetic data.

### 3. Data-efficient Image Transformer (DeiT) Pipeline
* **`ksd1_05_deit_real.ipynb`**: Shifts the architecture to a Data-efficient Image Transformer (DeiT). It trains and evaluates the model strictly on the real dataset, outputs comprehensive metrics (ROC/PR curves, Confusion Matrix), and features a custom `DeiTV2GradCAM` interactive UI for visual explainability.
* **`ksd1_06_deit_real_gan.ipynb`**: Completes the comparative analysis by training and evaluating the DeiT model on the mixed (Real + GAN) dataset, allowing for a direct comparison of how synthetic augmentation impacts the DeiT architecture.

### 4. Swin Transformer Pipeline
* **`ksd1_07_swin_real.ipynb`**: Applies the Swin Transformer architecture. It incorporates an advanced preprocessing technique called `polar_ultra_crop`. It trains and evaluates the model on the real dataset, calculating extensive evaluation metrics (including Balanced Accuracy, ROC-AUC, PR-AUC, and Cohen's Kappa) alongside an interactive confusion matrix.
* **`ksd1_08_swin_real_gan.ipynb`**: Focuses on training the Swin Transformer on the augmented (Real + GAN) dataset to examine the model's adaptability when synthetic data is introduced during training.

### 5. LeViT Pipeline
* **`ksd1_09_levit_real.ipynb`**: Explores the LeViT architecture for classification. It trains the model on the real dataset, outputs classification reports, precision-recall curves, and ROC curves, and includes an interactive dropdown tool for running Grad-CAM on specific test images.
* **`ksd1_10_levit_real_gan.ipynb`**: Trains and evaluates the LeViT architecture on the mixed (Real + GAN0.5) dataset to analyze the performance enhancements provided by the synthetic data.

### 6. Unified Testing & Explainability UI
* **`ksd1_11_test_ui_gradcam.ipynb`**: Provides a comprehensive, interactive evaluation dashboard. It features Jupyter widget controls (Dropdowns and Sliders) allowing users to dynamically select between all eight trained model variations (ViT, DeiT, Swin, LeViT on both Real and GAN-augmented datasets) and run Grad-CAM visualizations on individual test images with adjustable overlay opacities.

## Author
Md. Taher Bin Omar Hijbullah