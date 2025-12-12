# Generative Models for Anomaly Detection in Medical Imaging

## Project Overview
This project explores the use of deep learning and generative models for anomaly detection in medical imaging, focusing on brain MRI scans for tumor analysis. The objective is to evaluate how generative models learn structural patterns in medical images and identify deviations that may indicate pathological conditions. The project integrates exploratory data analysis (EDA), supervised learning, and unsupervised generative modeling into a complete and interpretable pipeline.

## Objectives
- Analyze the structure and characteristics of a brain MRI dataset through EDA  
- Establish a supervised baseline using ResNet18  
- Implement a CycleGAN for unpaired image-to-image translation between tumor and non-tumor domains  
- Quantitatively evaluate generated images using Frechet Inception Distance (FID) and Inception Score (IS)  
- Frame anomaly detection around structural consistency rather than pixel-level reconstruction  

## Dataset
- Type: Brain MRI images  
- Domains: Tumor and No Tumor  
- Characteristics:
  - Variable image resolution and brightness  
  - Axial brain MRI slices  
- The dataset was preprocessed and split into training, validation, and test sets.

## Exploratory Data Analysis (EDA)
The EDA phase focused on class distribution, resolution variability, brightness and intensity ranges, and visual inspection to detect noise or corrupted images. This analysis confirmed that the dataset was clean, well-structured, and suitable for both supervised and generative modeling approaches.

## Methods

### Supervised Baseline: ResNet18
- Architecture: ResNet18  
- Task: Binary classification (Tumor vs No Tumor)  
- Input size: 224 × 224  
- Loss function: Cross-entropy loss  
- Purpose: Establish a strong baseline and provide a reference for evaluating generative models  

### Generative Model: CycleGAN
- Unpaired image-to-image translation between tumor and non-tumor domains  
- Architecture:
  - Generator networks based on encoder–decoder structures  
  - PatchGAN discriminators  
- Loss functions:
  - Adversarial loss  
  - Cycle consistency loss  
  - Identity loss  
- Optimization: Adam optimizer  
- Training was extended across multiple epochs with checkpointing for reproducibility.

## Evaluation Metrics

### Frechet Inception Distance (FID)
FID measures how closely the distribution of generated images matches the real data distribution by comparing high-level feature representations. Lower values indicate better alignment and greater structural similarity, making FID particularly suitable for medical imaging tasks.

### Inception Score (IS)
IS evaluates image quality and diversity by analyzing classification confidence and variability. As it was originally designed for natural images, IS is used here as a complementary metric.

### Results Summary

| Domain   | FID   | IS (mean ± std) |
|----------|-------|------------------|
| Tumor    | 115.39 | 2.01 ± 0.18 |
| No Tumor | 88.35  | 1.55 ± 0.09 |

The lower FID observed in the no-tumor domain reflects its lower anatomical variability, while the higher FID in the tumor domain highlights the complexity of pathological tissue.

## Anomaly Detection Perspective
Anomaly detection in this project is framed as the identification of structural deviations learned through domain translation. Reconstruction errors and visual differences between real and generated images provide insight into potential anomalies, making this approach suitable for exploratory clinical analysis and monitoring.

## Notes on Image Appearance
Generated images may appear smoother or larger due to resizing and interpolation effects introduced during preprocessing and upsampling operations within convolutional generative networks. This behavior is expected and does not indicate poor model performance, as preserving global anatomical structure is prioritized over pixel-level sharpness.

## Technologies Used
- Python  
- PyTorch  
- Torchvision  
- CycleGAN  
- ResNet18  
- Torch-Fidelity  
- Google Colab  
