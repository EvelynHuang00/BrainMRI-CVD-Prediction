# Cardiovascular Disease Prediction from Brain MRI

A deep learning pipeline for predicting **cerebrovascular disease (CVD)** from T1-weighted structural brain MRI using a **3D convolutional neural network (CNN)**.

This project was developed as part of a research project using UK Biobank neuroimaging data. It investigates whether structural patterns in brain MRI contain predictive information associated with cerebrovascular disease and uses saliency-based visualization to examine which image regions contribute to model predictions.

## Project Overview

Cerebrovascular disease can produce structural changes in the brain that may not be readily captured through conventional visual inspection. This project approaches the problem as a binary image-classification task:

> **Can a 3D CNN distinguish individuals with cerebrovascular disease from matched controls using structural brain MRI?**

The end-to-end pipeline includes:

- cohort construction from UK Biobank data
- matched case-control sampling
- 3D MRI preprocessing
- CNN training and validation
- model performance evaluation
- saliency-based model interpretation

## Data

The analysis uses **T1-weighted structural brain MRI** from the UK Biobank.

Cerebrovascular disease cases were identified using ICD-10 code **I67**. After filtering for participants with available MRI data and applying disease-related exclusion criteria, 237 eligible positive cases were identified for matching.

To reduce confounding between cases and controls, participants were matched using:

- age
- sex
- BMI
- ethnicity

The resulting modeling datasets use a **1:1 case-control ratio**.

Raw UK Biobank imaging data are not included in this repository due to data access restrictions.

## MRI Preprocessing

Each MRI is loaded as a 3D NIfTI volume and processed before being passed to the neural network.

The preprocessing pipeline:

1. identifies the nonzero region of the MRI volume;
2. crops the volume to remove empty background surrounding the brain;
3. converts voxel values to 32-bit floating point representation;
4. standardizes nonzero voxel intensities using within-image z-score normalization;
5. converts the processed volume into a PyTorch tensor with an additional channel dimension.

Cropping reduces unnecessary background information and memory usage, while normalization improves consistency in voxel intensity scale across participants.

## 3D CNN Architecture

The classification model is implemented in **PyTorch** and operates directly on volumetric MRI data.

The network contains five 3D convolutional blocks:

```text
3D MRI Volume
      ↓
Conv3D: 1 → 32
MaxPool3D
      ↓
Conv3D: 32 → 64
MaxPool3D
      ↓
Conv3D: 64 → 128
MaxPool3D
      ↓
Conv3D: 128 → 256
MaxPool3D
      ↓
Conv3D: 256 → 512
MaxPool3D
      ↓
Fully Connected Layers
8192 → 2048 → 512
      ↓
Dropout
      ↓
Binary CVD Prediction
```

ReLU activation is applied throughout the hidden layers, with dropout used in the fully connected portion of the network for regularization.

## Model Training

The model is trained as a binary classifier using:

- binary cross-entropy loss
- stochastic gradient descent with momentum
- L2 weight decay
- learning-rate scheduling
- validation-based early stopping
- model checkpointing

Training and validation losses are monitored across epochs to identify convergence and reduce overfitting.

## Model Evaluation

Performance is evaluated using multiple classification metrics rather than accuracy alone:

- ROC-AUC
- PR-AUC
- accuracy
- precision
- recall
- F1 score
- confusion matrix

Bootstrap resampling is used to quantify uncertainty in ROC-AUC and PR-AUC estimates.

An alternative classification threshold is also evaluated using **Youden's J statistic**, allowing sensitivity and specificity to be considered rather than relying exclusively on the default 0.5 probability threshold.

### ROC and Precision-Recall Curves

<p align="center">
<img width="48%" alt="ROC curve" src="https://github.com/user-attachments/assets/cd67bf08-b360-4b1a-afa6-a972d798b830">
<img width="48%" alt="Precision-recall curve" src="https://github.com/user-attachments/assets/68717fd7-eb34-4a77-825e-3f2e64a85ebc">
</p>

The model achieved approximately **0.72 ROC-AUC** and **0.66 PR-AUC** on held-out evaluation data, indicating that structural MRI contains predictive signal for distinguishing CVD cases from matched controls.

Given the relatively small matched sample, these results are interpreted as evidence of predictive signal rather than as a clinically deployable diagnostic model.

## Model Interpretation

To investigate which regions of the MRI contribute most strongly to model predictions, gradient-based **saliency maps** are generated from the trained 3D CNN.

Saliency values quantify how sensitive the model output is to changes in individual input voxels. The resulting maps can be visualized across:

- axial
- sagittal
- coronal

views.

Participant-level maps can also be aggregated to examine regions that consistently receive high saliency across subjects.

### Average Saliency Map

<p align="center">
<img width="90%" alt="Average MRI saliency map" src="https://github.com/user-attachments/assets/2c123bf5-89f0-43a8-bdbf-d6bcab3e4dfc">
</p>

These visualizations provide a qualitative interpretation of the spatial regions influencing CNN predictions. They should not, however, be interpreted as evidence that the highlighted regions are causally associated with cerebrovascular disease.

## Repository Structure

```text
BrainMRI-CVD-Prediction/
│
├── config/
│   └── config_matched.yml
│
├── data/
│   ├── dataset.py
│   └── dataset_cropped.py
│
├── CBV_CNN_matched_optimal.py
│   # 3D CNN training, validation, and evaluation
│
├── CBV_CNN_hm_xyz.py
│   # Saliency maps across anatomical views
│
├── CBV_CNN_hm_highest_avg.py
│   # Aggregated high-saliency visualization
│
├── CNN_matching.ipynb
│   # Case-control matching
│
├── create_ds.ipynb
│   # Cohort and dataset construction
│
├── requirements.txt
└── README.md
```

## Technologies

**Deep Learning:** PyTorch, torchvision  
**Neuroimaging:** NiBabel, Nilearn  
**Data Processing:** NumPy, pandas  
**Machine Learning & Evaluation:** scikit-learn  
**Visualization:** Matplotlib  
**Compute:** HPC environment for 3D CNN training

## Research Context

This project was developed as part of a research project in neuroimaging and computational health.

The primary objective was to build an end-to-end pipeline connecting **clinical cohort construction, 3D medical-image processing, deep learning, statistical evaluation, and model interpretation**.

Because UK Biobank imaging data require authorized access, this repository provides the analysis and modeling code but does not distribute the underlying MRI data.
