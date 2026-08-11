# Detection of Cerebrovascular Diseases with in Vivo MRI and Deep Neural Networks

## Overview
Cerebrovascualr diseases (CVD) can contribute to the risk of other diseases, like dementia, and can further result in death. However, some vascular changes may be not detectable through the regular inspection of the brain scans, which indicates some specialized imaging is needed for CVD's detection. In this study, we used convolutional neural network (CNN) to detect CVD with the **T1 weighted structural MRI scans** of patients.  

This repository contains code for a project to detect CVD with MRI by using CNN. There are two names of the same MRI data type: T1 surface and T1 structural. Basically they are the same data but processed by different programs. We mainly focused on T1 surface in this study. The following steps are similar to creating CNNs for hypertension and artherosclerosis. **CNNs for hypertension and artherosclerosis share the same structure, configuration, and functions for retrieving images and labels from the UKBiobank. However, they are trained on different datasets. Therefore, in the later code, we can reuse the same `dataset.py` and `config_matched.yml`.**

We used T1 weighted structural MRI scans from **UKBiobank**, which originally contained around 500,000 participants with age range from 44 to 83 years old. CVD is identified by icd code **I67** in UKBiobank. After removing participants who do not have MRI scan and patients who are diagnosed with other diseases, we had 237 positive cases with T1 surface image for matching. Matching was done with four variables: **age**, **sex**, **BMI**, **ethnicity**.

## Prerequisites
**Download [requirements.txt](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/requirements.txt) to install these packages**  
Python 3.8             
Pytorch 2.3.1  
torchvision 0.18.1  
matplotlib 3.7.5  
scikit-learn 1.3.2  
pandas 1.5.3  
numpy 1.23.5  
PyYAML 6.0.1  

## Run Code
**Don't forget to replace the path with your path!**
* Get valid participants: [EDA.ipynb](https://github.com/FelskyLab/MRI_CNN/blob/Artherosclerosis/EDA.ipynb) in `artherosclerosis` branch  
* Create datasets: [create_ds.ipynb](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/create_ds.ipynb)
* Download `dataset.py`(for getting items) and `__init__.py`(make sure python recognize PatientDataset as a function), move them to a folder in the current directory: [dataset](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/data/dataset_cropped.py), [\_\_init\_\_](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/data/__init__.py)
* Download `config_matched.yml` in the current directory: [configuration](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/config/config_matched.yml)
* Run main: [main](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/CBV_CNN_matched_optimal.py). Remember to create folders like `plot`, `result` in the current directory to store the output files and plots.
* Create heatmaps and interpret: [heatmap](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/CBV_CNN_hm_xyz.py)  


## Prepare Datasets & Image Preprocessing
Before we start to prepare datasets, we should get a list of valid participants. (see [EDA.ipynb](https://github.com/FelskyLab/MRI_CNN/blob/Artherosclerosis/EDA.ipynb) in artherosclerosis branch)  

Preparing datasets with valid patient list is done in this python file:  [create_ds.ipynb](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/create_ds.ipynb).  

Preparing **matched datasets** is done in this python file:  [CNN_matching](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/CNN_matching.ipynb). The final dataset we used is **train_matched.csv**, **val_matched.csv**, and **test_matched.csv**, which contain 138, 40, 20 participants respectively with 1:1 case control ratio.  

Image preprocessing(image normalization and cropping) is done in this python file when we pull the images from the dataset:  [dataset](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/data/dataset_cropped.py)  
Image normalization is to make sure CNN can recognize the same regions across different images(since the same regions can have differnt average voxel, which makes it hard for CNN to recognize).  
Cropping is to reduce the noise introduced by the black area outside of the brain.  

Last step before we feed the images into the CNN is load the configuration file: [config for T1 surface](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/config/config_matched.yml), [config for T1 structural](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/config/config_matched_tem.yml). Then it will read the paths of images and datasets for creating dataloaders.  

Finally, images and labels are ready for training!  

## Construct CNN  
The architecture of the CNN is defined here: [main](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/CBV_CNN_matched_optimal.py), which also includes functions for saving models, calculating metrics and their confidence intervals, and plotting AUCs and loss vs epoch.  

## Calculate the metrics
Metrics are calculated here: [main](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/CBV_CNN_matched_optimal.py). Accuracy, AUROC, AUPRC, and Confusion Matrix are used to evaluate the model performance.  
AUC plots of the final model: 
<p align="center">
<img width="599" alt="CBV_matched_lr0 003_auc" src="https://github.com/user-attachments/assets/cd67bf08-b360-4b1a-afa6-a972d798b830">
<img width="611" alt="CBV_matched_lr0 003_pr" src="https://github.com/user-attachments/assets/68717fd7-eb34-4a77-825e-3f2e64a85ebc">
</p>

## Create Heatmaps
The code for creating heatmaps: [heatmap](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/CBV_CNN_hm_xyz.py)  
The code for creating heatmaps of the highest weights: [highest weight heatmap](https://github.com/FelskyLab/MRI_CNN/blob/cerebrovascular/CBV_CNN_hm_highest_avg.py)  
Average Heatmaps of 10 participants in the testing dataset:  
<p align="center">
<img width="1068" alt="average_heatmap" src="https://github.com/user-attachments/assets/2c123bf5-89f0-43a8-bdbf-d6bcab3e4dfc">
</p>
From the heatmaps, we can have an understanding of which regions are considered to be important by CNN for CVD's detection.
