
## Interpretable Uncertainty-Aware Deep Regression with Cohort Saliency Analysis Pipeline

## Overview

This repository contains scripts to train a deep regression pipeline using a k-fold cross-validation approach. The pipeline involves generating a cross-validation split, formatting regression targets, and performing cross-validation on CT images also the GradCAM Saliency Map.

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- torchvision
- pydicom
- matplotlib
- PIL (Pillow)

## Getting Started

To train the deep regression pipeline, follow these steps:

1. Use `createNewSplit.py` to generate a k-fold cross-validation split from a list of IDs.

2. Utilize `formatTarget.py` to create text files containing subject IDs and corresponding values for a specified regression target. These files are organized based on the k-fold cross-validation split. Only include subjects for whom both a valid target value and an existing input image are available.

3. Execute `crossValidate.py` by providing the paths:
   - `CT_imageset_path`: Specify the path to the CT image set.
   - `output_dir`: Indicate the directory where the results will be stored.
   - `target_paths`: These paths should be created using `formatTarget.py` and should contain the IDs and targets based on the k-fold split.

By following these steps, you will train the deep regression pipeline, leveraging a k-fold cross-validation approach, ensuring that the input data is properly split, and the targets are formatted accordingly for effective training.



### GradCAM_Saliency_Map.ipynb: 
This python script utilizes GradCAM (Gradient-weighted Class Activation Mapping) to generate saliency maps for image targets with a pre-trained ResNet-50 model. GradCAM helps visualize which regions of an image are most influential in making the decision.

### createInferenceModule.py
This script is used to create an inference module for a neural network, which includes model weights, metadata about the regression targets, standardization parameters, and calibration factors. It ensures that the components are organized and stored in a structured manner for later use during inference or deployment.

### createNewSplit.py
This piece of code is used to create k-fold cross-validation splits by randomly shuffling and partitioning a list of image IDs (image names). It generates k subsets, writes the image names corresponding to each subset into separate text files, and saves them in a specified directory for later use in training and evaluation tasks.

### crossValidate.py
This script  is for training and evaluating neural networks, cross-validation, and storing results.

### dataLoading.py 
This script is for loading and processing data.

### evaluate.py
This code is for training a deep learning model for regression tasks that require estimating means and variances for target values. The model is based on a pre-trained ResNet-50 architecture, and the code provides functionality for efficient data loading, training loop management, and checkpoint saving. Additionally, custom loss functions are defined to handle specific regression requirements, such as handling missing ground truth values and estimating variances.

### formatTarget.py
This  script is for generating and formatting text files containing subject IDs and corresponding regression target values. The script takes input images, extracts labels from a given field file, and splits the data into subsets based on a specified cross-validation split.

### plotCompare.py
This Python script generates Bland-Altman and correlation plots to assess agreement between two datasets, along with computing agreement metrics like MAE and R². Useful for analyzing data agreement.

### StoreEvaluation
This Python script is used for aggregating and evaluating predictions. It combines predictions from different cross-validation subsets, calculates evaluation metrics like ICC, R², MAE, and MAPE, and generates various plots to visualize the agreement and uncertainty of the predictions. The code also includes functions for calibration and sparsification curve analysis, which are used to assess the reliability of the predictions. It's designed for assessing the performance of models on multiple targets and snapshots.

### train.py
This code is  for training a deep learning model for regression tasks, specifically for estimating means and variances while handling missing data points in the ground truth. It provides flexibility in terms of customization, data loading, and model checkpoint management.

## Acknowledgments

**Remember to replace placeholders such as `path/to/...` with the actual paths, and add any specific details about your project.**

This pipeline is inspired by [Taro's Deep Regression Pipeline](https://github.com/tarolangner/mri-biometry). Special thanks to Taro for providing a foundation for this work.
