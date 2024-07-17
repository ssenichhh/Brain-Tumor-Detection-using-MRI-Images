# Brain Tumor Detection using MRI Images

This project demonstrates how I used programming skills to detect brain tumors using MRI images. The dataset for this project comes from Kaggle, and I utilized a well-known deep learning model called VGG16 to achieve this.

## Table of Contents

- [Overview](#overview)
- [How to Use](#how-to-use)
- [Dataset](#dataset)
- [Steps I Took](#steps-i-took)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [Requirements](#requirements)

## Overview

In this project, I aimed to detect brain tumors using MRI images. I used a dataset from Kaggle and employed a pre-trained model called VGG16, which is great for image classification tasks. This project showcases my ability to work with data, preprocess it, train a model, and evaluate its performance.

## How to Use

1. **Clone the repository:**
    ```bash
    git clone https://github.com/ssenichhh/Brain-Tumor-Detection-using-MRI-Images
    ```
2. **Download the dataset from Kaggle.**
3. **Run the Jupyter Notebook** to see how the model is trained and evaluated.

## Dataset

The dataset I used is available on Kaggle:
- [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

The dataset is organized into two categories: images with tumors (`YES`) and images without tumors (`NO`).

## Steps I Took

### Preprocessing

1. **Resizing**: All images were resized to 224x224 pixels.
2. **Cropping**: I focused on regions of interest in the images.
3. **Data Augmentation**: I used techniques like rotation, shifting, and flipping to increase the diversity of the training data.

### Model

I used the VGG16 model, which is already trained on a large dataset. I fine-tuned it to classify MRI images as either having a brain tumor or not.

### Training

I trained the model using augmented data and used early stopping to avoid overfitting. This means the training stopped automatically when the model performance stopped improving.

### Evaluation

I evaluated the model using accuracy and a confusion matrix, which helps to see how well the model distinguishes between images with and without tumors.

## Results

The model was able to detect brain tumors in MRI images with good accuracy. I visualized some sample images from the dataset, as well as some augmented images, to show the variety of data used for training.

## Acknowledgments

- The dataset is provided by Kaggle.
- The VGG16 model is available through Keras.

## Requirements

To access the VGG16 weights stored in Google Drive, follow this link:
- [VGG16 Weights](https://drive.google.com/drive/folders/1xCkJOzZWLrJbd9MhkdFzrBxB-mYWO68V?usp=sharing)

Thank you for checking out my project! If you have any questions or feedback, feel free to reach out.
