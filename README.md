# grainpalette---a-deep-learning-odyssey-in-rice-type-classification-through-transfer-learning

# GrainPalette - A Deep Learning Odyssey in Rice Type Classification through Transfer Learning

## Overview

**GrainPalette** is a deep learning-based project that uses **transfer learning** to classify rice grain types. Using a pre-trained MobileNetV2 model, we fine-tune the model to recognize different types of rice (such as Basmati, Jasmine, and Arborio). The goal of this project is to automate rice classification, aiding in agricultural processes, research, and quality control.

## Dataset

The dataset used in this project consists of images of rice grains categorized into 5 types:
1. Basmati
2. Jasmine
3. Arborio
4. Ipsala
5. Karacadag

This dataset is publicly available on platforms like **Kaggle** or can be accessed from the **UCI ML Repository**. It is divided into training and testing sets, each containing images categorized by rice type.

## Project Files

### 1. **Data Preprocessing (`1_data_preprocessing.ipynb`)**
This Jupyter notebook includes:
- Loading and exploring the rice dataset.
- Preprocessing the images (resizing, normalization, augmentation).
- Splitting the dataset into training, validation, and test sets.

### 2. **Model Training (`2_model_training.ipynb`)**
This notebook contains:
- Loading the **MobileNetV2** pre-trained model.
- Adding a custom classification head for rice type prediction.
- Freezing the base layers of MobileNetV2 and fine-tuning the final layers.
- Model training using the augmented images.

### 3. **Model Evaluation (`3_model_evaluation.ipynb`)**
- Evaluating the model’s performance on the test data.
- Visualizing accuracy and loss curves.
- Generating a confusion matrix and classification report.

### 4. **Streamlit Application (`4_streamlit_app.py`)** *(optional)*
An optional Streamlit app that allows users to upload rice grain images and classify them in real-time using the trained model.

### 5. **Trained Model (`rice_model.h5`)**
This file contains the saved weights and architecture of the trained rice classification model.

### 6. **Label Map (`rice_label_map.json`)**
A JSON file containing the mapping of class indices to rice types (e.g., 0: Basmati, 1: Jasmine, etc.).

## Demo Video

For a live demonstration of the model in action, including the model evaluation and predictions, please refer to the following video:

[Watch the demo video](link_to_video)

## Installation

To run this project locally, you need to have **Python 3.6+** installed along with the required packages. You can install the dependencies using the `requirements.txt` file.

### Install Dependencies
```bash
pip install -r requirements.txt
