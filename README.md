# Chest X-Ray Pneumonia Detector

A deep learning-based diagnostic support tool utilizing Convolutional Neural Networks (CNN) to detect pneumonia from pediatric chest X-ray images. This project provides a complete end-to-end pipeline from data preprocessing and model training to evaluation and inference.

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Usage](#usage)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Inference / Prediction](#inference--prediction)
7. [Results & Performance](#results--performance)
8. [Clinical Disclaimer](#clinical-disclaimer)
9. [License](#license)

---

## 🔍 Overview
Pneumonia is an inflammatory condition of the lung primarily affecting the microscopic air sacs (alveoli). Early and accurate diagnosis from chest radiographs is crucial for effective treatment. This repository contains a deep learning classifier trained to distinguish between **Normal** chest X-rays and those exhibiting signs of **Pneumonia** (viral or bacterial).

---

## 📊 Dataset
The model is trained on the publicly available **Chest X-Ray Images (Pegasus/Kaggle / Guangzhou Women and Children's Medical Center)** dataset.
- **Total Images:** ~5,850 X-Ray images (JPEG)
- **Categories:** 2 Classes (`NORMAL`, `PNEUMONIA`)
- **Structure:** Divided into `train`, `val`, and `test` directories.

---

## 🧠 Model Architecture
- **Architecture Type:** Deep ANN / Custom Convolutional Neural Network (CNN)
- **Input Dimensions:** `224x224x3` (RGB normalized)
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam with learning rate reduction on plateau
- **Regularization:** Dropout, Batch Normalization, and Data Augmentation (rotation, zoom, horizontal flip) to prevent overfitting.

---

## 📁 Project Structure
```text
chest-xray-pneumonia-detector/
│
├── data/                    # Dataset directory (ignored in git)
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/                  # Saved model weights (.h5 / .keras / .pt)
│
├── notebooks/               # Exploratory data analysis & prototyping
│   └── EDA_and_baseline.ipynb
│
├── src/                     # Source code package
│   ├── __init__.py
│   ├── dataset.py           # Data loading and augmentation pipelines
│   ├── model.py             # CNN architecture definitions
│   ├── train.py             # Training loop and checkpointing
│   └── evaluate.py          # Metrics, ROC curves, and confusion matrix
│
├── app.py                   # Streamlit web application for interactive demo
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
