Pneumonia Detection using Deep Learning (ResNet18)

A deep learning-based medical imaging project that detects Pneumonia from chest X-ray images using the ResNet18 architecture. The model is trained on a Kaggle dataset and deployed with a simple web interface for real-time inference.

Project Overview

This project focuses on building a binary image classification system to detect pneumonia from chest X-rays using transfer learning with ResNet18.

It includes:

Data preprocessing pipeline
ResNet18-based model training
Model evaluation using standard metrics
Web interface for image upload and prediction
Grad-CAM-based explainability

Input Chest X-Ray Image
        ↓
Image Preprocessing (Resize, Normalize)
        ↓
ResNet18 Model (Fine-tuned)
        ↓
Sigmoid Output Layer
        ↓
Prediction: Normal / Pneumonia
        ↓
Grad-CAM Heatmap Visualization
        ↓
Frontend Display



💻 Tech Stack
🔬 Machine Learning
Python
PyTorch / TensorFlow
OpenCV
Scikit-learn
🌐 Frontend
HTML
CSS
JavaScript
📊 Visualization
Matplotlib
Seaborn
Grad-CAM
