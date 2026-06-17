---
title: TomLeafVision
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
python_version: 3.10.0
# rebuild
---

TomLeafVision

TomLeafVision is a web-based application that helps identify common tomato leaf diseases from an uploaded image. The project uses deep learning to classify tomato leaves into different disease categories and provides useful information about symptoms, treatment, and prevention.

The application was built to demonstrate the practical use of computer vision and machine learning in agriculture. Farmers, students, and researchers can use it to quickly analyze tomato leaf conditions and receive basic disease-related guidance.

Features
Upload a tomato leaf image through a simple web interface
Detects Early Blight, Late Blight, and Healthy leaves
Validates whether the uploaded image is actually a tomato leaf
Checks image quality before making predictions
Displays prediction confidence scores
Provides disease symptoms and treatment suggestions
Shows preventive measures for better crop health
Technologies Used
Python
TensorFlow / Keras
MobileNetV2
OpenCV
PyTorch
OpenAI CLIP
Hugging Face Transformers
Gradio
NumPy
How It Works
The user uploads an image of a tomato leaf.
The system checks image quality, including brightness and blur.
A CLIP-based validation step verifies that the image contains a tomato leaf.
The trained MobileNetV2 model predicts the disease category.
The application generates a report containing:
Predicted disease
Confidence score
Severity level
Symptoms
Treatment recommendations
Prevention tips
Detectable Classes
Early Blight
Late Blight
Healthy
