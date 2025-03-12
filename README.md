# Handwriting Classification using Deep Learning

## Overview
This project focuses on classifying handwritten documents by recognizing unique handwriting styles. The approach involves **Convolutional Neural Networks (CNNs)** and **MobileNetV2** for feature extraction and classification. The dataset consists of scanned document images, which are processed into individual lines for classification.

## Features
### 1. Data Preparation and Processing
- The dataset is divided into two sets: **Development Set** and **Test Set**.
- Further split into **Training, Validation, and Testing** subsets.
- **Sliding window technique** is applied to segment images for model training.
- Labels are **one-hot encoded**, and images are **normalized** for better model performance.

### 2. Model Architecture
- Implements a **CNN model** for classification.
- Uses **MobileNetV2** as a pre-trained model for transfer learning.
- Alternative architectures like **ResNet50** and **VGG16** are tested.
- The model consists of convolutional layers, **batch normalization**, and **dropout layers** for regularization.

### 3. Training and Evaluation
- **Early stopping** is used to prevent overfitting.
- The model is trained using **categorical cross-entropy loss** and **Adam optimizer**.
- Evaluation metrics include:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
- **Confusion matrix visualization** for performance analysis.

### 4. Performance and Findings
- Initial models faced **overfitting issues**, leading to modifications in data augmentation.
- **MobileNetV2 outperformed the CNN model** in accuracy and efficiency.
- Due to **GPU memory constraints**, images were resized to optimize processing.

## File Structure
### Main Files
- **`40_classes_cnn.py`**: Main script handling **data loading, preprocessing, model training, and evaluation** for 40 classes.
- **`CodeLib.py`**: A library of helper functions for **image processing, dataset handling, augmentation, and model training**.

## Getting Started
### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib
- SciPy

## Results & Discussion
- The **MobileNetV2-based model** achieved higher accuracy compared to traditional CNN.
- Sliding window preprocessing improved training efficiency.
- Further improvements could be achieved with **data augmentation** and **hyperparameter tuning**.

## Future Work
- Implement **more robust augmentation techniques**.
- Experiment with **different feature extraction models**.
- Optimize **GPU usage** for training on higher-resolution images.

## Acknowledgments
This project was inspired by **Deep Learning for Vision Systems** by Mohamed Elgendy and **Deep Learning with Python** by Francois Chollet.

## License
This project is licensed under the MIT License.

