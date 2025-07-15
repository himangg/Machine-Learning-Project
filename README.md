# DeepDetect: Detect Fake and AI-generated Images

## Abstract

This project addresses the critical challenge of detecting deepfake and AI-generated images, which are increasingly difficult to distinguish from real media. We explore and compare classical machine learning techniques (SVM, Random Forests with HOG, LBP, SIFT features) and deep learning (CNN) for the task of image authenticity classification. Our goal is to enhance the ability to discern genuine content from fabricated images, thereby reducing the potential for misinformation.

---

## 1. Introduction

The proliferation of AI-generated content, especially deepfake images, poses significant risks to public trust, privacy, and security. Synthetic media is often indistinguishable from real content, making detection a pressing problem. This project aims to develop robust systems for detecting AI-generated images using both classical and deep learning approaches.

---

## 2. Literature Survey

Recent research has focused on automated systems for fake image detection using machine learning. Notable works include:
- **Detecting Fake Images Using Machine Learning**: Proposes CNN-based feature extraction and classification for high accuracy in content moderation and forensics.
- **Deep Learning for Image Authentication**: Compares ResNet and VAE models, highlighting the importance of hyperparameter tuning for distinguishing real and synthetic images.

---

## 3. Dataset and Exploratory Data Analysis

- **Dataset**: [Kaggle DeepFake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
    - Contains 256x256 pixel images of real and AI-generated human faces.
    - No significant class imbalance was found.
- **EDA**:
    - Bar graphs, scatter plots, and violin plots were used to analyze class distribution, image dimensions, color, and saturation.
    - Mean face images and RGB intensity plots revealed some differences, but not always significant for classification.

---

## 4. Methodology

### 4.1 Preprocessing and Data Augmentation
- **Image resizing** to reduce dimensionality and computational load.
- **Data augmentation** (spatial, color, noise, mild distortions) to improve generalization and mitigate overfitting.

### 4.2 Feature Extraction (Classical ML)
- **Histogram of Oriented Gradients (HOG)**: Captures edge and gradient structure, robust to illumination and scale.
- **Local Binary Patterns (LBP)**: Encodes local texture, useful for distinguishing real and synthetic content.
- **Scale-Invariant Feature Transform (SIFT)**: Detects keypoints and descriptors robust to scale, rotation, and illumination.

### 4.3 Model Training (Classical ML)
- **Support Vector Machines (SVM)**: Used with HOG, LBP, and SIFT features.
- **Random Forests**: Used with LBP features.
- **Ensemble Methods**:
    - **Voting Classifier**: Majority voting among models.
    - **Stacking Classifier**: Meta-classifier combines predictions for improved accuracy.

#### Hyperparameter Tuning
- Grid search was used to optimize feature extraction and model parameters.
- Best results:
    - **SIFT+SVM**: 61% accuracy
    - **HOG+SVM**: 68.9% accuracy
    - **LBP+SVM**: 60% accuracy
    - **Voting Classifier**: 72% accuracy
    - **Stacking Classifier**: 78% accuracy

### 4.4 Deep Learning (CNN)
- **Preprocessing**: Images normalized to [0,1], resized to 128x128.
- **Architecture**:
    - Four convolutional layers (filters: 32, 64, 64, 128), each with batch normalization, max pooling, and dropout (0.2).
    - Dense layer (256 units), softmax output for binary classification.
    - Dropout and batch normalization for regularization and faster convergence.
- **Training**:
    - Grid search for optimizer (Adam, RMSprop), learning rate (0.001, 0.0001), batch size (32, 64).
    - Early stopping on validation loss (patience=6, min_delta=0.001).
    - Best model: batch_size=32, learning_rate=0.001, optimizer=RMSprop.

---

## 5. Results and Analysis

### 5.1 Classical Machine Learning
- **Voting Classifier Test Accuracy**: 72%
- **Stacking Classifier Accuracy**: 78%
- **HOG+SVM**: Accuracy 66%, Precision 68%, F1 65%, ROC-AUC 0.73
- **LBP+Random Forest**: Accuracy 54%, Precision 57%, Recall 38%, F1 46%, ROC-AUC 0.73
- **LBP+SVM**: Accuracy 57%, Precision 62%, Recall 37%, F1 46%, ROC-AUC 0.73
- **SIFT+SVM**: Accuracy 68%, Precision 61%, Recall 64%, F1 62%

### 5.2 Deep Learning (CNN)
- **Validation Accuracy**: ~90.28%
- **Test Accuracy**: 84.76%
- CNNs significantly outperformed classical approaches, demonstrating strong generalization and robustness.

---

## 6. Conclusion

- Classical ML models with engineered features (HOG, LBP, SIFT) and ensemble methods can achieve reasonable performance (up to 78% accuracy).
- CNNs, with their ability to learn hierarchical features, outperform classical models, achieving over 90% validation accuracy and 84.76% test accuracy.
- Grid search and regularization techniques (dropout, batch normalization, early stopping) are crucial for optimal performance and generalization.
- The project demonstrates the strengths and limitations of both classical and deep learning approaches for deepfake detection.

---

