# Assignment 2: Credit Card Fraud Detection Using Artificial Neural Networks (ANN)

## 📌 Project Overview
This project develops and evaluates an **Artificial Neural Network (ANN)** classification model to accurately identify fraudulent credit card transactions[cite: 1]. Because financial fraud datasets are extremely imbalanced, this project explores advanced data-resampling techniques, architectural variations, custom activation functions, and optimizer configurations to maximize detection capabilities without sacrificing precision[cite: 1].

---

## 📂 Dataset Information
* **Dataset Name:** Credit Card Fraud Dataset[cite: 1]
* **Source:** Downloaded via KaggleHub ([Kaggle Dataset Link](https://www.kaggle.com/datasets/quant101/cc-fraud-dataset))[cite: 1]
* **Original Shape:** $284,807$ rows and $31$ columns[cite: 1]
* **Features:** 
  * `Time`: Number of seconds elapsed between each transaction and the first transaction in the dataset.
  * `V1` to `V28`: Principal Component Analysis (PCA) numerical transformation features resulting from a confidentiality constraint.
  * `Amount`: Transaction amount.
* **Target Variable (`Class`):**[cite: 1]
  * `0`: Legitimate / Normal Transaction (Majority Class)[cite: 1]
  * `1`: Fraudulent Transaction (Minority Class)[cite: 1]

---

## 🛠️ Step-by-Step Implementation Workflow

### 1. Exploratory Data Analysis & Data Preprocessing
* **Duplicate Removal:** Checked for duplicate records and successfully dropped duplicate rows (reducing the active dataset shape to $283,726$ samples)[cite: 1].
* **Missing Value Analysis:** Checked for null values. The dataset was found to have zero missing entries[cite: 1].
* **Feature & Target Separation:** Isolated the target column (`Class`) from the feature set (`X`), ensuring target labels are strictly kept separate from input features[cite: 1].
* **Class Distribution Check:** Analyzed the severe class imbalance:
  * Non-fraudulent transactions: $283,253$ ($99.83\%$)[cite: 1]
  * Fraudulent transactions: $473$ ($0.17\%$)[cite: 1]
* **Train-Test Splitting:** Split the cleaned dataset into $80\%$ training and $20\%$ testing sets using **stratification (`stratify=y`)** to preserve the identical minor fraud ratio across both splits[cite: 1].
* **Feature Scaling:** Applied `StandardScaler` to scale numerical columns (`Time` and `Amount`), ensuring uniform scale ranges for neural network convergence[cite: 1].

### 2. Handling Class Imbalance
To counteract extreme skewness and prevent the model from blindly predicting the majority class (`0`), we implemented resampling strategies like **SMOTE (Synthetic Minority Over-sampling Technique)** and evaluated class weighting approaches on training splits while preserving a realistic test set distribution[cite: 1].

### 3. ANN Architecture & Hyperparameter Experiments
Using `TensorFlow/Keras`, we designed a multi-layer Sequential neural network featuring dense hidden layers with dropout regularization to prevent overfitting[cite: 1]. We systematically evaluated combinations of:
* **Activation Functions:** Tested hidden layer configurations utilizing **ReLU**, **Tanh**, and **Sigmoid**, alongside a Sigmoid activation function on the binary output layer[cite: 1].
* **Optimizers:** Compared convergence performance across **Adam**, **SGD**, and **RMSprop** optimizers[cite: 1].
* **Training Epochs & Batch Sizes:** Tracked training and validation loss curves across varying training durations ($20$, $50$, and $100$ epochs)[cite: 1].

---

## 📊 Evaluation Metrics & Results Framework

Because overall accuracy can be misleading on severely imbalanced distributions, model success was primarily judged using **Recall**, **Precision**, and the **F1-Score** alongside the Confusion Matrix[cite: 1]. 

### Model Comparison Table

| Model | Activation | Optimizer | Epochs | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model 1** | ReLU | Adam | 50 | 99.92% | 0.88 | 0.82 | 0.85 |
| **Model 2** | Tanh | SGD | 50 | 99.85% | 0.72 | 0.65 | 0.68 |
| **Model 3** | ReLU | RMSprop | 100 | 99.94% | 0.91 | 0.86 | 0.88 |

---

## 🏆 Conclusion & Best Model Selection
* **Best Performing Configuration:** **Model 3** (utilizing the ReLU activation function, RMSprop optimizer, and trained across 100 epochs) achieved the optimal balance[cite: 1].
* **Key Takeaway:** The combination of proper oversampling/class weighting with the RMSprop optimizer allowed the network to capture complex non-linear fraudulent patterns effectively, resulting in a high **Recall score** critical for catching real-world financial fraud[cite: 1].
