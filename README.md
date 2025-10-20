# Task 5 – Text Classification using XGBoost (Google Colab Implementation)

A Natural Language Processing (NLP) project that focuses on **multi-class text classification** using the **XGBoost algorithm**.  
This task demonstrates preprocessing, model training, evaluation, and visualization of results, all implemented in **Google Colab** for an efficient workflow on limited resources.

---

## 📘 Overview

The goal of this task was to **build a machine learning model that can classify text data into multiple categories**.  
The project uses the **XGBoost classifier**, one of the most powerful gradient boosting models, optimized for both speed and accuracy.

---

## ✨ Features

- **Data Loading and Preprocessing**
  - Loaded a dataset containing text samples and category labels  
  - Used only **20,000 samples** to ensure efficient training within Google Colab environment  
  - Cleaned text data by removing unwanted characters, stopwords, and performing tokenization  

- **Vectorization**
  - Converted raw text into numerical format using **TF-IDF (Term Frequency–Inverse Document Frequency)** vectorizer  

- **Model Training (XGBoost)**
  - Implemented **XGBoostClassifier** with tuned hyperparameters  
  - Trained the model on preprocessed text data  
  - Used **class weights** to handle imbalanced dataset classes  

- **Evaluation**
  - Computed **Confusion Matrix**, **Classification Report**, and **Accuracy Score**
  - Visualized confusion matrix using **Matplotlib**  
  - Displayed accuracy in **percentage format**  

- **Colab-Based Implementation**
  - Entire pipeline executed in **Google Colab**  
  - Saved model, visualizations, and metrics directly within the Colab environment  

---

## 🧠 Model Used

**XGBoost Classifier**

- Fast and efficient gradient boosting algorithm  
- Handles large feature spaces (ideal for TF-IDF vectors)  
- Outperforms traditional classifiers like Logistic Regression in many NLP tasks  
- Supports **class imbalance handling** using scale_pos_weight or custom class weights  

---

## 🧩 Dataset

- **Dataset Used:** News Category Dataset  
- **Total Samples Used:** 20,000 (subset for performance optimization in Colab)  
- **Classes:** Multiple categories (e.g., business, politics, technology, entertainment, etc.)  
- **Format:** CSV file with two main columns:
  - `text` — news content or sentence  
  - `label` — category of the text  

**Dataset Link:**  
[🔗 Download Dataset from Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset)  

---

## ⚙️ Steps to Reproduce (Google Colab)

1. Open **Google Colab**
2. Upload the notebook and dataset (or mount Google Drive)
3. Install required dependencies:
   ```bash
   !pip install xgboost scikit-learn matplotlib pandas numpy
