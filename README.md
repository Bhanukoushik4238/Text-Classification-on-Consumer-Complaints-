# Task 5 – Text Classification using XGBoost (Google Colab Implementation)

A Natural Language Processing (NLP) project that focuses on **multi-class text classification** using the **XGBoost algorithm**.  
This task demonstrates preprocessing, model training, evaluation, and visualization of results, all implemented in **Google Colab** for an efficient workflow on limited resources.

---

## Overview

The goal of this task was to **build a machine learning model that can classify text data into multiple categories**.  
The project uses the **XGBoost classifier**, one of the most powerful gradient boosting models, optimized for both speed and accuracy.

---

## Features

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

## Model Used

**XGBoost Classifier**

- Fast and efficient gradient boosting algorithm  
- Handles large feature spaces (ideal for TF-IDF vectors)  
- Outperforms traditional classifiers like Logistic Regression in many NLP tasks  
- Supports **class imbalance handling** using scale_pos_weight or custom class weights  

---

##  Dataset

- **Dataset Used:** News Category Dataset  
- **Total Samples Used:** 20,000 (subset for performance optimization in Colab)  
- **Classes:** Multiple categories (e.g., business, politics, technology, entertainment, etc.)  
- **Format:** CSV file with two main columns:
  - `text` — news content or sentence  
  - `label` — category of the text  

**Dataset Link:**  
[ Download Dataset from Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset)  

---

##  Steps to Reproduce (Google Colab)

1. Open **Google Colab**
2. Upload the notebook and dataset (or mount Google Drive)
3. Install required dependencies:
   ```bash
   !pip install xgboost scikit-learn matplotlib pandas numpy

## Screen Shots 
<img width="1920" height="1200" alt="{9DA2955E-C0A6-4E2F-8407-4E5907BDB8A9}" src="https://github.com/user-attachments/assets/ff1e4f02-f06e-4440-a184-10c96ff22e98" />
<img width="1871" height="870" alt="{BE3915FF-FCD6-4250-B196-49F662ADA719}" src="https://github.com/user-attachments/assets/458146e7-f290-45c9-90f2-fa8c6ee63db2" />
<img width="1920" height="1200" alt="{CDCE6774-38A2-44B6-9D2D-ECAB3D93DFD9}" src="https://github.com/user-attachments/assets/28683564-b119-4af1-aa11-9eb4827a60ec" />
<img width="1920" height="1200" alt="{27E5922F-726D-4D7C-8268-497BD13DB18B}" src="https://github.com/user-attachments/assets/6e715733-ae68-45a7-ba8c-bc42ab9793d5" />
<img width="1917" height="1197" alt="{CC067A6A-93EC-474F-A902-F977CAB834C8}" src="https://github.com/user-attachments/assets/a43cd66a-d8d6-4b99-8985-5b0e855a6e36" />
<img width="1920" height="1200" alt="{934E567B-0102-4CE1-B3DC-B0610AC500B2}" src="https://github.com/user-attachments/assets/8f702a72-8134-4f9b-93b8-6eb90085c637" />
<img width="1920" height="1199" alt="{A7B2DA4C-F408-4DF3-BC24-536CC3ADCEF1}" src="https://github.com/user-attachments/assets/67348c50-05da-4508-8720-76b323b5c560" />








