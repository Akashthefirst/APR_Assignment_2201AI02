# Heart Disease Classification Using Machine Learning

A comparative analysis of Logistic Regression and Support Vector Machines for heart disease prediction using clinical data.

## 📋 Project Overview

This project implements and evaluates machine learning models to classify heart disease status using clinical and diagnostic data. Two supervised learning algorithms, Logistic Regression and Support Vector Machines (SVM) with an RBF kernel, are applied. The goal is to develop automated tools that support early and accurate heart disease diagnosis.

## 📊 Dataset Information

The dataset is a **multivariate clinical dataset** containing **14 attributes**:

| Feature | Description |
|---------|-------------|
| Age | Patient age |
| Sex | Gender |
| Chest Pain Type | Type of chest pain experienced |
| Resting Blood Pressure | Blood pressure at rest |
| Serum Cholesterol | Cholesterol levels |
| Fasting Blood Sugar | Blood sugar after fasting |
| Resting ECG Results | Electrocardiographic results at rest |
| Max Heart Rate | Maximum heart rate achieved |
| Exercise-Induced Angina | Angina triggered by exercise |
| Oldpeak | ST depression induced by exercise |
| ST Slope | Slope of peak exercise ST segment |
| Major Vessels | Number of major vessels |
| Thalassemia | Thalassemia test results |
| **Target** | Heart disease presence (0: No, 1: Yes) |

## 🔧 Features and Methods

- **Data preprocessing** involves handling missing values through mean/median imputation for numerical features and mode imputation for categorical features, binary conversion of the target variable, and standard scaling
- **Logistic Regression and SVM classifiers** are trained and evaluated using an 80-20 train-test split with stratified sampling
- **Principal Component Analysis (PCA)** reduces feature dimensionality for visualization of model decision boundaries
- **Performance metrics** include precision, recall, F1-score, and accuracy, with comparisons visualized through evaluation charts

## 🤖 Machine Learning Models

### Logistic Regression
- **Approach**: Probabilistic binary classifier using logistic function
- **Strengths**: Interpretable coefficients, probability outputs
- **Use Case**: Linear relationships, risk estimation

### Support Vector Machine (SVM)
- **Kernel**: Radial Basis Function (RBF)
- **Approach**: Optimal hyperplane with maximum margin
- **Strengths**: Handles non-linear patterns, robust to outliers

## 📈 Model Performance

| Model | Test Accuracy | AUC Score |
|-------|---------------|-----------|
| **Logistic Regression** | **0.89** | **0.93** |
| **SVM (RBF)** | 0.86 | 0.92 |

## 🔍 Principal Component Analysis (PCA)

- **Purpose**: Dimensionality reduction and data visualization
- **Benefits**: 
  - Simplifies complex datasets
  - Reduces computational complexity
  - Enables 2D/3D visualization
  - Helps understand data patterns and decision boundaries

## 📁 Project Contents

- **APR_Assignment_1.ipynb**: Jupyter notebook containing the complete implementation, including data preprocessing, model training, hyperparameter tuning, evaluation, and visualization
- **UCL Heart Disease Dataset**: The dataset used, containing clinical and diagnostic features for heart disease classification
- **APR_Assignment_1_report.pdf**: A detailed project report covering theory, methodology, results, and discussion of the models

## 🚀 How to Run

1. **Ensure all dependencies are installed**: Python 3.x, scikit-learn, matplotlib, numpy, pandas
2. **Load the UCL Heart Disease dataset** in the notebook environment
3. **Execute cells sequentially** in APR_Assignment_1.ipynb to preprocess data, train models, tune hyperparameters, evaluate, and visualize results
4. **View the report** (APR_Assignment_1_report.pdf) for detailed explanations and analysis

## 📊 Results Summary

- **Both Logistic Regression and SVM achieve high accuracy** in heart disease classification
- **Logistic Regression** shows slightly higher test accuracy (0.89) and AUC (0.93) compared to SVM
- **SVM** demonstrates robustness with good performance (0.86 accuracy, 0.92 AUC) and ability to handle complex patterns
- **PCA effectively visualizes** the decision boundary and data patterns in reduced dimensional space
- **Performance comparison charts** provide intuitive insights on model strengths and classification capabilities

## 👨‍💻 Author

**Akash Sinha** (2201AI02)  
*September 19, 2025*

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- Clinical domain experts for feature interpretation guidance
- Open-source machine learning community

---

**Note**: This project is for educational and research purposes. Clinical decisions should always involve qualified healthcare professionals.
