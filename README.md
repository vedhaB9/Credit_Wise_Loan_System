# CreditWise Loan System

## 📌 Project Overview

CreditWise Loan System is a machine learning project that predicts whether a loan application will be approved based on applicant and loan-related information.

The project focuses on data preprocessing, exploratory data analysis, feature encoding, multicollinearity analysis, feature scaling, and Logistic Regression for loan approval prediction.

## 🎯 Objective

The main objective of this project is to build a machine learning classification system that can predict loan approval outcomes from applicant and loan-related features.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 🔄 Project Workflow

1. Load the loan approval dataset
2. Remove unnecessary columns
3. Handle missing values
4. Perform exploratory data analysis
5. Encode categorical features
6. Analyze feature correlations
7. Split the dataset into training and testing sets
8. Apply feature scaling
9. Train a Logistic Regression model
10. Evaluate the model using classification metrics

## 🧹 Data Preprocessing

The dataset is preprocessed before training the machine learning model.

### Missing Value Handling

- Numerical missing values are handled using the **mean** strategy.
- Categorical missing values are handled using the **most frequent** strategy.

### Feature Removal

The `Gender` column is removed during preprocessing.

The `Applicant_ID` column is excluded from the machine learning features because it is an identifier rather than a predictive feature.

## 🔤 Feature Encoding

Categorical variables are converted into numerical representations using encoding techniques.

### Label Encoding

Label Encoding is applied to:

- Education Level
- Loan Approved

### One-Hot Encoding

One-Hot Encoding is applied to:

- Employer Category
- Loan Purpose
- Marital Status
- Employment Status
- Property Area

The first category is dropped to reduce redundant features.

## 📊 Exploratory Data Analysis

Exploratory analysis is performed to understand the dataset and loan approval distribution.

Visualizations include:

- Loan approval distribution
- Loan purpose distribution
- Correlation heatmap

## 🔎 Correlation Analysis

A correlation heatmap is used to examine relationships between numerical features and identify potential multicollinearity among features.

## ⚙️ Feature Scaling

StandardScaler is used to standardize the input features before training the Logistic Regression model.

The scaler is fitted only on the training data and then applied to both training and testing data.

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression is used as the classification algorithm to predict whether a loan application will be approved.

The model is trained using the preprocessed and scaled training data.

## 📈 Model Evaluation

The model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- Confusion Matrix

These metrics help evaluate the model's classification performance.

## 🚀 Future Improvements

- Compare Logistic Regression with other classification algorithms
- Perform hyperparameter tuning
- Add cross-validation
- Analyze feature importance
- Improve model evaluation using additional classification metrics
- Build an interactive loan approval prediction interface

