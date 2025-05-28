# Credit Card Fraud Detection with AdaBoost

This project applies AdaBoost classification to detect fraudulent transactions in credit card datasets. The notebook explores hyperparameter tuning, evaluation metrics, and visualization of fraud prediction performance.

Developed as part of the **Machine Learning** course in the Bachelor of Information Sciences – Major in Data Science at Massey University.

## 📘 Course Context

- **Course:** Machine Learning  
- **Semester:** Semester 1, 2022  
- **University:** Massey University  
- **Program:** Bachelor of Information Sciences – Major in Data Science

## 🔍 Project Objectives

- Detect fraud in financial transactions using AdaBoost
- Deal with class imbalance using performance metrics beyond accuracy
- Perform hyperparameter tuning with cross-validation
- Evaluate model with confusion matrix, ROC and PR curves

## 🧪 Key Features

- `AdaBoostClassifier` with `DecisionTreeClassifier` as base estimator
- Use of:
  - `StratifiedKFold` for cross-validation
  - `GridSearchCV` for hyperparameter optimization
  - `classification_report`, `confusion_matrix`, `roc_auc_score`, `precision_recall_curve`
- Visualization of model performance with seaborn and matplotlib

## 🧰 Technologies Used

- Python 3  
- scikit-learn  
- pandas  
- matplotlib / seaborn  
- Jupyter Notebook

## ▶️ How to Run

1. Open `adaboosassig.ipynb` in Jupyter Notebook.
2. Execute all cells to follow the training and evaluation steps.

## 📌 Notes

- The dataset used is anonymized and was likely provided by the course instructor.
- Class imbalance is significant (fraud cases are rare), so evaluation focuses on `precision`, `recall`, and `f1-score`.