# 💳 Credit Risk Prediction

An end-to-end machine learning project for predicting **loan default risk** using the German Credit dataset.

The project covers exploratory data analysis, preprocessing, model comparison, evaluation, feature importance, and an interactive Streamlit application.

## 🚀 Features

- Exploratory Data Analysis (EDA)
- Default-risk driver analysis
- One-hot encoding for categorical variables
- Feature scaling for numerical variables
- Logistic Regression and Random Forest
- Accuracy, Precision, Recall, F1-score and ROC-AUC
- ROC curve comparison
- Confusion matrix
- Random Forest feature importance
- Saved trained model pipeline
- Interactive Streamlit prediction dashboard

## 📊 Dataset

The project uses the public **German Credit dataset**, containing 1,000 applicants and borrower/loan characteristics.

The original target is `credit_risk`:

- `1` = good credit
- `0` = bad credit

For this project, the target is transformed into:

- `0` = lower/default-risk class
- `1` = default-risk class

> **Note:** This is an educational portfolio project. It should not be used as the sole basis for real lending decisions.

## 🧠 Machine Learning Workflow

```text
Raw Dataset
     ↓
Exploratory Data Analysis
     ↓
Feature / Target Preparation
     ↓
Train-Test Split
     ↓
Preprocessing
 ┌───────────────┐
 │ Numerical     │ → StandardScaler
 │ Categorical   │ → OneHotEncoder
 └───────────────┘
     ↓
Model Training
 ┌──────────────────────┐
 │ Logistic Regression  │
 │ Random Forest        │
 └──────────────────────┘
     ↓
Model Evaluation
     ↓
Best Model Selection
     ↓
Saved Model
     ↓
Streamlit Web App
```

## 📈 Evaluation

The project compares models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The best model is selected based on ROC-AUC on the held-out test set.

## 📁 Project Structure

```text
credit-risk-prediction/
│
├── credit_risk_project.py
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── credit_risk_model.joblib
│   ├── feature_columns.csv
│   └── model_info.txt
│
└── outputs/
    ├── eda_default_drivers.png
    ├── roc_curves.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── model_comparison.csv
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/PRIMEAYUSH/credit-risk-prediction.git
cd credit-risk-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🔬 Run the ML Analysis

```bash
python credit_risk_project.py
```

This generates the evaluation charts and saves the trained model inside `models/`.

## 🖥️ Run the Web App

```bash
streamlit run app.py
```

The application lets a user enter applicant information and receive an estimated default-risk probability.

## 🔮 Future Improvements

- Hyperparameter tuning with cross-validation
- Probability calibration
- SHAP-based individual prediction explanations
- Threshold optimization based on business cost
- Model monitoring and drift detection
- Cloud deployment
- Fairness and bias evaluation

## 🛠️ Technologies

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Streamlit` · `Joblib`

## 👨‍💻 Author

**Ayush Narganwe**

GitHub: [PRIMEAYUSH](https://github.com/PRIMEAYUSH)
