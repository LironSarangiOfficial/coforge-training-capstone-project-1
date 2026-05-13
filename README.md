# Loan Default Prediction

## 📌 Overview
This project builds a machine learning model to predict whether a loan will default using financial, demographic, and loan-related features.

---

## 📂 Dataset
- ~1.4 lakh records  
- Features include:
  - Loan amount, income, credit score  
  - Interest rate, LTV, debt-to-income ratio  
  - Gender, region, loan type, etc.

---

## ⚙️ Workflow

### 1. Data Cleaning
- Removed invalid entries (e.g., incorrect property values)
- Handled missing values
- Fixed inconsistent categorical values
- Dropped useless columns (ID, year)

---

### 2. Exploratory Data Analysis (EDA)
- Feature vs target (Status) analysis
- Distribution and correlation checks
- Outlier detection and handling
- Identified key financial drivers

---

### 3. Feature Engineering
- Encoded categorical variables
- Scaled numerical features
- Fixed LTV calculation issues
- Added derived features (e.g., negative interest spread flag)

---

### 4. Model Training
Models trained:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- KNN  
- Naive Bayes  

Used GridSearchCV for hyperparameter tuning and model selection based on F1-score.

---

### 5. Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- ROC-AUC Curve  

---

## 🏆 Best Model
- Selected automatically using highest F1-score  
- Model saved at: `./outputs/best_loan_default_model.pkl`

---

## 📦 Model Serialization
- Model saved using joblib
- Reloaded and validated for prediction consistency

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
```

Run the notebook/script to:
- Train models
- Evaluate performance
- Save the best model

---

## ✅ Key Insights
- Credit score, LTV, income → strong predictors  
- Data imbalance heavily affects performance  
- Threshold tuning improves recall significantly  

---

## 📌 Future Improvements
- Use advanced models (XGBoost, LightGBM)  
- Better feature selection  
- Deploy using FastAPI or Streamlit  

---

