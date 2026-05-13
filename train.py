import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split, GridSearchCV

# Import the Logistic Regression model and evaluation metrics from scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# SMOTE for class imbalance
from imblearn.over_sampling import SMOTE

import os
import joblib
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('./data/Loan_Default.csv')

df = df.drop_duplicates()
df = df.drop(columns=['Upfront_charges', 'ID', 'year'])

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'string']).columns


# Creating imputer for simple missing data handling
num_imputer_mean = SimpleImputer(strategy='mean')
num_imputer_median = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Pre processing
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
df['rate_of_interest'] = df['rate_of_interest'].replace(0, np.nan)
df['income'] = df['income'].replace(0, np.nan)
df[['rate_of_interest', 'Interest_rate_spread']] = num_imputer_median.fit_transform(df[['rate_of_interest', 'Interest_rate_spread']])
df[['term', 'income']] = num_imputer_mean.fit_transform(df[['term', 'income']]).astype('int64')
df[['property_value','dtir1']] = num_imputer_median.fit_transform(df[['property_value','dtir1']])
# since property_value null values are handled, we would recalculate LTV values
df['LTV'] = df['loan_amount']/df['property_value']
df['age'] = df['age'].replace({'<25':'0-25', '>74':'75+'})
# handling outliers
df = df[df['loan_amount'] <= df['property_value']]



# Encode categorical variables
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# The scaling of numerical features will be done after splitting the data into training and testing sets to avoid data leakage.
# The code for scaling will be added in the next steps after splitting the data.
X = df.drop('Status', axis=1)
Y = df['Status']


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
# Stratify is used to ensure that the distribution of the target variable 'Status' is maintained in both the training and testing sets, 
# which is important for imbalanced datasets.

scaler = StandardScaler()
# x_train_res_scaled = scaler.fit_transform(x_train_res)
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

models = {
    "log_reg": {
        "model": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "params": {
            "C": [0.1, 1, 10]
        }
    },

    "decision_tree": {
        "model": DecisionTreeClassifier(class_weight='balanced'),
        "params": {
            "max_depth": [5, 10, 15, 20],
            "min_samples_split": [2, 5]
        }
    },

    "random_forest": {
        "model": RandomForestClassifier(class_weight='balanced'),
        "params": {
            "n_estimators": [100, 120, 140, 160, 180, 200],
            "max_depth": [10, 20]
        }
    },

    "knn": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7],
            "weights": ['uniform', 'distance']
        }
    },

    "naive_bayes": {
        "model": GaussianNB(),
        "params": {}   # no grid needed
    }
}

results = []
best_model = None
best_score = 0

for name, config in models.items():
    print(f"Running {name}...")

    grid = GridSearchCV(
        config["model"],
        config["params"],
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    # grid.fit(x_train_res_scaled, y_train_res)
    grid.fit(x_train_scaled, y_train)

    score = grid.best_score_

    results.append({ "model": name, "best_score": score, "best_params": grid.best_params_
    })

    # Track best model
    if score > best_score:
        best_score = score
        best_model = grid.best_estimator_
        results_df = pd.DataFrame(results).sort_values(by='best_score', ascending=False)


os.makedirs("./model", exist_ok=True)

with open('model/best_loan_model.pkl', 'wb') as file:
    joblib.dump((scaler, best_model), file)





