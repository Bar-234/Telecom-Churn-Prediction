import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML import
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, accuracy_score,
                             precision_score ,recall_score , f1_score )
from sklearn.impute import SimpleImputer
import xgboost as xgb

import pickle
import os

from config import *

print("="*80)
print("TELECOM CUSTOMER CHURN PREDICTION")
print("="*80)

# Step 1 : Load Data
print("\n[1] Cleaning data...")
df = pd.read_csv(DATA_PATH)
print(f" Loaded {len(df)} customers with {len(df.columns)} features")
print(f" Churn rate : {df['Churn'].value_counts(normalize=True)['Yes']*100:.1f}%")

# Step 2 : Data cleaning
print("\n[2] Cleaning data")

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(),inplace=True )

# Drop customerID
df.drop('customerID', axis=1, inplace=True, errors='ignore')

# Step 3 : Feature Engineering
print("\n[3] Engineering feature...")

# Create 6 Business feature
df['CustomerValueScore'] = df["MonthlyCharges"] + df['tenure']
df['servicesCount']= (df[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport','StreamingTV','StreamingMovies']]== 'Yes').sum(axis=1)
df['AvgMonthlyValue'] = df['TotalCharges'] / ( df['tenure'] + 1)
df['IsNewCustomer'] = (df['tenure']<= 6).astype(int)
df['HasPremiumServices'] = (df['servicesCount'] >= 4).astype(int)
df['TenureCatrgory'] = pd.cut(df['tenure'], 
                                    bins= [0,12, 24, 48,72],
                                    labels=['New','Establishing','Mature','Loyal'])

# Step 4: Encoding
print("\n[4] Encoding features...")

# Binary encoding 
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No' :0}) 

# One-hot encoding 
nominal_cols= ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
               'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 
                'Contract','PaymentMethod']

df_model = pd.get_dummies(df, columns= nominal_cols, drop_first= True) 

# Encode target 
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})
df= df.dropna(subset=['Churn'])

print (f" Final feature count : {len(df.columns)-1}")

# Step 5 : Train-Test Split
print("n[5] Splitting data...")

X= df_model.drop('Churn', axis=1)
y= df_model['Churn']

X = pd.get_dummies(X, drop_first=True)
imputer= SimpleImputer(strategy= 'mean')
X= pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
                                                     stratify=y)

# Scale feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print (f" Train : {X_train.shape[0]}| Test: {X_test.shape[0]}")

# Step 6 : TRAIN MODELS
print("\n[6] Training models... ")

models= {}
results = {}

# Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr

# Random forest
print (" Training Random Forest....")
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
models['Random Forest'] = rf

# Step 7: Evaluate Model
print ("\n[7] Evaluating models....")
















