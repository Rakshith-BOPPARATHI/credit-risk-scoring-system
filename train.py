"""Training script for Credit Risk Scoring System."""
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from src.model import CreditRiskPreprocessor

print("\n" + "="*70)
print("CREDIT RISK SCORING SYSTEM - INITIALIZATION")
print("="*70)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Status: System Ready")
print("="*70)

# Generate sample dataset
print("\n" + "="*70)
print("GENERATING SAMPLE TRAINING DATA")
print("="*70)

np.random.seed(42)
n_samples = 1000

data = {
    'Income': np.random.normal(60000, 20000, n_samples),
    'LoanAmount': np.random.normal(150000, 50000, n_samples),
    'CreditHistory': np.random.uniform(0, 30, n_samples),
    'WorkExperience': np.random.choice(['0-2 years', '2-5 years', '5+ years'], n_samples),
    'HomeOwnership': np.random.choice(['Rent', 'Mortgage', 'Own'], n_samples)
}

df = pd.DataFrame(data)

# Generate target variable
df['LoanToIncome'] = df['LoanAmount'] / df['Income']
default_probability = 1 / (1 + np.exp(-(df['LoanToIncome'] * 2 - df['CreditHistory'] / 10 - 1)))
df['Default'] = (np.random.random(n_samples) < default_probability).astype(int)
df.drop('LoanToIncome', axis=1, inplace=True)

print(f"\n✅ Generated {n_samples} loan application records")
print(f"Default rate: {df['Default'].mean()*100:.2f}%")
print(f"\nDataset shape: {df.shape}")

# Train model
print("\n" + "="*70)
print("TRAINING LOGISTIC REGRESSION MODEL")
print("="*70)

X = df[['Income', 'LoanAmount', 'CreditHistory', 'WorkExperience', 'HomeOwnership']]
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n✅ Train set: {len(X_train)} | Test set: {len(X_test)}")

preprocessor = CreditRiskPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_processed[preprocessor.feature_names], y_train)
print("✅ Model training complete!")

# Evaluate
y_pred = model.predict(X_test_processed[preprocessor.feature_names])
y_pred_proba = model.predict_proba(X_test_processed[preprocessor.feature_names])[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n✅ Accuracy: {accuracy*100:.2f}% | ROC-AUC: {roc_auc:.4f}")

# Save model coefficients
scaling_params = preprocessor.get_scaling_params()
coefficients = model.coef_[0]
intercept = model.intercept_[0]

model_data = {
    'intercept': float(intercept),
    'coefficients': {name: float(coef) for name, coef in zip(preprocessor.feature_names, coefficients)},
    'scaling_params': {
        'means': {name: float(mean) for name, mean in zip(preprocessor.feature_names, scaling_params['means'])},
        'stds': {name: float(std) for name, std in zip(preprocessor.feature_names, scaling_params['stds'])}
    },
    'performance': {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc)
    }
}

os.makedirs('models', exist_ok=True)
with open('models/model_coefficients.json', 'w') as f:
    json.dump(model_data, f, indent=2)

print("\n✅ Model coefficients saved to models/model_coefficients.json")
print("\n" + "="*70)
print("✅ TRAINING COMPLETE")
print("="*70)
