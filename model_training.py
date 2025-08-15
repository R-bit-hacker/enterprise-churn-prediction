import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import sys

if len(sys.argv) != 2:
    print("Usage: python model_training.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]

# Load data
df = pd.read_csv(csv_file)

# Basic preprocessing
df = df.dropna()
y = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop('Churn', axis=1)

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and columns
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")
print("Model saved as model.pkl and columns as model_columns.pkl")
