import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Dataset Initialization
data = {
    'Age': [28, 45, 35, 50, 30, 42, 26, 48, 38, 55],
    'AnnualIncome': [6.5, 12, 8, 15, 7, 10, 5.5, 14, 9, 16],
    'CreditScore': [720, 680, 750, 640, 710, 660, 730, 650, 700, 620],
    'LoanAmount': [5, 10, 6, 12, 5, 9, 4, 11, 7, 13],
    'LoanTerm': [5, 10, 7, 15, 5, 10, 4, 12, 8, 15],
    'EmploymentType': ['Salaried', 'Self-Employed', 'Salaried', 'Self-Employed', 'Salaried', 'Salaried', 'Salaried', 'Self-Employed', 'Salaried', 'Self-Employed'],
    'LoanDefault': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# 2. Preprocessing
# Encode categorical 'EmploymentType' (Salaried=0, Self-Employed=1)
le = LabelEncoder()
df['EmploymentType'] = le.fit_transform(df['EmploymentType'])

# Define Features and Target
X = df.drop('LoanDefault', axis=1)
y = df['LoanDefault']

# KNN is distance-based, so Scaling is mandatory
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Model Building (Using k=3 for this small dataset)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_scaled, y)

# 4. Quick Evaluation on training data (Simplified for the example)
y_pred = knn.predict(X_scaled)

print("=== KNN Loan Default Model ===")
print(f"Model Accuracy on training data: {knn.score(X_scaled, y)*100:.0f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))
print("\nFeature Importance (Correlation with Default):")
print(df.corr()['LoanDefault'].sort_values(ascending=False))
