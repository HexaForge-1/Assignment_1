import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 1. Dataset Initialization
data = {
    'Experience': [2, 5, 1, 8, 4, 10, 3, 6, 7, 2],
    'TrainingHours': [40, 60, 20, 80, 50, 90, 30, 70, 75, 25],
    'WorkingHours': [38, 42, 35, 45, 40, 48, 37, 44, 46, 36],
    'Projects': [3, 6, 2, 8, 5, 9, 4, 7, 7, 3],
    'Productivity': [62, 78, 55, 88, 72, 92, 65, 82, 85, 60]
}
df = pd.DataFrame(data)

# 2. Identify Importance (Standardized Linear Model)
X = df[['Experience', 'TrainingHours', 'WorkingHours', 'Projects']]
y = df['Productivity']

# Scale features to mean=0, std=1 to compare coefficients directly
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model_main = LinearRegression().fit(X_scaled, y)

# 3. Test Diminishing Returns (Quadratic Model)
df['WH_sq'] = df['WorkingHours']**2
X_quad = df[['WorkingHours', 'WH_sq']]
model_quad = LinearRegression().fit(X_quad, y)

# --- Display Results ---
importance = pd.Series(model_main.coef_, index=X.columns).abs().sort_values(ascending=False)
print("=== Model Results ===")
print("Feature Importance (Absolute Weights):")
print(importance)
print(f"\nQuadratic Coefficient for Working Hours: {model_quad.coef_[1]:.4f}")
