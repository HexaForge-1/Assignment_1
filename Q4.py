import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# 1. Generate Sample Data
data = {
    'Area': [1500, 1800, 2400, 3000, 3500, 4000, 2000, 2800, 3200, 4500],
    'Bedrooms': [2, 3, 3, 4, 4, 5, 2, 3, 4, 5],
    'Price': [300, 350, 450, 520, 580, 650, 380, 480, 540, 720] # Price in $1000s
}
df = pd.DataFrame(data)

# 2. Fit a Linear Regression Model
X = df[['Area', 'Bedrooms']]
y = df['Price']
model = LinearRegression().fit(X, y)

# 3. Create a Grid for the Surface Plane
x_surf, y_surf = np.meshgrid(np.linspace(df.Area.min(), df.Area.max(), 10), 
                             np.linspace(df.Bedrooms.min(), df.Bedrooms.max(), 10))
onlyX = pd.DataFrame({'Area': x_surf.ravel(), 'Bedrooms': y_surf.ravel()})
fittedY = model.predict(onlyX)
fittedY = fittedY.reshape(x_surf.shape)

# 4. Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of actual data
ax.scatter(df['Area'], df['Bedrooms'], df['Price'], c='red', marker='o', alpha=1, label='Actual House Data')

# Surface plot of regression model
ax.plot_surface(x_surf, y_surf, fittedY, color='blue', alpha=0.3, label='Regression Plane')

# Labels
ax.set_xlabel('Area (sq ft)')
ax.set_ylabel('No. of Bedrooms')
ax.set_zlabel('Price ($1000s)')
ax.set_title('3D Regression Model: House Price Prediction')

plt.legend()
plt.show()
