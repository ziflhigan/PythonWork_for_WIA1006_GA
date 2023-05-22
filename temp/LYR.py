import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


# Load the dataset
data = pd.read_csv('HousePricing.csv')

data = pd.get_dummies(data, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'], drop_first=True)

# Analyze the dataset
print(data.head())
print(data.shape)
print(data.describe())
print(data.info())

# Identify features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Handle missing values
data.isnull().sum()
df = data.apply(pd.to_numeric, errors='coerce')
df = df.fillna(df.mean())

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Implement polynomial regression model
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

# Evaluate linear regression model
y_pred = lin_reg.predict(X_test)
print('Mean Squared Error (MSE):', mean_squared_error(y_test, y_pred))
print('R-squared (R^2) Score:', r2_score(y_test, y_pred))

# Evaluate polynomial regression model
X_poly_test = poly_reg.fit_transform(X_test)
y_pred_2 = lin_reg_2.predict(X_poly_test)
print('Mean Squared Error (MSE):', mean_squared_error(y_test, y_pred_2))
print('R-squared (R^2) Score:', r2_score(y_test, y_pred_2))

# Visualize results
plt.scatter(y_test, y_pred)
plt.plot(y_test, y_test, color='red')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Linear Regression')
plt.show()

plt.scatter(y_test, y_pred_2)
plt.plot(y_test, y_test, color='red')
plt.xlabel('Actual House Prices')
plt.ylabel('Predicted House Prices')
plt.title('Polynomial Regression')
plt.show()