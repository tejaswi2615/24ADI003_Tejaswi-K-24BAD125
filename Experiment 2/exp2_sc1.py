import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

print("Tejaswi -24BAD125")

#read csv file
df = pd.read_csv("c:/Users/Teju/Downloads/bottle.csv")
df.head()
print(df.columns)

#Select features and target variable
features = ['Depthm', 'Salnty', 'O2ml_L']
target = 'T_degC'

X = df[features]
y = df[target]

#Handle missing values (mean imputation)
X = X.fillna(X.mean())
y = y.fillna(y.mean())

#Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

#Predict temperature
y_pred = lr_model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)

# Actual vs Predicted plot
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted Temperature")
plt.show()

#Residual plot
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='red')
plt.xlabel("Predicted Temperature")
plt.ylabel("Residuals")
plt.title("Residual Errors")
plt.show()

#Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("Ridge R²:", r2_score(y_test, ridge_pred))

#Lasso Regression
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
print("Lasso R²:", r2_score(y_test, lasso_pred))
