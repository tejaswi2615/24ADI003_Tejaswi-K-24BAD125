import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

print("Tejaswi - 24BAD125")

# Load dataset
df = pd.read_csv("c:/Users/Teju/Downloads/auto-mpg.csv")

# Clean dataset
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df.dropna(inplace=True)

#Select feature and target
X = df[['horsepower']]
y = df['mpg']

#Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#train - test  split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

degrees = [2, 3, 4]
train_errors = []
test_errors = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    # Errors
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_errors.append(train_mse)
    test_errors.append(test_mse)
    
    rmse = np.sqrt(test_mse)
    r2 = r2_score(y_test, y_test_pred)
    
    print(f"Degree {degree} → Train MSE: {train_mse:.2f}, "
          f"Test MSE: {test_mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")


X_range = np.linspace(X.min(), X.max(), 100)
X_range_df = pd.DataFrame(X_range, columns=['horsepower'])
X_range_scaled = scaler.transform(X_range_df)

plt.figure(figsize=(7,5))
plt.scatter(X, y, color='black', label='Actual Data')

for degree in degrees:
    poly = PolynomialFeatures(degree)
    X_poly_range = poly.fit_transform(X_range_scaled)
    
    model = LinearRegression()
    model.fit(poly.fit_transform(X_scaled), y)
    
    y_curve = model.predict(X_poly_range)
    plt.plot(X_range, y_curve, label=f'Degree {degree}')

plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Regression Curve Fitting")
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(degrees, train_errors, marker='o', label='Training Error')
plt.plot(degrees, test_errors, marker='o', label='Testing Error')

plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Training vs Testing Error Comparison")
plt.legend()
plt.show()

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X_scaled)

ridge = Ridge(alpha=10)
ridge.fit(X_poly, y)


