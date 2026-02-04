import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

print("Tejaswi - 24BAD125")
#read the csv file
df = pd.read_csv("c:/Users/Teju/Downloads/StudentsPerformance.csv")

#label encoding
le = LabelEncoder()
df['parental level of education'] = le.fit_transform(df['parental level of education'])
df['test preparation course'] = le.fit_transform(df['test preparation course'])

#calculating target variable
df['final_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

# Create additional features as not present in the dataset
np.random.seed(42)
df['study_hours'] = np.random.randint(1, 8, size=len(df))
df['attendance'] = np.random.randint(60, 100, size=len(df))
df['sleep_hours'] = np.random.randint(4, 9, size=len(df))

# missing values
df['study_hours'].fillna(df['study_hours'].mean(), inplace=True)
df['attendance'].fillna(df['attendance'].mean(), inplace=True)
df['sleep_hours'].fillna(df['sleep_hours'].mean(), inplace=True)


# Select features and apply scaling
X = df[['study_hours',
        'attendance',
        'parental level of education',
        'test preparation course',
        'sleep_hours']]
y = df['final_score']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#Multilinear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

#Predict student performance
y_pred = model.predict(X_test)

# 1Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)

# Analyze regression coefficients
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("\nRegression Coefficients")
print(coefficients)

#Optimization using Ridge and Lasso Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

print("\nRidge Coefficients:", ridge.coef_)
print("Lasso Coefficients:", lasso.coef_)

# Predicted vs Actual exam scores
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Predicted vs Actual Exam Scores")
plt.show()

# Coefficient magnitude comparison
plt.figure(figsize=(6,4))
plt.bar(X.columns, model.coef_)
plt.xticks(rotation=45)
plt.title("Feature Coefficient Magnitudes")
plt.show()

# Residual distribution plot
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()
