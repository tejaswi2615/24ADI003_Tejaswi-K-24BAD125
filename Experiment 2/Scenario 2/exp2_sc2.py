import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, roc_auc_score
)

print("Tejaswi - 24BAD125")

#Read csv file
df = pd.read_csv("c:/Users/Teju/Downloads/LICI - Daily data.csv")
df.head()

#Create binary target variable
df['Price_Movement'] = np.where(df['Close'] > df['Open'], 1, 0)

#Select features and target
features = ['Open', 'High', 'Low', 'Close']
X = df[features]
y = df['Price_Movement']

#Handle missing values
X = X.fillna(X.mean())

#Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

#Predictions
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label="AUC = " + str(round(auc, 2)))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': log_model.coef_[0]
})

sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title("Feature Importance")
plt.show()

# Hyperparameter tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# Optimized model evaluation
best_model = grid.best_estimator_
best_pred = best_model.predict(X_test)

print("Optimized Accuracy:", accuracy_score(y_test, best_pred))
print("Optimized F1 Score:", f1_score(y_test, best_pred))
