import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("Tejaswi K - 24BAD125")

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

# 3. Data Inspection
print("First 5 Samples:\n", pd.DataFrame(X, columns=feature_names).head())
print("\nClass Distribution:", np.bincount(y))

# 4. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
X_scaled, y, test_size=0.2, random_state=42)

# 6. Train Gaussian Naïve Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 7. Predict
y_pred = gnb.predict(X_test)

# 8. Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# 9. Compare Predictions
print("\nActual:", y_test[:10])
print("Predicted:", y_pred[:10])

# 10. Class Probabilities
probabilities = gnb.predict_proba(X_test[:5])
print("\nClass Probabilities (First 5 Samples):\n", probabilities)

# 11. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=class_names)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Select Petal Length & Petal Width
X_2d = X_scaled[:, 2:4]
X_train2, X_test2, y_train2, y_test2 = train_test_split(
X_2d, y, test_size=0.2, random_state=42)

gnb2 = GaussianNB()
gnb2.fit(X_train2, y_train2)

# Create mesh grid
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
np.arange(y_min, y_max, 0.02))

Z = gnb2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
plt.xlabel("Petal Length (scaled)")
plt.ylabel("Petal Width (scaled)")
plt.title("Decision Boundary (Gaussian NB)")
plt.show()

# 13. Probability Distribution Plot (Gaussian assumption)
for i in range(3):
    plt.figure()
    plt.hist(X[y == i, 2], bins=10)
    plt.title(f"Petal Length Distribution - {class_names[i]}")
    plt.xlabel("Petal Length")
    plt.ylabel("Frequency")
    plt.show()

# 14. Optional Comparison with Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nLogistic Regression Accuracy:",
accuracy_score(y_test, y_pred_lr))