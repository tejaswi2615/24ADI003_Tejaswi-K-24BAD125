import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Tejaswi K - 24BAD125")

df = pd.read_csv("c:\\Users\\Teju\\Downloads\\income_random_forest.csv")

# Add noise
np.random.seed(0)
noise = np.random.normal(0, 2, df.shape)
df = df + noise
df['Income'] = (df['Income'] > 0.5).astype(int)

X = df.drop("Income", axis=1)
y = df["Income"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

trees = [10, 50, 100, 200]
accuracies = []

for t in trees:
    rf = RandomForestClassifier(n_estimators=t, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))
    print(f"n_estimators={t}  ->  Accuracy: {accuracies[-1]:.4f}")

plt.figure(figsize=(6, 5))
plt.plot(trees, accuracies, marker='o', lw=2)
for x, y_val in zip(trees, accuracies):
    plt.text(x, y_val + 0.005, f'{y_val:.3f}', ha='center', fontsize=9)
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Number of Trees")
plt.tight_layout()
plt.show()

rf_best = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
rf_best.fit(X_train, y_train)

plt.figure(figsize=(6, 5))
feat_imp = rf_best.feature_importances_
feat_names = X.columns.tolist()
idx = np.argsort(feat_imp)
plt.barh([feat_names[i] for i in idx], feat_imp[idx])
plt.title("Feature Importance")
plt.tight_layout()
plt.show()