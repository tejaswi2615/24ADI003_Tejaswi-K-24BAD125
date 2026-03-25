import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

print("Tejaswi K - 24BAD125")

df = pd.read_csv("c:\\Users\\Teju\\Downloads\\diabetes_bagging.csv")

# Add noise to data
np.random.seed(0)
noise = np.random.normal(0, 2.5, df.shape)
df = df + noise
df['Outcome'] = (df['Outcome'] > 0.5).astype(int)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

bag = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=3),
                        n_estimators=50, random_state=42)
bag.fit(X_train, y_train)
y_pred_bag = bag.predict(X_test)

acc_dt = accuracy_score(y_test, y_pred_dt)
acc_bag = accuracy_score(y_test, y_pred_bag)

print("Decision Tree Accuracy:", acc_dt)
print("Bagging Accuracy:", acc_bag)

plt.figure(figsize=(6, 5))
bars = plt.bar(["Decision Tree", "Bagging"], [acc_dt, acc_bag],
               color=["#f78166", "#3fb950"], width=0.4)
for bar, val in zip(bars, [acc_dt, acc_bag]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', fontweight='bold')
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1.15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_bag)
ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"]).plot(cmap="Blues")
plt.title("Confusion Matrix - BaggingClassifier")
plt.tight_layout()
plt.show()