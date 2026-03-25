import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE

print("Tejaswi K - 24BAD125")

df = pd.read_csv("c:\\Users\\Teju\\Downloads\\fraud_smote.csv")

# Add noise to feature columns only
np.random.seed(0)
feature_cols = [col for col in df.columns if col != 'Fraud']
df[feature_cols] = df[feature_cols] + np.random.normal(0, 0.5, df[feature_cols].shape)

X = df.drop("Fraud", axis=1)
y = df["Fraud"]

print("Before SMOTE:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

print("After SMOTE:\n", pd.Series(y_res).value_counts())

plt.figure(figsize=(6, 5))
plt.bar(['Normal', 'Fraud'], y_train.value_counts().sort_index(),
        color='steelblue', alpha=0.7)
plt.title("Class Distribution Before SMOTE")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.bar(['Normal', 'Fraud'], pd.Series(y_res).value_counts().sort_index(),
        color='orange', alpha=0.7)
plt.title("Class Distribution After SMOTE")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
model.fit(X_res, y_res)

y_prob = model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='blue', lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (After SMOTE)")
plt.tight_layout()
plt.show()