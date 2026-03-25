import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

print("Tejaswi K - 24BAD125")

df = pd.read_csv("c:\\Users\\Teju\\Downloads\\churn_boosting.csv")

le = LabelEncoder()
for col in ['ContractType', 'InternetService']:
    df[col] = le.fit_transform(df[col])

# Add noise to numerical columns only
np.random.seed(0)
num_cols = ['Tenure', 'MonthlyCharges']
df[num_cols] = df[num_cols] + np.random.normal(0, 3, df[num_cols].shape)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2),
                         n_estimators=50, random_state=42)
ada.fit(X_train, y_train)

gbc = GradientBoostingClassifier(n_estimators=50, max_depth=2, random_state=42)
gbc.fit(X_train, y_train)

fpr_a, tpr_a, _ = roc_curve(y_test, ada.predict_proba(X_test)[:, 1])
fpr_g, tpr_g, _ = roc_curve(y_test, gbc.predict_proba(X_test)[:, 1])
auc_a = auc(fpr_a, tpr_a)
auc_g = auc(fpr_g, tpr_g)

print(f"AdaBoost AUC         : {auc_a:.4f}")
print(f"GradientBoosting AUC : {auc_g:.4f}")

plt.figure(figsize=(6, 5))
plt.plot(fpr_a, tpr_a, label=f'AdaBoost (AUC={auc_a:.3f})', lw=2)
plt.plot(fpr_g, tpr_g, label=f'GradientBoost (AUC={auc_g:.3f})', lw=2)
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
feat_imp = gbc.feature_importances_
feat_names = X.columns.tolist()
idx = np.argsort(feat_imp)
plt.barh([feat_names[i] for i in idx], feat_imp[idx])
plt.title('Feature Importance - GradientBoosting')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()