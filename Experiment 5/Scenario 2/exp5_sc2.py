import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Tejaswi K - 24BAD125")
print("Decision Tree Classification - Loan Prediction")

df = pd.read_csv("c:\\Users\\Krishnaraj\\Downloads\\train_u6lujuX_CVtuZ9i (1).csv")  
print("\nFirst 5 rows:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())

# Drop Loan_ID if present
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)

# Fill missing numerical values with median
df['ApplicantIncome'].fillna(df['ApplicantIncome'].median(), inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

# Fill missing categorical values with mode
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['Education'].fillna(df['Education'].mode()[0], inplace=True)
df['Property_Area'].fillna(df['Property_Area'].mode()[0], inplace=True)

# Select required features
features = ['ApplicantIncome', 'LoanAmount',
            'Credit_History', 'Education', 'Property_Area']

X = df[features]
y = df['Loan_Status']

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)


depth_values = range(1, 16)
train_acc = []
test_acc = []

for depth in depth_values:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    
    train_acc.append(accuracy_score(y_train, model.predict(X_train)))
    test_acc.append(accuracy_score(y_test, model.predict(X_test)))

# Plot Depth vs Accuracy
plt.figure()
plt.plot(depth_values, train_acc, marker='o', label='Train Accuracy')
plt.plot(depth_values, test_acc, marker='o', label='Test Accuracy')
plt.xlabel("Tree Depth")
plt.ylabel("Accuracy")
plt.title("Depth vs Accuracy (Overfitting Analysis)")
plt.legend()
plt.show()

# Best depth (based on test accuracy)
best_depth = depth_values[np.argmax(test_acc)]
print("\nBest Depth:", best_depth)

# Train final pruned model
dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)


print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


importance = dt.feature_importances_
feature_names = X.columns

plt.figure()
plt.barh(feature_names, importance)
plt.xlabel("Importance Score")
plt.title("Feature Importance")
plt.show()


print("\nFinal Train Accuracy:", accuracy_score(y_train, dt.predict(X_train)))
print("Final Test Accuracy:", accuracy_score(y_test, y_pred))

if accuracy_score(y_train, dt.predict(X_train)) > accuracy_score(y_test, y_pred):
    print("Model may be overfitting.")
else:
    print("No significant overfitting detected.")


shallow_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
deep_tree = DecisionTreeClassifier(random_state=42)

shallow_tree.fit(X_train, y_train)
deep_tree.fit(X_train, y_train)

print("\nShallow Tree Accuracy:",
      accuracy_score(y_test, shallow_tree.predict(X_test)))

print("Deep Tree Accuracy:",
      accuracy_score(y_test, deep_tree.predict(X_test)))


plt.figure(figsize=(15, 8))
plot_tree(dt,
          feature_names=X.columns,
          class_names=le.classes_,
          filled=True)
plt.title("Decision Tree Structure")
plt.show()