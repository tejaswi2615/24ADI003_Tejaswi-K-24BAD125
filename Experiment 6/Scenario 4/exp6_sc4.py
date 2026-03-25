import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("Tejaswi K - 24BAD125")

# Load dataset
df = pd.read_csv("c:\\Users\\Teju\\Downloads\\heart_stacking.csv")

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Base models
estimators = [
    ("lr", LogisticRegression(max_iter=1000)),
    ("dt", DecisionTreeClassifier()),
    ("svm", SVC(probability=True))
]

# Stacking
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack.fit(X_train, y_train)

# Individual models
lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
dt = DecisionTreeClassifier().fit(X_train, y_train)
svm = SVC().fit(X_train, y_train)

# Accuracy
models = ["LR", "DT", "SVM", "Stacking"]
scores = [
    accuracy_score(y_test, lr.predict(X_test)),
    accuracy_score(y_test, dt.predict(X_test)),
    accuracy_score(y_test, svm.predict(X_test)),
    accuracy_score(y_test, stack.predict(X_test))
]

# Plot
plt.bar(models, scores)
plt.title("Model Comparison")
plt.show()