print("TEjaswi K - 24BAD125")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. & 2. Load and Prepare Dataset
# Note: Ensure the file path is correct for your local environment
try:
    df = pd.read_csv("c:\\Users\\Krishnaraj\\Downloads\\spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
except FileNotFoundError:
    print("Error: Dataset file not found. Please check the path.")

# 3. Data Preprocessing
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

df['cleaned_message'] = df['message'].apply(clean_text)

# 4. & 5. Feature Extraction & Label Encoding
# Using TF-IDF with stop_words removal as per instructions
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned_message'])

encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])  # ham=0, spam=1

# 6. Train-Test Split (80/20)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42
)

# 7. & 8. Train Model & Predict
model = MultinomialNB(alpha=1.0)  # Laplace smoothing alpha=1
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 9. Performance Evaluation
print("\n--- Model Performance ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

# 10. Analyze Misclassified Examples
# We use indices_test to map predictions back to the original dataframe
misclassified_indices = indices_test[y_test != y_pred]
print("\n--- Some Misclassified Messages ---")
if len(misclassified_indices) > 0:
    print(df.loc[misclassified_indices, ['label', 'message']].head())
else:
    print("No misclassifications found!")

# 11. Impact of Laplace Smoothing
model_no_smoothing = MultinomialNB(alpha=1e-10) # Using a tiny alpha to avoid log(0)
model_no_smoothing.fit(X_train, y_train)
y_pred_no_smooth = model_no_smoothing.predict(X_test)
print(f"\nAccuracy without Laplace smoothing: {accuracy_score(y_test, y_pred_no_smooth):.4f}")

# --- Visualizations ---

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks([0.5, 1.5], ['Ham', 'Spam'])
plt.yticks([0.5, 1.5], ['Ham', 'Spam'])
plt.show()

# Feature Importance (Top Spam Words)
feature_names = vectorizer.get_feature_names_out()
spam_probs = model.feature_log_prob_[1]
top_indices = np.argsort(spam_probs)[-15:]

plt.figure(figsize=(8, 6))
plt.barh(feature_names[top_indices], spam_probs[top_indices], color='salmon')
plt.title("Top 15 Words Influencing Spam Classification (Log Prob)")
plt.xlabel("Log Probability")
plt.show()