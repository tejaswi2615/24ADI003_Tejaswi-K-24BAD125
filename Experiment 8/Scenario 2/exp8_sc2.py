# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("Tejaswi - 24BAD125")

# 2. Load dataset
data = pd.read_csv("c:\\Users\\Teju\\Downloads\\Mall_Customers.csv")

print(data.head())

# 3. Select numerical features
X = data[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]

# 4. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Apply PCA (fit all components first)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 6. Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("\nExplained Variance Ratio:")
print(explained_variance)

# 7. Cumulative variance
cumulative_variance = np.cumsum(explained_variance)
print("\nCumulative Variance:")
print(cumulative_variance)

# ---------- Scree Plot ----------
plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o')
plt.xlabel("Principal Components")
plt.ylabel("Variance")
plt.title("Scree Plot")
plt.show()

# ---------- Cumulative Variance Plot ----------
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Variance")
plt.title("Cumulative Variance")
plt.show()

# 8. Reduce to 2D
pca_2 = PCA(n_components=2)
X_reduced = pca_2.fit_transform(X_scaled)

# Convert to DataFrame
pca_df = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])

print("\nReduced Data (first 5 rows):")
print(pca_df.head())

plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - 2D Visualization")
plt.show()