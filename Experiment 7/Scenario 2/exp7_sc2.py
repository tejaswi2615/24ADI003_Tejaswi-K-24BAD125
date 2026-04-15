# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

print("Tejaswi K - 24BAD125")

# Load dataset
data = pd.read_csv("c:\\Users\\Teju\\Downloads\\Mall_Customers.csv")

# Select features and scale
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
X_scaled = StandardScaler().fit_transform(X)

aic, bic = [], []
for k in range(1, 11):
    model = GaussianMixture(n_components=k, random_state=42).fit(X_scaled)
    aic.append(model.aic(X_scaled))
    bic.append(model.bic(X_scaled))

plt.plot(range(1, 11), aic, marker='o', label='AIC')
plt.plot(range(1, 11), bic, marker='s', label='BIC')
plt.xlabel("Components")
plt.ylabel("Score")
plt.title("AIC & BIC")
plt.legend()
plt.show()

gmm = GaussianMixture(n_components=5, random_state=42).fit(X_scaled)

probs = gmm.predict_proba(X_scaled)
labels = gmm.predict(X_scaled)
data['GMM_Cluster'] = labels

prob_df = pd.DataFrame(probs, columns=[f'C{i}' for i in range(5)])
prob_df['Cluster'] = labels
print("\nSoft Clustering (first 10 rows):")
print(prob_df.head(10))

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.title("GMM Clustering")
plt.xlabel("Income (scaled)")
plt.ylabel("Score (scaled)")
plt.show()

x, y = np.meshgrid(
    np.linspace(X_scaled[:, 0].min()-0.5, X_scaled[:, 0].max()+0.5, 300),
    np.linspace(X_scaled[:, 1].min()-0.5, X_scaled[:, 1].max()+0.5, 300)
)
grid = np.c_[x.ravel(), y.ravel()]
z = -gmm.score_samples(grid).reshape(x.shape)

plt.contourf(x, y, z, levels=20, alpha=0.4)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], marker='X', s=200)
plt.title("GMM Contours")
plt.xlabel("Income")
plt.ylabel("Score")
plt.show()


for i in range(5):
    plt.figure()
    plt.hist(probs[:, i], bins=20)
    plt.title(f"Cluster {i} Probability Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.show()

sil = silhouette_score(X_scaled, labels)
print("\n--- Evaluation ---")
print("Log-Likelihood:", gmm.score(X_scaled))
print("AIC:", gmm.aic(X_scaled))
print("BIC:", gmm.bic(X_scaled))
print("Silhouette:", sil)

# ---------- K-Means Comparison (Separate Plots) ----------
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
k_labels = kmeans.fit_predict(X_scaled)
k_sil = silhouette_score(X_scaled, k_labels)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.title(f"GMM Clustering (Silhouette: {sil:.2f})")
plt.xlabel("Income")
plt.ylabel("Score")
plt.show()

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=k_labels)
plt.title(f"K-Means Clustering (Silhouette: {k_sil:.2f})")
plt.xlabel("Income")
plt.ylabel("Score")
plt.show()

print("\nCluster Means:")
print(data.groupby('GMM_Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2))