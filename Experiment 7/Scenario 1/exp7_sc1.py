import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

print("Tejaswi K - 24BAD125")

# Load the dataset
data = pd.read_csv("c:\\Users\\Teju\\Downloads\\Mall_Customers.csv")

# Display first few rows
print(data.head())

#Data preprocessing
# Check missing values
print(data.isnull().sum())
#Select relevant features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Elbow Method to find optimal K
inertia_values = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(range(1, 11), inertia_values, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters= 5 , random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Assign cluster labels
data['Cluster'] = clusters

#Visualize clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, marker='X')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')
plt.show()

# Evaluation Metrics
print("Inertia:", kmeans.inertia_)

sil_score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", sil_score)

# 10. Interpret clusters
print("\nCluster-wise mean values:")
print(data.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())