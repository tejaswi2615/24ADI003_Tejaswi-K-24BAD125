import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Tejasswi - 24BAD125")

data = pd.read_csv("c:\\Users\\Teju\\Downloads\\ratings.csv")
data = data[['userId', 'movieId', 'rating']]
data = data[(data['rating'] >= 1) & (data['rating'] <= 5)]

print("Dataset shape:", data.shape)
print(data.head())

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

all_users  = data['userId'].unique()
all_movies = data['movieId'].unique()

user_index  = {u: i for i, u in enumerate(all_users)}
movie_index = {m: i for i, m in enumerate(all_movies)}

n_users  = len(all_users)
n_movies = len(all_movies)

row = train_data['userId'].map(user_index)
col = train_data['movieId'].map(movie_index)
val = train_data['rating'].values

train_matrix = csr_matrix((val, (row, col)), shape=(n_users, n_movies), dtype=np.float64)

user_means = np.array(train_matrix.sum(axis=1)).flatten() / \
             np.array((train_matrix > 0).sum(axis=1)).flatten()

train_matrix_norm = train_matrix.copy().astype(np.float64)
cx = train_matrix_norm.tocoo()
for i, j, v in zip(cx.row, cx.col, cx.data):
    train_matrix_norm[i, j] = v - user_means[i]

k = 50
U, sigma, VT = svds(train_matrix_norm, k=k)
sigma = np.diag(sigma)

reconstructed = np.dot(np.dot(U, sigma), VT) + user_means.reshape(-1, 1)
reconstructed = np.clip(reconstructed, 1, 5)

test_row  = test_data['userId'].map(user_index)
test_col  = test_data['movieId'].map(movie_index)

valid_mask = test_row.notna() & test_col.notna()
test_data  = test_data[valid_mask]
test_row   = test_row[valid_mask].astype(int)
test_col   = test_col[valid_mask].astype(int)

actual_vals = test_data['rating'].values
pred_vals   = reconstructed[test_row, test_col]

rmse = np.sqrt(mean_squared_error(actual_vals, pred_vals))
mae  = mean_absolute_error(actual_vals, pred_vals)

print("\nRMSE:", rmse)
print("MAE :", mae)

index_user  = {i: u for u, i in user_index.items()}
index_movie = {i: m for m, i in movie_index.items()}

def recommend_movies(user_id, n=5):
    if user_id not in user_index:
        return "User not found"
    u_idx = user_index[user_id]
    rated = set(train_data[train_data['userId'] == user_id]['movieId'])
    preds = {
        index_movie[i]: reconstructed[u_idx, i]
        for i in range(n_movies)
        if index_movie[i] not in rated
    }
    top_n = pd.Series(preds).sort_values(ascending=False).head(n)
    top_n.index.name = 'movieId'
    return top_n

user_id = all_users[0]
print(f"\nTop recommendations for User {user_id}:")
print(recommend_movies(user_id, n=5))

original_sample = pd.DataFrame(
    train_matrix.toarray()[:10, :10],
    index=all_users[:10],
    columns=all_movies[:10]
)
reconstructed_sample = pd.DataFrame(
    reconstructed[:10, :10],
    index=all_users[:10],
    columns=all_movies[:10]
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(original_sample, ax=axes[0], cmap='coolwarm', vmin=0, vmax=5,
            linewidths=0.5, linecolor='white')
axes[0].set_title("Original User-Item Matrix (Sample)")
axes[0].set_xlabel("movieId")
axes[0].set_ylabel("userId")

sns.heatmap(reconstructed_sample, ax=axes[1], cmap='coolwarm', vmin=1, vmax=5,
            linewidths=0.5, linecolor='white')
axes[1].set_title("Reconstructed Matrix (Sample)")
axes[1].set_xlabel("movieId")
axes[1].set_ylabel("userId")

plt.tight_layout()
plt.savefig("heatmap_comparison.png", dpi=150)
plt.show()

k_values    = [10, 20, 50, 100]
rmse_values = []

for k_val in k_values:
    U_k, s_k, VT_k = svds(train_matrix_norm, k=k_val)
    recon_k = np.dot(np.dot(U_k, np.diag(s_k)), VT_k) + user_means.reshape(-1, 1)
    recon_k = np.clip(recon_k, 1, 5)
    pred_k  = recon_k[test_row, test_col]
    rmse_k  = np.sqrt(mean_squared_error(actual_vals, pred_k))
    rmse_values.append(rmse_k)
    print(f"  k={k_val:3d}  RMSE={rmse_k:.4f}")

plt.figure(figsize=(7, 4))
plt.plot(k_values, rmse_values, marker='o', color='steelblue', linewidth=2)
plt.xlabel("Latent Factors (k)")
plt.ylabel("RMSE")
plt.title("RMSE vs Latent Factors")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("rmse_vs_k.png", dpi=150)
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(actual_vals, pred_vals, alpha=0.15, s=8, color='steelblue')
plt.plot([1, 5], [1, 5], color='red', linewidth=1.5, linestyle='--', label='Perfect prediction')
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Ratings")
plt.xlim(0.5, 5.5)
plt.ylim(0.5, 5.5)
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()