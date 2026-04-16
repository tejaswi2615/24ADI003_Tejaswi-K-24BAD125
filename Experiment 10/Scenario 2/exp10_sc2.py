import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

print("Tejasswi - 24BAD125")

data = pd.read_csv("c:\\Users\\Teju\\Downloads\\ratings.csv")
data = data[['userId', 'movieId', 'rating']]
data = data[(data['rating'] >= 1) & (data['rating'] <= 5)]

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_matrix = train_data.pivot_table(index='userId', columns='movieId', values='rating')
all_users = train_matrix.index
all_movies = train_matrix.columns

rating_mean = train_matrix.stack().mean()
train_filled = train_matrix.fillna(rating_mean)

nmf_model = NMF(n_components=15, max_iter=500, random_state=42)
W = nmf_model.fit_transform(train_filled)
H = nmf_model.components_

nmf_predicted = np.clip(np.dot(W, H), 1, 5)
nmf_pred_df = pd.DataFrame(nmf_predicted, index=all_users, columns=all_movies)

svd_model = TruncatedSVD(n_components=15, random_state=42)
W_svd = svd_model.fit_transform(train_filled)
svd_predicted = np.clip(np.dot(W_svd, svd_model.components_), 1, 5)
svd_pred_df = pd.DataFrame(svd_predicted, index=all_users, columns=all_movies)

test_data_filtered = test_data[
    test_data['userId'].isin(all_users) & test_data['movieId'].isin(all_movies)
]

nmf_preds_test = [
    nmf_pred_df.loc[row.userId, row.movieId]
    for row in test_data_filtered.itertuples()
]
svd_preds_test = [
    svd_pred_df.loc[row.userId, row.movieId]
    for row in test_data_filtered.itertuples()
]
actual_test = test_data_filtered['rating'].values

nmf_rmse = np.sqrt(mean_squared_error(actual_test, nmf_preds_test))
svd_rmse = np.sqrt(mean_squared_error(actual_test, svd_preds_test))

print(f"NMF RMSE: {nmf_rmse:.4f}")
print(f"SVD RMSE: {svd_rmse:.4f}")

def precision_recall_at_k(pred_df, train_matrix, user_id, k=5, threshold=3.5):
    if user_id not in pred_df.index:
        return 0, 0

    user_train_ratings = train_matrix.loc[user_id]
    unrated_movies = user_train_ratings[user_train_ratings.isna()].index
    user_pred = pred_df.loc[user_id, unrated_movies]
    top_k = user_pred.sort_values(ascending=False).head(k).index

    user_all_rated = data[data['userId'] == user_id]
    relevant_movies = set(user_all_rated[user_all_rated['rating'] >= threshold]['movieId'])

    hits = len(set(top_k) & relevant_movies)
    precision = hits / k
    recall = hits / len(relevant_movies) if len(relevant_movies) > 0 else 0
    return precision, recall

nmf_precisions, nmf_recalls = [], []
svd_precisions, svd_recalls = [], []

for uid in nmf_pred_df.index[:50]:
    p_nmf, r_nmf = precision_recall_at_k(nmf_pred_df, train_matrix, uid)
    p_svd, r_svd = precision_recall_at_k(svd_pred_df, train_matrix, uid)
    nmf_precisions.append(p_nmf)
    nmf_recalls.append(r_nmf)
    svd_precisions.append(p_svd)
    svd_recalls.append(r_svd)

print(f"\nNMF  Precision@5: {np.mean(nmf_precisions):.4f} | Recall@5: {np.mean(nmf_recalls):.4f}")
print(f"SVD  Precision@5: {np.mean(svd_precisions):.4f} | Recall@5: {np.mean(svd_recalls):.4f}")

user_id = train_matrix.index[0]
user_train_ratings = train_matrix.loc[user_id]
unrated = user_train_ratings[user_train_ratings.isna()].index
recommendations = nmf_pred_df.loc[user_id, unrated].sort_values(ascending=False).head(5)

print(f"\nTop 5 Recommendations for User {user_id}:")
print(recommendations)

plt.figure(figsize=(6, 5))
plt.imshow(W[:10, :10], aspect='auto')
plt.title("User Features (NMF)")
plt.xlabel("Latent Factors")
plt.ylabel("Users")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(nmf_pred_df.iloc[:10, :10], aspect='auto')
plt.title("Reconstructed Matrix (NMF)")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.colorbar()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
recommendations.sort_values().plot(kind='barh')
plt.title("Top 5 Recommendations")
plt.xlabel("Predicted Rating")
plt.ylabel("Movie ID")
plt.tight_layout()
plt.show()

metrics = {
    'Model': ['NMF', 'SVD'],
    'RMSE': [nmf_rmse, svd_rmse],
    'Precision@5': [np.mean(nmf_precisions), np.mean(svd_precisions)],
    'Recall@5': [np.mean(nmf_recalls), np.mean(svd_recalls)]
}
metrics_df = pd.DataFrame(metrics)
print("\nModel Comparison:")
print(metrics_df.to_string(index=False))