import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Tejaswi K - 24BAD125")

# Load dataset
df = pd.read_csv("c:\\Users\\Teju\\Downloads\\ratings.csv")

# Create User-Item Matrix
user_item_matrix = df.pivot_table(index='userId', columns='movieId', values='rating')
print("\nUser-Item Matrix shape:", user_item_matrix.shape)

# Mean centering
user_mean = user_item_matrix.mean(axis=1)
user_item_centered = user_item_matrix.sub(user_mean, axis=0)

# Fill missing
user_item_filled = user_item_centered.fillna(0)

# Cosine similarity
user_similarity = cosine_similarity(user_item_filled)
user_similarity_df = pd.DataFrame(user_similarity,
                                 index=user_item_matrix.index,
                                 columns=user_item_matrix.index)

# ⭐ Strong filtering (IMPORTANT)
def get_top_n_users(user_id, sim_df, n=10, threshold=0.3):
    sims = sim_df[user_id].sort_values(ascending=False).iloc[1:]
    sims = sims[sims > threshold]
    return sims.head(n)

print("\nTop similar users for user 1:")
print(get_top_n_users(1, user_similarity_df))

# ⭐ Improved prediction with regularization
def predict_rating(user_id, movie_id, similar_users, matrix, mean, reg=15):
    ratings = matrix.loc[similar_users.index, movie_id]
    sims = similar_users[ratings != 0]
    ratings = ratings[ratings != 0]

    if len(ratings) == 0:
        return mean[user_id]

    numerator = np.dot(sims, ratings)
    denominator = sims.sum() + reg  # regularization

    pred = mean[user_id] + (numerator / denominator)
    return float(np.clip(pred, 0.5, 5.0))

# Recommend movies
def recommend_movies(user_id, n=5):
    similar_users = get_top_n_users(user_id, user_similarity_df)
    unseen = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id].isna()].index

    predictions = {}
    for movie in unseen:
        pred = predict_rating(user_id, movie, similar_users,
                              user_item_filled, user_mean)
        predictions[movie] = pred

    return sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]

print("\nTop Recommendations for user 1:")
recommendations = recommend_movies(1)
print(recommendations)

# ⭐ CORRECT EVALUATION (NO DATA LEAKAGE)
y_true, y_pred = [], []

for user in user_item_matrix.index[:50]:  # more users = better stability
    rated = user_item_matrix.loc[user].dropna()
    if len(rated) < 10:
        continue

    split = int(len(rated) * 0.8)
    train_movies = rated.index[:split]
    test_movies = rated.index[split:]

    # Build temp matrix
    temp_matrix = user_item_matrix.copy()

    # Hide test ratings
    temp_matrix.loc[user, test_movies] = np.nan

    # Recompute mean + centering
    temp_mean = temp_matrix.mean(axis=1)
    temp_centered = temp_matrix.sub(temp_mean, axis=0).fillna(0)

    # Similarity
    temp_sim = cosine_similarity(temp_centered)
    temp_sim_df = pd.DataFrame(temp_sim,
                               index=temp_matrix.index,
                               columns=temp_matrix.index)

    similar_users = get_top_n_users(user, temp_sim_df)

    if similar_users.empty:
        continue

    for movie in test_movies:
        pred = predict_rating(user, movie, similar_users,
                              temp_centered, temp_mean)

        y_true.append(rated[movie])
        y_pred.append(pred)

# Metrics
if y_true:
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_val = mean_absolute_error(y_true, y_pred)

    print("\nEvaluation Metrics:")
    print("RMSE:", round(rmse_val, 4))
    print("MAE:", round(mae_val, 4))
else:
    print("Not enough data")

# Sparsity
sparsity = 1 - (user_item_matrix.count().sum() / user_item_matrix.size)
print("\nMatrix Sparsity:", round(sparsity, 4))

# Visualizations
plt.figure(figsize=(10, 6))
sns.heatmap(user_item_matrix.fillna(0).iloc[:20, :20], cmap='coolwarm')
plt.title("User-Item Matrix Heatmap")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(user_similarity_df.iloc[:20, :20], cmap='viridis')
plt.title("User Similarity Heatmap")
plt.show()

movies = [str(r[0]) for r in recommendations]
ratings = [round(r[1], 2) for r in recommendations]

plt.figure(figsize=(8, 5))
plt.bar(movies, ratings)
plt.title("Top Recommended Movies")
plt.xlabel("Movie ID")
plt.ylabel("Predicted Rating")
plt.show()