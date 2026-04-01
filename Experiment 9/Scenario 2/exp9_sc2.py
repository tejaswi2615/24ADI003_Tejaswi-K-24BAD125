import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

print("Tejaswi K - 24BAD125")

# Load dataset
df = pd.read_csv("c:\\Users\\Teju\\Downloads\\ratings.csv")

item_user_matrix = df.pivot_table(
    index='movieId', columns='userId', values='rating'
).fillna(0)

print(f"Item-User Matrix Shape: {item_user_matrix.shape}")

item_sim = cosine_similarity(item_user_matrix)
item_sim_df = pd.DataFrame(
    item_sim,
    index=item_user_matrix.index,
    columns=item_user_matrix.index
)

def get_similar_movies(movie_id, n=5):
    return item_sim_df[movie_id].sort_values(ascending=False).iloc[1:n+1]

print("\nTop movies similar to Movie ID 1:")
print(get_similar_movies(1))

def predict_with_tree(target_user, target_movie, n_similar=10):
    similar_ids = get_similar_movies(target_movie, n=n_similar).index

    train_data = item_user_matrix.loc[
        :, item_user_matrix.loc[target_movie] > 0
    ].T

    if train_data.empty:
        return 0

    valid_similar = [m for m in similar_ids if m in train_data.columns]
    if not valid_similar:
        return 0

    X_train = train_data[valid_similar]
    y_train = train_data[target_movie]

    if len(y_train) < 2:
        return 0

    tree = DecisionTreeRegressor(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)

    user_features = item_user_matrix.loc[
        valid_similar, target_user
    ].values.reshape(1, -1)

    pred = tree.predict(user_features)[0]
    return float(np.clip(pred, 0.5, 5.0))

def proper_precision_at_k(user_id, k=5, threshold=3.5,
                         test_frac=0.4, random_state=42):

    user_ratings = df[df['userId'] == user_id].copy()

    if len(user_ratings) < k + 5:
        print("Not enough ratings.")
        return None, [], []

    test = user_ratings.sample(frac=test_frac, random_state=random_state)
    train = user_ratings.drop(test.index)

    seen_movies = set(train['movieId'])
    all_movies = set(df['movieId'].unique())

    unseen_movies = list(all_movies - seen_movies)

    liked_in_test = set(test[test['rating'] >= threshold]['movieId'])

    if not liked_in_test:
        print("No liked movies in test.")
        return None, [], []

    predictions = {}

    for movie in unseen_movies:
        try:
            pred = predict_with_tree(user_id, movie)
            if pred > 0:
                predictions[movie] = pred
        except:
            continue

    print(f"\nTotal predictions made: {len(predictions)}")

    if len(predictions) < k:
        print("Not enough predictions.")
        return None, [], []

    top_k_movies = set(
        m for m, _ in sorted(
            predictions.items(), key=lambda x: x[1], reverse=True
        )[:k]
    )

    hits = len(top_k_movies & liked_in_test)
    precision = hits / k

    return precision, list(top_k_movies), list(liked_in_test)

user_id = 1
user_ratings_u1 = df[df['userId'] == user_id]

y_true_list, y_pred_list = [], []

for _, row in user_ratings_u1.head(20).iterrows():
    try:
        pred = predict_with_tree(user_id, row['movieId'])
        if pred > 0:
            y_true_list.append(row['rating'])
            y_pred_list.append(pred)
    except:
        continue

rmse = np.sqrt(mean_squared_error(y_true_list, y_pred_list))

def user_based_predict(target_user, target_movie, n=20):
    user_item = df.pivot_table(
        index='userId', columns='movieId', values='rating'
    )

    u_mean = user_item.mean(axis=1)
    u_centered = user_item.sub(u_mean, axis=0).fillna(0)

    sim = cosine_similarity(u_centered)
    sim_df = pd.DataFrame(sim,
                          index=u_centered.index,
                          columns=u_centered.index)

    sims = sim_df[target_user].sort_values(ascending=False).iloc[1:]
    sims = sims[sims > 0.1].head(n)

    u_fill = user_item.fillna(0)

    ratings = u_fill.loc[sims.index, target_movie]
    ratings = ratings[ratings > 0]
    sims = sims[ratings.index]

    if sims.sum() == 0:
        return u_mean[target_user]

    neighbor_means = u_mean[sims.index]
    centered = ratings - neighbor_means

    pred = u_mean[target_user] + np.dot(sims, centered) / sims.sum()
    return float(np.clip(pred, 0.5, 5.0))

ub_true, ub_pred = [], []

for _, row in user_ratings_u1.head(20).iterrows():
    try:
        pred = user_based_predict(user_id, row['movieId'])
        if pred > 0:
            ub_true.append(row['rating'])
            ub_pred.append(pred)
    except:
        continue

rmse_ub = np.sqrt(mean_squared_error(ub_true, ub_pred))

k = 5
precision, top_k_movies, liked_movies = proper_precision_at_k(user_id, k=k)

print("\nEvaluation Metrics:")
print("Item-Based RMSE:", round(rmse, 4))
print("User-Based RMSE:", round(rmse_ub, 4))

if precision is not None:
    print(f"Precision@{k}:", round(precision, 2))

print("\nTop-K Recommended Movies:", top_k_movies)
print("Liked Movies (Test):", liked_movies)

print("\nItem Similarity Sample:")
print(item_sim_df.iloc[:5, :5])

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(item_sim_df.iloc[:20, :20], cmap='magma')
plt.title("Item Similarity Heatmap")
plt.show()

# RMSE Comparison
plt.figure(figsize=(8, 5))
methods = ['User-Based', 'Item-Based']
values = [rmse_ub, rmse]
bars = plt.bar(methods, values)

for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height(),
             round(val, 4),
             ha='center')

plt.title("RMSE Comparison")
plt.ylabel("RMSE")
plt.show()

# Top Similar Movies Plot
top_sim_movies = get_similar_movies(1)

print("\nTop Similar Movies Values:")
print(top_sim_movies)

plt.figure(figsize=(8, 5))
top_sim_movies.plot(kind='bar')
plt.title("Top Similar Movies to Movie 1")
plt.show()