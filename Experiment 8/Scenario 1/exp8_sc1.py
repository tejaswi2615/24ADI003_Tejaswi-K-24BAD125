# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

print("Tejasswi - 24BAD125")

# 2. Load dataset
data = pd.read_csv("c:\\Users\\Teju\\Downloads\\Groceries_dataset.csv")

print(data.head())

# 3. Convert data into transactions
transactions = data.groupby('Member_number')['itemDescription'].apply(list).values

# One-hot encoding
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_data, columns=te.columns_)

# 4. Apply Apriori (set reasonable support)
frequent_items = apriori(df, min_support=0.02, use_colnames=True)

# Sort for clarity
frequent_items = frequent_items.sort_values(by='support', ascending=False)

print("\nFrequent Itemsets:")
print(frequent_items.head())

# 5. Generate association rules
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.3)

# 6. Filter strong rules
rules = rules[(rules['lift'] > 1) & (rules['confidence'] >= 0.4)]

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# 7. Visualization

# Bar chart of top itemsets
frequent_items.head(10).plot(
    x='itemsets', y='support', kind='bar'
)
plt.title("Top Frequent Itemsets")
plt.xlabel("Itemsets")
plt.ylabel("Support")
plt.show()

# Support vs Confidence scatter
plt.scatter(rules['support'], rules['confidence'])
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")
plt.show()

# 8. Interpretation (cluster-like summary)
print("\nTop Rules Interpretation:")
for i, row in rules.head(5).iterrows():
    print(f"If {set(row['antecedents'])} → {set(row['consequents'])}")
    print(f"Support: {row['support']:.2f}, Confidence: {row['confidence']:.2f}, Lift: {row['lift']:.2f}\n")

import networkx as nx

# Take only top rules to avoid clutter
top_rules = rules.sort_values(by='lift', ascending=False).head(10)

# Create directed graph
G = nx.DiGraph()

# Add edges
for _, row in top_rules.iterrows():
    for antecedent in row['antecedents']:
        for consequent in row['consequents']:
            G.add_edge(antecedent, consequent, weight=row['lift'])

# Draw graph
plt.figure(figsize=(8,6))
pos = nx.spring_layout(G, k=0.5)

nx.draw(G, pos, with_labels=True, node_size=2500, node_color='lightgreen', font_size=10)

# Edge labels (lift values)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Network Graph of Association Rules (Top Rules)")
plt.show()