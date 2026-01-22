import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("TejaswiK - 24BAD125")
df_house = pd.read_csv("housing.csv")

print(df_house.head())
print(df_house.info())

print(df_house.isnull().sum())

plt.scatter(df_house['area'], df_house['price'])
plt.title("Area vs Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df_house.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
