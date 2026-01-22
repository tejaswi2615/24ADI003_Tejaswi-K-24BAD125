import pandas as pd
import matplotlib.pyplot as plt

print("Tejaswi - 24BAD125")
df_bank = pd.read_csv("marketing_campaign.csv", sep='\t')

print(df_bank.head())
print(df_bank.info())

print(df_bank.isnull().sum())

df_bank['Age'] = 2026 - df_bank['Year_Birth']

plt.hist(df_bank['Age'], bins=20)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.boxplot(df_bank['Income'].dropna())
plt.title("Income Distribution")
plt.ylabel("Income")
plt.show()
