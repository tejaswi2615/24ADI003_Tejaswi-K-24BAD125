import pandas as pd
import matplotlib.pyplot as plt

print("Tejaswi K - 24BAD125")
df_health = pd.read_csv("diabetes.csv")

#Inspect dataset
print(df_health.head())
print(df_health.info())

#missing values
print(df_health.isnull().sum())

#Glucose levels
plt.hist(df_health['Glucose'], bins=20)
plt.title("Glucose Level Distribution")
plt.xlabel("Glucose")
plt.ylabel("Frequency")
plt.show()

#Age
plt.boxplot(df_health['Age'])
plt.title("Age Distribution")
plt.ylabel("Age")
plt.show()

