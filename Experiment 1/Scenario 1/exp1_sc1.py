import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

print("Mithra Ravi- 24BAD070")     
# Load dataset
df_ecom = pd.read_csv("data.csv", encoding="ISO-8859-1")

# Inspect dataset
print(df_ecom.head())
print(df_ecom.tail())
print(df_ecom.info())
print(df_ecom.describe())

# Check missing values
print(df_ecom.isnull().sum())

# Create Sales column
df_ecom['Sales'] = df_ecom['Quantity'] * df_ecom['UnitPrice']

#sales per product
product_sales = df_ecom.groupby('Description')['Sales'].sum().head(10)

# Line chart
product_sales.plot(kind='line', marker='o')
plt.title("Sales Trend of Top Products")
plt.xlabel("Product")
plt.ylabel("Sales")
plt.show()

# Bar chart
product_sales.plot(kind='bar')
plt.title("Top 10 Products by Sales")
plt.xlabel("Product")
plt.ylabel("Sales")
plt.show()


