import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

house_prices_path = Path(__file__).parent.parent / 'Base_de_dados' / 'house_prices.csv'

house_prices_base = pd.read_csv(house_prices_path)

print(house_prices_base.describe())
print(house_prices_base.isnull().sum())
print(house_prices_base.corr(numeric_only=True))

figure = plt.figure(figsize=(20, 20))
sns.heatmap(house_prices_base.corr(numeric_only=True), annot=True)
plt.show()

print(house_prices_base.shape)

X_houses_price = house_prices_base.iloc[:, 5:6].values
y_houses_price = house_prices_base.iloc[:, 2].values

print(X_houses_price, y_houses_price)
