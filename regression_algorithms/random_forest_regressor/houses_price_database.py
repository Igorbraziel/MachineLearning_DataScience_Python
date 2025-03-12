import pandas as pd

from pathlib import Path

house_prices_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'house_prices.csv'

house_prices_base = pd.read_csv(house_prices_path)

print(house_prices_base)

X_house_prices = house_prices_base.iloc[:, 3:19].values
y_house_prices = house_prices_base.iloc[:, 2].values

print(X_house_prices.shape, y_house_prices.shape)

from sklearn.model_selection import train_test_split

X_training, X_test, y_training, y_test = train_test_split(X_house_prices, y_house_prices, test_size=0.3, random_state=0)

from sklearn.ensemble import RandomForestRegressor

random_forest_regressor = RandomForestRegressor(n_estimators=100)
random_forest_regressor.fit(X_training, y_training)

previsions = random_forest_regressor.predict(X_test)

print('Score (training):', random_forest_regressor.score(X_training, y_training))
print('Score (test):', random_forest_regressor.score(X_test, y_test))

from sklearn.metrics import mean_absolute_error

print('MAE:', mean_absolute_error(y_test, previsions))
