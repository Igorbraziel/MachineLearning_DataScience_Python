import pandas as pd
import plotly.express as px

from pathlib import Path

house_prices_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'house_prices.csv'

house_prices_base = pd.read_csv(house_prices_path)

print(house_prices_base)

X_house_prices = house_prices_base.iloc[:, 3:19].values
y_house_prices = house_prices_base.iloc[:, 2].values

print(X_house_prices.shape, y_house_prices.shape)

from sklearn.model_selection import train_test_split

X_training, X_test, y_training, y_test = train_test_split(X_house_prices, y_house_prices, test_size=0.3, random_state=0)

# Stadardisation
from sklearn.preprocessing import StandardScaler

X_training_scaler = StandardScaler()
y_training_scaler = StandardScaler()

X_test_scaler = StandardScaler()
y_test_scaler = StandardScaler()

X_training = X_training_scaler.fit_transform(X_training)
y_training = y_training_scaler.fit_transform(y_training.reshape(-1, 1)).ravel()

X_test = X_test_scaler.fit_transform(X_test)
y_test = y_test_scaler.fit_transform(y_test.reshape(-1, 1)).ravel()

from sklearn.neural_network import MLPRegressor

neural_network_regressor = MLPRegressor()
neural_network_regressor.fit(X_training, y_training)

print('Score (training):', neural_network_regressor.score(X_training, y_training))
print('Score (test):', neural_network_regressor.score(X_test, y_test))

previsions = neural_network_regressor.predict(X_test)

y_test = y_test_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
previsions = y_test_scaler.inverse_transform(previsions.reshape(-1, 1)).ravel()

from sklearn.metrics import mean_absolute_error

print('Mean Absolute Error:', mean_absolute_error(y_test, previsions))