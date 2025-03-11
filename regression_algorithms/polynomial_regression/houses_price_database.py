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

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)

X_training_poly = poly.fit_transform(X_training)
X_test_poly = poly.transform(X_test)

from sklearn.linear_model import LinearRegression

polynomial_regression = LinearRegression()
polynomial_regression.fit(X_training_poly, y_training)

print(polynomial_regression.score(X_training_poly, y_training))
print(polynomial_regression.score(X_test_poly, y_test))

previsions = polynomial_regression.predict(X_test_poly)

from sklearn.metrics import mean_absolute_error

print('MAE:', mean_absolute_error(y_test, previsions))





