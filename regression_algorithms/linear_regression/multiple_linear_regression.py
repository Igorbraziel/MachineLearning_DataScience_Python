import pandas as pd

from sklearn.model_selection import train_test_split

from pathlib import Path

house_prices_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'house_prices.csv'

house_prices_base = pd.read_csv(house_prices_path)

print(house_prices_base)

X_houses_price = house_prices_base.iloc[:, 3:19].values
y_houses_price = house_prices_base.iloc[:, 2].values

print(X_houses_price.shape, y_houses_price.shape)

# train and test

X_training, X_test, y_training, y_test = train_test_split(X_houses_price, y_houses_price, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(X_training, y_training)

print('b0:', multiple_linear_regression.intercept_)
print('coefficients:', multiple_linear_regression.coef_, 'len:', len(multiple_linear_regression.coef_))

print('Score Training:', multiple_linear_regression.score(X_training, y_training))
print('Score Test:', multiple_linear_regression.score(X_test, y_test))

previsions = multiple_linear_regression.predict(X_test)

from sklearn.metrics import mean_absolute_error

print('MAE:', mean_absolute_error(y_test, previsions))