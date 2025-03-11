import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

house_prices_path = Path(__file__).parent.parent.parent / 'Base_de_dados' / 'house_prices.csv'

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

from sklearn.model_selection import train_test_split

X_training, X_test, y_training, y_test = train_test_split(X_houses_price, y_houses_price, test_size=0.3, random_state=0)

print(X_training.shape, y_training.shape)
print(X_test.shape, y_test.shape)

from sklearn.linear_model import LinearRegression

simple_linear_regression = LinearRegression()
simple_linear_regression.fit(X_training, y_training)

print(simple_linear_regression.intercept_, simple_linear_regression.coef_)

previsions = simple_linear_regression.predict(X_training)

print(simple_linear_regression.score(X_training, y_training))

import plotly.graph_objects as go
import plotly.express as px

graphic1 = px.scatter(x=X_training.ravel(), y=y_training)
graphic2 = px.line(x=X_training.ravel(), y=previsions.ravel())
graphic2.data[0].line.color = 'red'
graphic3 = go.Figure(data=graphic1.data + graphic2.data)
graphic3.show()

# Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE)
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

previsions = simple_linear_regression.predict(X_test)

print('MAE:', mean_absolute_error(y_test, previsions))
print('MSE:', mean_squared_error(y_test, previsions))
print('RMSE:', root_mean_squared_error(y_test, previsions))

# Scatter Plot
graphic1 = px.scatter(x=X_test.ravel(), y=y_test)
graphic2 = px.line(x=X_test.ravel(), y=previsions)
graphic2.data[0].line.color = 'red'
graphic3 = go.Figure(data=graphic1.data + graphic2.data)
graphic3.show()